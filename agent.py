"""
agent.py — HR Agent with supervisor-controlled LangGraph + Hybrid Search.

Hybrid Search pipeline (rag_node):
  1. Dense retrieval  — Pinecone top-20 (semantic)
  2. Sparse retrieval — BM25 top-20 (keyword / exact skill match)
  3. RRF fusion       — Reciprocal Rank Fusion merges both ranked lists
  4. Cross-encoder    — BAAI/bge-reranker-base re-ranks fused top-20 → top-3

Graph topology:
  START → supervisor ──► rag_node ──► supervisor
                     ──► jd_node  ──► supervisor
                     ──► END
"""
import os, json, operator, pickle
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END

load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
INDEX_NAME         = os.getenv("PINECONE_INDEX", "hr-rag")
NAMESPACE          = "candidates"
POLICY_INDEX       = os.getenv("PINECONE_POLICY_INDEX", "policy-rag")
POLICY_NAMESPACE   = "policies"
LLM_MODEL          = "openai/gpt-4o-mini"
LLM_for_supervisor="openai/gpt-4o-mini"
LLM_for_JD="nvidia/nemotron-3.5-lightning:free"
EMBED_MODEL        = "openai/text-embedding-3-small"
BM25_PATH          = "bm25_index.pkl"
DENSE_TOP_K        = 20   # candidates fetched from Pinecone
BM25_TOP_K         = 20   # candidates fetched from BM25
RRF_K              = 60   # RRF constant (standard value)
FINAL_TOP_K        = 3    # after cross-encoder rerank

# ---------- clients ----------
pc            = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# ---------- load BM25 index (built by ingest.py) ----------
with open(BM25_PATH, "rb") as f:
    _bm25_payload: dict = pickle.load(f)

_bm25: BM25Okapi   = _bm25_payload["bm25"]
_bm25_ids: list    = _bm25_payload["ids"]
_bm25_texts: list  = _bm25_payload["texts"]

# ============================================================
# State
# ============================================================
class AgentState(TypedDict):
    messages:        Annotated[list[dict], operator.add]
    next:            str
    rag_result:      str
    jd_result:       str
    policy_result:   str
    policy_sources:  list[dict]
    final_answer:    str
    _rag_query:      str
    _jd_description: str

# ============================================================
# Supervisor tool schemas
# ============================================================
SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_to_rag",
            "description": (
                "Route to the RAG node to search the candidate database. "
                "Use when the user wants to find, shortlist, or compare candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for candidate retrieval"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_policy",
            "description": (
                "Route to the Policy RAG node to answer company policy questions. "
                "Use when the user asks about HR policies, leave, benefits, code of conduct, or any corporate policy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Policy-related question"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_jd",
            "description": (
                "Route to the JD node to generate a Job Description. "
                "Use when the user wants to create, draft, or write a JD or job posting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role_description": {"type": "string", "description": "Short role description to generate JD from"}
                },
                "required": ["role_description"],
            },
        },
    },
]

# ============================================================
# Hybrid Search helpers
# ============================================================
def embed_query(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def dense_search(vec: list[float], top_k: int) -> list[tuple[str, str]]:
    """Returns [(id, text), ...] from Pinecone dense search."""
    index   = pc.Index(INDEX_NAME)
    results = index.query(vector=vec, top_k=top_k, namespace=NAMESPACE, include_metadata=True)
    return [(m["id"], m["metadata"]["text"]) for m in results["matches"]]

def bm25_search(query: str, top_k: int) -> list[tuple[str, str]]:
    """Returns [(id, text), ...] from local BM25 search."""
    tokens  = query.lower().split()
    scores  = _bm25.get_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(_bm25_ids[i], _bm25_texts[i]) for i in indices]

def reciprocal_rank_fusion(
    dense_hits: list[tuple[str, str]],
    bm25_hits:  list[tuple[str, str]],
    k: int = RRF_K,
) -> list[str]:
    """
    Fuse two ranked lists using RRF.
    score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)
    Returns deduplicated texts sorted by fused score.
    """
    scores: dict[str, float] = {}
    id_to_text: dict[str, str] = {}

    for rank, (cid, text) in enumerate(dense_hits, start=1):
        scores[cid]     = scores.get(cid, 0.0) + 1.0 / (k + rank)
        id_to_text[cid] = text

    for rank, (cid, text) in enumerate(bm25_hits, start=1):
        scores[cid]     = scores.get(cid, 0.0) + 1.0 / (k + rank)
        id_to_text[cid] = text

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [id_to_text[cid] for cid in ranked_ids]

def rerank(query: str, docs: list[str], top_k: int = FINAL_TOP_K) -> list[str]:
    scores = cross_encoder.predict([[query, doc] for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    print(f"  [rerank] Cross-encoder scores: {[round(float(s), 4) for s, _ in ranked]}")
    return [doc for _, doc in ranked[:top_k]]

# ============================================================
# Node 1 — SUPERVISOR
# ============================================================
def supervisor(state: AgentState) -> AgentState:
    print(f"\n[supervisor] Deciding next step...")

    context = ""
    if state.get("rag_result"):
        context += f"\n\nRAG search result:\n{state['rag_result']}"
    if state.get("jd_result"):
        context += f"\n\nJD generation result:\n{state['jd_result']}"
    if state.get("policy_result"):
        context += f"\n\nPolicy RAG result:\n{state['policy_result']}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an HR supervisor agent coordinating three specialist nodes:\n"
                "- route_to_rag: searches the candidate database (hybrid RAG + re-ranking)\n"
                "- route_to_jd: generates a Job Description from a short description\n"
                "- route_to_policy: answers company policy questions (leave, benefits, conduct, HR policies)\n\n"
                "Call the appropriate tool if a task is still pending. "
                "If all required tasks are complete, respond directly with the final "
                "synthesized answer WITHOUT calling any tool."
            ),
        },
        {"role": "user", "content": state["messages"][0]["content"]},
    ]
    if context:
        messages.append({"role": "assistant", "content": f"Completed so far:{context}"})

    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=SUPERVISOR_TOOLS,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        tc   = msg.tool_calls[0]
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        if name == "route_to_rag":
            print(f"  [supervisor] → rag_node | query: \"{args['query']}\"")
            return {
                "messages":        [],
                "next":            "rag",
                "rag_result":      "",
                "jd_result":       state.get("jd_result", ""),
                "policy_result":   state.get("policy_result", ""),
                "policy_sources":  state.get("policy_sources", []),
                "final_answer":    "",
                "_rag_query":      args["query"],
                "_jd_description": state.get("_jd_description", ""),
            }

        if name == "route_to_jd":
            print(f"  [supervisor] → jd_node | role: \"{args['role_description']}\"")
            return {
                "messages":        [],
                "next":            "jd",
                "rag_result":      state.get("rag_result", ""),
                "jd_result":       "",
                "policy_result":   state.get("policy_result", ""),
                "policy_sources":  state.get("policy_sources", []),
                "final_answer":    "",
                "_rag_query":      state.get("_rag_query", ""),
                "_jd_description": args["role_description"],
            }

        if name == "route_to_policy":
            print(f"  [supervisor] → policy_rag_node | query: \"{args['query']}\"")
            return {
                "messages":        [],
                "next":            "policy",
                "rag_result":      state.get("rag_result", ""),
                "jd_result":       state.get("jd_result", ""),
                "policy_result":   "",
                "policy_sources":  [],
                "final_answer":    "",
                "_rag_query":      state.get("_rag_query", ""),
                "_jd_description": state.get("_jd_description", ""),
            }

    print(f"  [supervisor] → all tasks done, synthesizing final answer")
    return {
        "messages":        [],
        "next":            "end",
        "rag_result":      state.get("rag_result", ""),
        "jd_result":       state.get("jd_result", ""),
        "policy_result":   state.get("policy_result", ""),
        "policy_sources":  state.get("policy_sources", []),
        "final_answer":    msg.content or "",
        "_rag_query":      state.get("_rag_query", ""),
        "_jd_description": state.get("_jd_description", ""),
    }

# ============================================================
# Node 2 — RAG NODE  (Hybrid: Dense + BM25 → RRF → CrossEncoder)
# ============================================================
def rag_node(state: AgentState) -> AgentState:
    query = state.get("_rag_query", "")
    print(f"\n[rag_node] Hybrid search | query: \"{query}\"")

    # 1. Dense retrieval
    vec         = embed_query(query)
    dense_hits  = dense_search(vec, top_k=DENSE_TOP_K)
    print(f"  [dense]  Top {len(dense_hits)} candidates from Pinecone:")
    for i, (cid, d) in enumerate(dense_hits[:5], 1):
        print(f"    {i}. [{cid}] {d[:75]}...")

    # 2. BM25 sparse retrieval
    bm25_hits   = bm25_search(query, top_k=BM25_TOP_K)
    print(f"  [bm25]   Top {len(bm25_hits)} candidates from BM25:")
    for i, (cid, d) in enumerate(bm25_hits[:5], 1):
        print(f"    {i}. [{cid}] {d[:75]}...")

    # 3. RRF fusion
    fused_docs  = reciprocal_rank_fusion(dense_hits, bm25_hits)
    print(f"  [rrf]    Fused pool: {len(fused_docs)} unique candidates")

    # 4. Cross-encoder rerank → top-3
    top_docs    = rerank(query, fused_docs, top_k=FINAL_TOP_K)
    print(f"  [result] Final top {FINAL_TOP_K} after reranking:")
    for i, d in enumerate(top_docs, 1):
        print(f"    {i}. {d[:85]}...")

    return {
        "messages":        [],
        "next":            "supervisor",
        "rag_result":      "\n---\n".join(top_docs),
        "jd_result":       state.get("jd_result", ""),
        "policy_result":   state.get("policy_result", ""),
        "policy_sources":  state.get("policy_sources", []),
        "final_answer":    "",
        "_rag_query":      query,
        "_jd_description": state.get("_jd_description", ""),
    }

# ============================================================
# Node 3 — JD NODE  (Direct LLM call, no retrieval)
# ============================================================
def jd_node(state: AgentState) -> AgentState:
    role_description = state.get("_jd_description", "")
    print(f"\n[jd_node] Generating JD for: \"{role_description}\"")

    resp = openai_client.chat.completions.create(
        model=LLM_for_JD,
        messages=[{
            "role": "user",
            "content": (
                "You are an expert technical recruiter. "
                "Generate a professional Job Description based on the role description below.\n\n"
                "Sections to include:\n"
                "1. Job Title\n"
                "2. About the Role (2-3 sentences)\n"
                "3. Key Responsibilities (5-7 bullets)\n"
                "4. Required Skills & Qualifications (5-7 bullets)\n"
                "5. Nice to Have (3 bullets)\n"
                "6. What We Offer (3 bullets)\n\n"
                f"Role Description: {role_description}"
            ),
        }],
    )
    jd = resp.choices[0].message.content
    print(f" [jd_node] using {LLM_for_JD} JD generated ({len(jd)} chars)")


    return {
        "messages":        [],
        "next":            "supervisor",
        "rag_result":      state.get("rag_result", ""),
        "jd_result":       jd,
        "policy_result":   state.get("policy_result", ""),
        "policy_sources":  state.get("policy_sources", []),
        "final_answer":    "",
        "_rag_query":      state.get("_rag_query", ""),
        "_jd_description": role_description,
    }

# ============================================================
# Node 4 — POLICY RAG NODE  (Dense search against policy-rag index)
# ============================================================
def policy_rag_node(state: AgentState) -> AgentState:
    query = state["messages"][0]["content"]
    print(f"\n[policy_rag_node] Policy search | query: \"{query}\"")

    vec   = embed_query(query)
    index = pc.Index(POLICY_INDEX)
    results = index.query(vector=vec, top_k=5, namespace=POLICY_NAMESPACE, include_metadata=True)
    docs  = [m["metadata"]["text"] for m in results["matches"]]
    sources = [
        {
            "id": m["id"],
            "source": m["metadata"].get("source", "policy"),
            "page": m["metadata"].get("page", ""),
            "images": m["metadata"].get("images", ""),
        }
        for m in results["matches"]
    ]
    print(f"  [policy_rag_node] Retrieved {len(docs)} policy chunks")

    context = "\n---\n".join(docs)
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an HR policy expert. Answer the question using only the provided policy excerpts."},
            {"role": "user",   "content": f"Policy excerpts:\n{context}\n\nQuestion: {query}"},
        ],
    )
    answer = resp.choices[0].message.content
    print(f"  [policy_rag_node] Answer generated ({len(answer)} chars)")

    return {
        "messages":        [],
        "next":            "supervisor",
        "rag_result":      state.get("rag_result", ""),
        "jd_result":       state.get("jd_result", ""),
        "policy_result":   answer,
        "policy_sources":  sources,
        "final_answer":    "",
        "_rag_query":      state.get("_rag_query", ""),
        "_jd_description": state.get("_jd_description", ""),
    }

# ============================================================
# Conditional edge
# ============================================================
def supervisor_router(state: AgentState) -> Literal["rag_node", "jd_node", "policy_rag_node", "__end__"]:
    route = state.get("next", "end")
    if route == "rag":    return "rag_node"
    if route == "jd":     return "jd_node"
    if route == "policy": return "policy_rag_node"
    return "__end__"

# ============================================================
# Build graph
# ============================================================
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",     supervisor)
    g.add_node("rag_node",       rag_node)
    g.add_node("jd_node",        jd_node)
    g.add_node("policy_rag_node", policy_rag_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor", supervisor_router,
        {"rag_node": "rag_node", "jd_node": "jd_node", "policy_rag_node": "policy_rag_node", "__end__": END},
    )
    g.add_edge("rag_node",        "supervisor")
    g.add_edge("jd_node",         "supervisor")
    g.add_edge("policy_rag_node", "supervisor")
    return g.compile()

# ============================================================
# run()
# ============================================================
def run(user_query: str) -> str:
    print(f"\n{'='*60}\n[user] {user_query}\n{'='*60}")
    result = build_graph().invoke({
        "messages":        [{"role": "user", "content": user_query}],
        "next":            "",
        "rag_result":      "",
        "jd_result":       "",
        "policy_result":   "",
        "final_answer":    "",
        "_rag_query":      "",
        "_jd_description": "",
    })
    print(f"\n{'='*60}\n[final answer]\n{result['final_answer']}\n{'='*60}")
    return result["final_answer"]

if __name__ == "__main__":
    run("Quarterly expense trend in Q1 for corporate travel expense?")
    run("if my working model is hybrid, how many times i have to visit the onsite?")
    # run("Write a JD for a Senior DevOps Engineer with Kubernetes and AWS at a fintech startup")
    # run("I need to hire a computer vision engineer — find matching candidates and write a JD")
