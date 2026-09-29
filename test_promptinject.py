import os
import requests
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

POLICY_INDEX       = os.getenv("PINECONE_POLICY_INDEX", "policy-rag")
NAMESPACE          = "policies"
EMBED_MODEL        = "openai/text-embedding-3-small"
LLM_MODEL          = "openai/gpt-4o-mini"
INJECTION_MODEL     = "protectai/deberta-v3-base-prompt-injection-v2"
INJECTION_API_URL   = f"https://api-inference.huggingface.co/models/{INJECTION_MODEL}"
INJECTION_THRESHOLD = 0.85

st.set_page_config(page_title="Policy RAG (Injection Guard)", page_icon="🛡️", layout="centered")
st.title("🛡️ Policy RAG with Prompt Injection Guard")
st.caption("Queries are screened for prompt injection before reaching the policy index.")


def load_clients():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(POLICY_INDEX)
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    return index, client


def is_injection(text: str) -> tuple[bool, float]:
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    resp = requests.post(INJECTION_API_URL, headers=headers, json={"inputs": text}, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    # API returns [[{label, score}, ...]] or [{label, score}, ...]
    items = results[0] if isinstance(results[0], list) else results
    top = max(items, key=lambda x: x["score"])
    injected = top["label"].upper() == "INJECTION"
    return injected, round(top["score"], 4)


def query_policy_rag(index, client, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    embed_resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
    vector = embed_resp.data[0].embedding

    matches = index.query(vector=vector, top_k=top_k, namespace=NAMESPACE, include_metadata=True)
    chunks = [m.metadata for m in matches.matches]
    context = "\n\n".join(c.get("text", "") for c in chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful HR policy assistant. Answer using only the provided policy context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, chunks


index, client = load_clients()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about company policies..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        injected, score = is_injection(prompt)

        if injected and score >= INJECTION_THRESHOLD:
            answer = f"🚫 **Prompt injection detected** (confidence: `{score}`). Query blocked."
            st.error(answer)
        else:
            if injected:
                st.info(f"ℹ️ Low-confidence injection signal (`{score}`) — proceeding with query.")
            with st.spinner("Querying policy index..."):
                answer, sources = query_policy_rag(index, client, prompt)
            st.markdown(answer)
            if sources:
                with st.expander(f"📄 Sources ({len(sources)} policy chunks)"):
                    for src in sources:
                        page_info = f" — Page {src['page']}" if src.get("page") else ""
                        st.caption(f"`{src.get('id', '')}` — {src.get('source', '')}{page_info}")

    st.session_state.history.append({"role": "assistant", "content": answer})
