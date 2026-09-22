import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from transformers import pipeline

load_dotenv()

POLICY_INDEX       = os.getenv("PINECONE_POLICY_INDEX", "policy-rag")
NAMESPACE          = "policies"
EMBED_MODEL        = "openai/text-embedding-3-small"
LLM_MODEL          = "openai/gpt-4o-mini"
INJECTION_MODEL    = "protectai/deberta-v3-base-prompt-injection-v2"
INJECTION_THRESHOLD = 0.85  # if confidence above 0.85 - 1(injected) otherwise it 0 (safe)

st.set_page_config(page_title="Policy RAG (Injection Guard)", page_icon="🛡️", layout="centered")
st.title("🛡️ Policy RAG with Prompt Injection Guard")
st.caption("Queries are screened for prompt injection before reaching the policy index.")


@st.cache_resource(show_spinner="Loading injection detector...")
def load_detector():
    return pipeline("text-classification", model=INJECTION_MODEL)


@st.cache_resource(show_spinner="Connecting to Pinecone & OpenAI...")
def load_clients():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(POLICY_INDEX)
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    return index, client


def is_injection(detector, text: str) -> tuple[bool, float]:
    result = detector(text, truncation=True, max_length=512)[0]
    injected = result["label"].upper() == "INJECTION"
    return injected, round(result["score"], 4)


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


detector = load_detector()
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
        injected, score = is_injection(detector, prompt)

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
