import streamlit as st

st.set_page_config(page_title="HR Agent", page_icon="🤖", layout="centered")
st.title("🤖 HR Agent")
st.caption("Ask about candidates, generate JDs, or query company policies.")
st.header("⚡️ Powered by OpenAI, Pinecone, and LangChain")


@st.cache_resource(show_spinner="Loading agent (first time only)...")
def get_graph():
    from agent import build_graph
    return build_graph()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = get_graph().invoke({
                "messages":        [{"role": "user", "content": prompt}],
                "next":            "",
                "rag_result":      "",
                "jd_result":       "",
                "policy_result":   "",
                "policy_sources":  [],
                "final_answer":    "",
                "_rag_query":      "",
                "_jd_description": "",
            })
            answer = result["final_answer"]
        st.markdown(answer)
        sources = result.get("policy_sources", [])
        if sources:
            with st.expander(f"📄 Sources ({len(sources)} policy chunks)"):
                for src in sources:
                    page_info = f" — Page {src['page']}" if src.get("page") else ""
                    st.caption(f"`{src['id']}` — {src['source']}{page_info}")
                    if src.get("images"):
                        st.caption(f"🖼️ Images referenced: {src['images']}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
