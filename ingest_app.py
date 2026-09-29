"""
ingest_app.py — Streamlit UI for candidate ingestion.
Modes:
  • Add New Profiles Only : skip existing IDs, upsert only new rows
  • Sync (add + remove)   : delete removed candidates, upsert new ones, rebuild BM25
  • Full Re-ingest        : wipe namespace, re-embed & upsert all rows, rebuild BM25
"""
import streamlit as st
from ingest import load_candidates, fit_bm25, pc, INDEX_NAME, NAMESPACE, BATCH_SIZE, EMBED_MODEL
from ingest import openai as oai_client

st.set_page_config(page_title="Ingest Candidates", page_icon="📥", layout="centered")
st.title("📥 Candidate Ingestion")

mode = st.selectbox(
    "Ingestion Mode",
    [
        "Add New Profiles Only",
        "Sync — Add new + Remove deleted",
        "Full Re-ingest (wipe & reload all)",
    ],
)

def fetch_existing_ids() -> set:
    index = pc.Index(INDEX_NAME)
    existing = set()
    try:
        for id_batch in index.list(namespace=NAMESPACE, limit=1000):
            existing.update(id_batch)
    except Exception:
        st.warning("Could not list IDs via index.list() — SDK may be outdated.")
    return existing

def upsert_with_progress(records: list[dict]):
    progress = st.progress(0, text="Upserting to Pinecone...")
    index = pc.Index(INDEX_NAME)
    total = len(records)
    upserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        resp  = oai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs  = [d.embedding for d in resp.data]
        index.upsert(
            vectors=[
                {"id": r["id"], "values": v, "metadata": r["metadata"]}
                for r, v in zip(batch, vecs)
            ],
            namespace=NAMESPACE,
        )
        upserted += len(batch)
        progress.progress(upserted / total, text=f"Upserted {upserted}/{total} candidates...")
    progress.empty()
    return upserted

if st.button("▶ Run Ingestion"):
    with st.spinner("Loading Excel..."):
        all_records = load_candidates()
    st.info(f"Excel loaded: **{len(all_records)}** total candidates")

    if mode.startswith("Add New"):
        with st.spinner("Fetching existing IDs from Pinecone..."):
            existing_ids = fetch_existing_ids()

        new_records = [r for r in all_records if r["id"] not in existing_ids]
        st.info(f"Already ingested: **{len(existing_ids)}** | New to add: **{len(new_records)}**")

        if not new_records:
            st.success("✅ Nothing new to ingest — Pinecone is already up to date.")
            st.stop()

        with st.spinner("Rebuilding BM25 index..."):
            fit_bm25(all_records)
        st.success("✅ BM25 index saved.")

        upserted = upsert_with_progress(new_records)
        st.success(f"✅ Done — **{upserted}** new candidates upserted to `{INDEX_NAME}/{NAMESPACE}`.")

    elif mode.startswith("Sync"):
        with st.spinner("Fetching existing IDs from Pinecone..."):
            existing_ids = fetch_existing_ids()

        excel_ids     = {r["id"] for r in all_records}
        ids_to_delete = existing_ids - excel_ids
        new_records   = [r for r in all_records if r["id"] not in existing_ids]

        st.info(f"Existing: **{len(existing_ids)}** | To delete: **{len(ids_to_delete)}** | To add: **{len(new_records)}**")

        if ids_to_delete:
            with st.spinner(f"Deleting {len(ids_to_delete)} removed candidates..."):
                index = pc.Index(INDEX_NAME)
                id_list = list(ids_to_delete)
                for i in range(0, len(id_list), 1000):
                    index.delete(ids=id_list[i : i + 1000], namespace=NAMESPACE)
            st.success(f"🗑️ Deleted **{len(ids_to_delete)}** candidates from Pinecone.")

        with st.spinner("Rebuilding BM25 index..."):
            fit_bm25(all_records)
        st.success("✅ BM25 rebuilt.")

        if new_records:
            upserted = upsert_with_progress(new_records)
            st.success(f"✅ Sync complete — **{upserted}** new candidates added.")
        else:
            st.success("✅ Sync complete — no new candidates to add.")

    else:  # Full Re-ingest
        with st.spinner("Deleting existing namespace..."):
            try:
                pc.Index(INDEX_NAME).delete(delete_all=True, namespace=NAMESPACE)
                st.info(f"Namespace `{NAMESPACE}` cleared.")
            except Exception as e:
                st.warning(f"Could not clear namespace: {e}")

        with st.spinner("Rebuilding BM25 index..."):
            fit_bm25(all_records)
        st.success("✅ BM25 index saved.")

        upserted = upsert_with_progress(all_records)
        st.success(f"✅ Full re-ingest complete — **{upserted}** candidates upserted to `{INDEX_NAME}/{NAMESPACE}`.")
