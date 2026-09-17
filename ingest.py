"""
ingest.py — Load candidates from Excel, upsert into Pinecone,
            and fit + save a BM25 model for hybrid search.
"""
import os, pickle, time
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
INDEX_NAME         = os.getenv("PINECONE_INDEX", "hr-rag")
NAMESPACE          = "candidates"
EXCEL_FILE         = "Candidate_CV_Vector_Database_100.xlsx"
SHEET_NAME         = "Candidate Profiles Master"
EMBED_MODEL        = "openai/text-embedding-3-small"
BM25_PATH          = "bm25_index.pkl"
BATCH_SIZE         = 20

pc     = Pinecone(api_key=PINECONE_API_KEY)
openai = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# ---------- load Excel ----------
def load_candidates() -> list[dict]:
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()
    records = []
    for _, row in df.iterrows():
        cid = str(row.get("Candidate ID", "")).strip()
        if not cid or cid == "nan":
            continue
        text = (
            f"{row['Full Name']} | {row['Target Role']} | {row['Domain Focus']} | "
            f"Seniority: {row['Seniority Level']} | YOE: {row['YOE']} | "
            f"Location: {row['Location']} | Work Mode: {row['Work Mode']} | "
            f"Notice: {row['Notice Period']} | CTC: {row['Expected CTC (LPA)']} LPA | "
            f"Skills: {row['Core Skills Inventory']} | "
            f"Summary: {row['Executive Summary']} | "
            f"Experience: {row['Work Experience History']} | "
            f"Project: {row['Featured Project Title']} — {row['Featured Project Architecture']}"
        )
        records.append({
            "id":   cid,
            "text": text,
            "metadata": {
                "name":      str(row["Full Name"]),
                "role":      str(row["Target Role"]),
                "domain":    str(row["Domain Focus"]),
                "seniority": str(row["Seniority Level"]),
                "yoe":       float(row["YOE"]) if pd.notna(row["YOE"]) else 0.0,
                "location":  str(row["Location"]),
                "work_mode": str(row["Work Mode"]),
                "notice":    str(row["Notice Period"]),
                "ctc":       float(row["Expected CTC (LPA)"]) if pd.notna(row["Expected CTC (LPA)"]) else 0.0,
                "skills":    str(row["Core Skills Inventory"]),
                "education": str(row["Education"]),
                "company":   str(row["Current Company"]),
                "text":      text,
            },
        })
    print(f"[ingest] Loaded {len(records)} candidates from Excel")
    return records

# ---------- BM25 ----------
def fit_bm25(records: list[dict]):
    corpus = [r["text"].lower().split() for r in records]
    bm25   = BM25Okapi(corpus)
    payload = {"bm25": bm25, "ids": [r["id"] for r in records], "texts": [r["text"] for r in records]}
    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[ingest] BM25 index fitted and saved → '{BM25_PATH}'")

# ---------- Pinecone upsert ----------
def upsert_pinecone(records: list[dict]):
    index = pc.Index(INDEX_NAME)
    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        resp  = openai.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs  = [d.embedding for d in resp.data]
        index.upsert(
            vectors=[
                {"id": r["id"], "values": v, "metadata": r["metadata"]}
                for r, v in zip(batch, vecs)
            ],
            namespace=NAMESPACE,
        )
        total += len(batch)
        print(f"[ingest] Upserted {total}/{len(records)} candidates...")
    print(f"[ingest] Done — {total} candidates in index='{INDEX_NAME}' namespace='{NAMESPACE}'")

# ---------- main ----------
if __name__ == "__main__":
    print(f"[ingest] Connecting to Pinecone index: '{INDEX_NAME}'")
    records = load_candidates()
    fit_bm25(records)
    upsert_pinecone(records)
