# Advanced HR RAG Agent — Architecture & Replication Guide

## Overview

A production-grade agentic HR assistant that combines **Hybrid Search** (dense + BM25),
**Cross-Encoder Re-ranking**, and a **LangGraph supervisor loop** to answer two types of
HR requests: candidate search and Job Description generation.

---

## Project Structure

```
Advanced RAG/
├── .env                                  # API keys and config
├── requirements.txt                      # Python dependencies
├── Candidate_CV_Vector_Database_100.xlsx # Source data (100 candidate profiles)
├── ingest.py                             # One-time data ingestion pipeline
├── agent.py                              # Agent graph, nodes, hybrid search
├── main.py                               # Test runner (9 test cases)
└── bm25_index.pkl                        # BM25 index saved by ingest.py (auto-generated)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                              │
│                         ingest.py                                   │
└─────────────────────────────────────────────────────────────────────┘

  Excel File (100 candidates, 21 columns)
          │
          ▼
  load_candidates()
  └── Builds rich text blob per candidate:
      Name | Role | Domain | Seniority | YOE | Location |
      Work Mode | Notice | CTC | Skills | Summary |
      Experience | Project Title | Project Architecture
          │
          ├──────────────────────────────────────────────────────────┐
          │                                                          │
          ▼                                                          ▼
  fit_bm25()                                              upsert_pinecone()
  └── BM25Okapi fit on full corpus                        └── Batch embed (20/batch)
  └── Saved to bm25_index.pkl                                 via OpenRouter
      {bm25, ids[], texts[]}                                   text-embedding-3-small
                                                               → 1536-dim vectors
                                                           └── Upsert to Pinecone
                                                               index : hr-rag
                                                               namespace : candidates
                                                               metric : cosine


┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT PHASE                                  │
│                          agent.py                                   │
└─────────────────────────────────────────────────────────────────────┘

  User Query
      │
      ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    LangGraph StateGraph                         │
  │                                                                 │
  │  AgentState = {                                                 │
  │    messages        : list[dict]   # chat history (user only)   │
  │    next            : str          # routing signal             │
  │    rag_result      : str          # filled by rag_node         │
  │    jd_result       : str          # filled by jd_node          │
  │    final_answer    : str          # filled by supervisor        │
  │    _rag_query      : str          # query passed to rag_node   │
  │    _jd_description : str          # desc passed to jd_node     │
  │  }                                                              │
  └─────────────────────────────────────────────────────────────────┘
      │
      ▼
  ┌──────────────┐
  │  SUPERVISOR  │  ← LLM: openai/gpt-4o-mini via OpenRouter
  │              │
  │  Tools:      │  Reads: messages[0], rag_result, jd_result
  │  route_to_rag│  Builds clean [system, user, assistant(context)]
  │  route_to_jd │  Never injects role=tool into messages
  └──────┬───────┘
         │
         │  conditional edge (supervisor_router)
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
┌──────────┐          ┌──────────┐
│ rag_node │          │ jd_node  │
└──────────┘          └──────────┘
    │                       │
    └──────────┬────────────┘
               │  (both edges return to supervisor)
               ▼
          SUPERVISOR
               │
               │  (no tool_calls → synthesize)
               ▼
             END
```

---

## Hybrid Search Pipeline (rag_node)

```
User Query
    │
    ├─[Step 1: Dense Retrieval]──────────────────────────────────────
    │   embed_query(query)
    │   └── OpenRouter text-embedding-3-small → 1536-dim vector
    │   dense_search(vec, top_k=20)
    │   └── Pinecone cosine similarity → top 20 candidates
    │
    ├─[Step 2: Sparse Retrieval]─────────────────────────────────────
    │   bm25_search(query, top_k=20)
    │   └── Tokenize query → BM25Okapi.get_scores()
    │   └── Sort by BM25 score → top 20 candidates
    │
    ├─[Step 3: RRF Fusion]───────────────────────────────────────────
    │   reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)
    │   └── score(doc) = 1/(60 + rank_dense) + 1/(60 + rank_bm25)
    │   └── Deduplicate → ~20-35 unique candidates ranked by fused score
    │
    └─[Step 4: Cross-Encoder Re-ranking]─────────────────────────────
        rerank(query, fused_docs, top_k=3)
        └── BAAI/bge-reranker-base (local, no API)
        └── Score each (query, doc) pair with cross-attention
        └── Sort descending → return top 3
```

### Why Hybrid over Pure Dense

| Scenario | Pure Dense | Hybrid (Dense + BM25) |
|---|---|---|
| `"LangGraph RAG engineer"` | Semantic match | BM25 boosts exact `LangGraph` token hits |
| `"XGBoost fintech fraud"` | May miss exact term | BM25 scores `XGBoost` precisely |
| `"gRPC Kafka distributed"` | Broad semantic | Exact tech stack terms ranked higher |
| General role queries | Good | Equally good (dense dominates via RRF) |

---

## LangGraph Node Responsibilities

### supervisor
- LLM call #1 (and #N for multi-task queries)
- Reads `rag_result` / `jd_result` from state as plain assistant context
- Calls `route_to_rag` or `route_to_jd` tool if task pending
- Responds directly (no tool call) when all tasks complete → sets `next="end"`
- Message history is always clean: `[system, user, assistant?]` — never `role=tool`

### rag_node
- Runs full hybrid search pipeline (dense → BM25 → RRF → rerank)
- Writes result to `rag_result` state field only
- Returns `messages=[]` — never pollutes chat history

### jd_node
- Single LLM call with structured 6-section JD prompt
- No retrieval — pure generation
- Writes result to `jd_result` state field only
- Returns `messages=[]` — never pollutes chat history

---

## Models & Services

| Component | Model / Service | Runs Where |
|---|---|---|
| LLM (routing + synthesis) | `openai/gpt-4o-mini` | OpenRouter API |
| LLM (JD generation) | `openai/gpt-4o-mini` | OpenRouter API |
| Dense embeddings | `openai/text-embedding-3-small` | OpenRouter API |
| Sparse retrieval | `BM25Okapi` (rank-bm25) | Local |
| Vector store | Pinecone Serverless | Cloud (AWS us-east-1) |
| Cross-encoder reranker | `BAAI/bge-reranker-base` | Local |
| Agent orchestration | LangGraph `StateGraph` | Local |

---

## Key Configuration Parameters

| Parameter | Value | Location | Effect |
|---|---|---|---|
| `DENSE_TOP_K` | 20 | agent.py | Candidates fetched from Pinecone |
| `BM25_TOP_K` | 20 | agent.py | Candidates fetched from BM25 |
| `RRF_K` | 60 | agent.py | RRF smoothing constant (standard = 60) |
| `FINAL_TOP_K` | 3 | agent.py | Final candidates after cross-encoder |
| `BATCH_SIZE` | 20 | ingest.py | Pinecone upsert batch size |
| `EMBED_MODEL` | text-embedding-3-small | both | 1536-dim, must match index |
| `PINECONE_INDEX` | hr-rag | .env | Index name (1536-dim, cosine metric) |
| `NAMESPACE` | candidates | both | Pinecone namespace |

---

## Data Schema

### Excel Source: `Candidate Profiles Master` sheet

| Column | Type | Used For |
|---|---|---|
| Candidate ID | str | Vector ID, BM25 ID |
| Full Name | str | Text blob, metadata |
| Target Role | str | Text blob, metadata |
| Domain Focus | str | Text blob, metadata |
| Seniority Level | str | Text blob, metadata |
| YOE | float | Text blob, metadata |
| Location | str | Text blob, metadata |
| Work Mode | str | Text blob, metadata |
| Notice Period | str | Text blob, metadata |
| Expected CTC (LPA) | float | Text blob, metadata |
| Core Skills Inventory | str | Text blob, metadata |
| Executive Summary | str | Text blob |
| Work Experience History | str | Text blob |
| Featured Project Title | str | Text blob |
| Featured Project Architecture | str | Text blob |
| Education | str | Metadata |
| Current Company | str | Metadata |

---

## Setup & Replication Steps

### 1. Prerequisites

- Python 3.10+
- Pinecone account with a serverless index created:
  - Dimension: `1536`
  - Metric: `cosine`
  - Cloud: `aws`, Region: `us-east-1`
- OpenRouter account with API key (used for both embeddings and LLM)

### 2. Clone & Install

```bash
git clone <repo>
cd "Advanced RAG"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` in the project root:

```env
PINECONE_API_KEY=<your_pinecone_api_key>
OPENROUTER_API_KEY=<your_openrouter_api_key>
PINECONE_INDEX=hr-rag
```

### 4. Ingest Data

```bash
python ingest.py
```

This will:
- Load 100 candidates from the Excel file
- Fit and save `bm25_index.pkl` locally
- Embed all candidates via OpenRouter (batches of 20)
- Upsert all vectors into Pinecone under namespace `candidates`

### 5. Run Tests

```bash
python main.py
```

### 6. Run Single Query

```python
from agent import run
run("Find senior ML engineers with LangChain and RAG experience")
run("Write a JD for a Senior DevOps Engineer with Kubernetes and AWS")
```

---

## Dependencies

```
pinecone-client==3.2.2     # Pinecone vector database client
openai>=1.0.0              # OpenAI-compatible client (pointed at OpenRouter)
sentence-transformers>=2.2.0  # CrossEncoder for re-ranking (BAAI/bge-reranker-base)
langgraph>=0.1.0           # Agent graph orchestration
python-dotenv>=1.0.0       # .env file loading
pandas>=2.0.0              # Excel data loading
openpyxl>=3.1.0            # Excel file engine for pandas
rank-bm25>=0.2.2           # Pure Python BM25 (no C++ build tools needed)
```

---

## Multi-Task Query Flow (supervisor loop)

For a query like _"Find matching candidates AND write a JD for computer vision engineer"_,
the graph executes multiple hops:

```
START
  → supervisor   (routes to rag)
  → rag_node     (hybrid search → top 3 candidates)
  → supervisor   (sees rag_result, routes to jd)
  → jd_node      (generates JD)
  → supervisor   (sees both results, synthesizes final answer)
  → END
```

---

## Known Constraints & Extension Points

| Constraint | Reason | How to Extend |
|---|---|---|
| Pinecone index uses `cosine` not `dotproduct` | Native Pinecone hybrid requires `dotproduct` | Recreate index with `dotproduct` and use `pinecone-text` sparse encoder for true native hybrid |
| BM25 index is file-based (`pkl`) | Avoids C++ build dependency (`mmh3`) | Replace with `pinecone-text` BM25Encoder once build tools available |
| Single namespace `candidates` | Current data is one pool | Add namespaces (e.g. `internal` / `external`) and route per namespace in supervisor |
| `FINAL_TOP_K=3` | Keeps LLM context concise | Increase for broader shortlists |
| No metadata filtering | All candidates searched equally | Add Pinecone metadata filters (e.g. `seniority`, `location`, `yoe`) to `dense_search()` |
