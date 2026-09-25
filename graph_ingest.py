"""
graph_ingest.py — PDF → Chunk → LLM extracts entities/relationships → Neo4j Aura

Steps:
  1. Load & chunk the PDF using pymupdf + RecursiveCharacterTextSplitter
  2. For each chunk, ask the LLM to extract a structured JSON of entities & relationships
  3. Parse and deduplicate across all chunks
  4. Insert nodes and relationships into Neo4j Aura
"""

import os, json, re
import fitz  # pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────────────
NEO4J_URI          = os.environ["NEO4J_URI"]
NEO4J_USERNAME     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD     = os.environ["NEO4J_PASSWORD"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

LLM_MODEL     = "openai/gpt-4o-mini"
PDF_FILE      = "Corporate_Policies_Dataset_Final.pdf"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


def get_driver():
    """Create Neo4j driver with SSL trust fix for Python 3.14 / self-signed certs."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ── STEP 1: Load PDF and chunk it ─────────────────────────────────────────────

def load_chunks(pdf_path: str) -> list[str]:
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    splitter  = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks    = splitter.split_text(full_text)
    print(f"[ingest] Step 1 ✓ Extracted {len(chunks)} chunks from '{pdf_path}'")
    return chunks


# ── STEP 2: LLM extracts structured JSON from each chunk ─────────────────────

EXTRACTION_PROMPT = """
You are a knowledge graph extractor. Given the text below, extract entities and relationships.

Return ONLY valid JSON in this exact format (no explanation, no markdown):
{{
  "entities": [
    {{"label": "EntityType", "name": "EntityName", "properties": {{"key": "value"}}}}
  ],
  "relationships": [
    {{"from": "EntityName1", "type": "RELATIONSHIP_TYPE", "to": "EntityName2"}}
  ]
}}

Rules:
- Entity labels must be one of: Policy, Department, Employee, Role, Benefit, Leave, Conduct, Penalty
- Relationship types must be UPPERCASE_SNAKE_CASE (e.g. APPLIES_TO, GOVERNS, ENTITLES, REPORTS_TO)
- Only extract what is explicitly stated in the text
- If nothing meaningful can be extracted, return: {{"entities": [], "relationships": []}}

Text:
\"\"\"
{chunk}
\"\"\"
"""

def extract_graph_from_chunk(chunk: str, chunk_idx: int) -> dict:
    """Step 2: Call LLM to extract entities and relationships from one chunk."""
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(chunk=chunk)}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data     = json.loads(raw)
        entities = data.get("entities", [])
        rels     = data.get("relationships", [])
        print(f"[ingest] Step 2 ✓ Chunk {chunk_idx+1}: {len(entities)} entities, {len(rels)} relationships")
        return data
    except json.JSONDecodeError:
        print(f"[ingest] Step 2 ✗ Chunk {chunk_idx+1}: JSON parse failed — skipping")
        return {"entities": [], "relationships": []}


# ── STEP 3: Deduplicate all extracted data across chunks ──────────────────────

def merge_extractions(all_extractions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Step 3: Deduplicate entities and relationships across all chunks."""
    seen_entities = {}   # name -> entity dict
    seen_rels     = set()
    relationships = []

    for extraction in all_extractions:
        for e in extraction.get("entities", []):
            name = e.get("name", "").strip()
            if name and name not in seen_entities:
                seen_entities[name] = e

        for r in extraction.get("relationships", []):
            key = (r.get("from", ""), r.get("type", ""), r.get("to", ""))
            if all(key) and key not in seen_rels:
                seen_rels.add(key)
                relationships.append(r)

    entities = list(seen_entities.values())
    print(f"[ingest] Step 3 ✓ Deduplicated → {len(entities)} entities, {len(relationships)} relationships")
    return entities, relationships


# ── STEP 4: Insert into Neo4j Aura ───────────────────────────────────────────

def _safe_key(k: str) -> str:
    """Sanitize property key to a valid Cypher identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", k).strip("_") or "prop"


def insert_entity(tx, entity: dict):
    label = entity.get("label", "Entity").strip()
    name  = entity.get("name", "").strip()
    props = entity.get("properties", {})
    if not name:
        return
    # Sanitize keys: "Annual Deductible" → "Annual_Deductible"
    safe_props = {_safe_key(k): v for k, v in props.items()}
    prop_str   = ", ".join(f"n.{k} = ${k}" for k in safe_props)
    set_clause = f", {prop_str}" if prop_str else ""
    tx.run(
        f"MERGE (n:{label} {{name: $name}}) SET n.name = $name{set_clause}",
        name=name, **safe_props,
    )


def insert_relationship(tx, rel: dict):
    from_name = rel.get("from", "").strip()
    to_name   = rel.get("to", "").strip()
    rel_type  = rel.get("type", "RELATED_TO").strip().upper().replace(" ", "_")
    if not from_name or not to_name:
        return
    tx.run(
        f"""
        MATCH (a {{name: $from_name}})
        MATCH (b {{name: $to_name}})
        MERGE (a)-[:{rel_type}]->(b)
        """,
        from_name=from_name, to_name=to_name,
    )


def insert_into_neo4j(entities: list[dict], relationships: list[dict]):
    """Step 4: Write all nodes and edges into Neo4j Aura."""
    driver = get_driver()

    with driver.session() as session:
        # 4a — Clear old data
        session.run("MATCH (n) DETACH DELETE n")
        print("[ingest] Step 4a ✓ Cleared existing graph")

        # 4b — Insert nodes
        for e in entities:
            session.execute_write(insert_entity, e)
        print(f"[ingest] Step 4b ✓ Inserted {len(entities)} nodes")

        # 4c — Insert relationships
        skipped = 0
        for r in relationships:
            try:
                session.execute_write(insert_relationship, r)
            except Exception:
                skipped += 1
        print(f"[ingest] Step 4c ✓ Inserted relationships (skipped {skipped} with missing nodes)")

    driver.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest(pdf_path: str = PDF_FILE):
    # Step 1 — PDF → chunks
    chunks = load_chunks(pdf_path)

    # Step 2 — LLM extracts JSON from each chunk
    all_extractions = [extract_graph_from_chunk(chunk, i) for i, chunk in enumerate(chunks)]

    # Step 3 — Deduplicate across all chunks
    entities, relationships = merge_extractions(all_extractions)

    # Step 4 — Insert into Neo4j Aura
    insert_into_neo4j(entities, relationships)

    print(f"\n[ingest] ✅ Graph ingestion complete! Nodes: {len(entities)} | Edges: {len(relationships)}")


if __name__ == "__main__":
    ingest()
