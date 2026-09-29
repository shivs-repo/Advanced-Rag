"""
graph_retrieve.py — True Graph RAG: Query → LLM generates Cypher → Neo4j → LLM answers

Steps:
  1. User sends a natural language query
  2. LLM generates a Cypher query based on the graph schema
  3. Execute the Cypher query on Neo4j Aura
  4. LLM refines the raw graph results into a final answer
"""

import os, re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────────────
NEO4J_URI          = os.environ["NEO4J_URI"]
NEO4J_USERNAME     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD     = os.environ["NEO4J_PASSWORD"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

LLM_MODEL = "openai/gpt-4o-mini"

openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# ── Step 1: Fetch live schema from Neo4j ─────────────────────────────────────

def fetch_schema_from_neo4j() -> str:
    """Query Neo4j directly to get current node labels and relationship types."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        labels    = [r["label"] for r in session.run("CALL db.labels()")]
        rel_types = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")]
        prop_keys = [r["propertyKey"] for r in session.run("CALL db.propertyKeys()")]
    driver.close()

    schema = (
        f"Node labels: {labels}\n"
        f"Relationship types: {rel_types}\n"
        f"Property keys: {prop_keys}"
    )
    print(f"[retrieve] Step 1 ✓ Fetched live schema from Neo4j")
    print(f"  Labels    : {labels}")
    print(f"  Rel types : {rel_types}")
    return schema

# ── STEP 2: LLM generates a Cypher query ─────────────────────────────────────

CYPHER_GEN_PROMPT = """
You are a Neo4j Cypher expert. Given the graph schema and a user question, 
write a Cypher query to retrieve the relevant information.

Graph Schema:
{schema}

Rules:
- Return ONLY the raw Cypher query, no explanation, no markdown, no code fences
- Use MATCH and OPTIONAL MATCH to traverse relationships
- Always RETURN meaningful node properties, not entire nodes
- If the question is broad, return up to 20 results using LIMIT 20
- Use case-insensitive matching with toLower() when filtering by name

User Question: {question}

Cypher Query:
"""

def generate_cypher(question: str, schema: str) -> str:
    """Step 2: Ask LLM to generate a Cypher query using the live schema."""
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": CYPHER_GEN_PROMPT.format(schema=schema, question=question)
        }],
        temperature=0,
    )
    cypher = resp.choices[0].message.content.strip()

    # Strip markdown code fences if LLM adds them anyway
    cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher)
    cypher = re.sub(r"\s*```$", "", cypher)

    print(f"[retrieve] Step 2 ✓ Generated Cypher:\n  {cypher}\n")
    return cypher


# ── STEP 3: Execute Cypher on Neo4j ──────────────────────────────────────────

def execute_cypher(cypher: str) -> list[dict]:
    """Step 3: Run the Cypher query on Neo4j Aura and return raw records."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result  = session.run(cypher)
            records = [dict(r) for r in result]
        print(f"[retrieve] Step 3 ✓ Cypher returned {len(records)} records")
        return records
    except Exception as e:
        print(f"[retrieve] Step 3 ✗ Cypher execution failed: {e}")
        return []
    finally:
        driver.close()


# ── STEP 4: LLM refines raw results into a final answer ──────────────────────

ANSWER_PROMPT = """
You are a helpful assistant. The user asked a question and we retrieved the 
following raw data from a knowledge graph to help answer it.

User Question: {question}

Raw Graph Data:
{raw_data}

Using only the data above, provide a clear and concise answer.
If the data is empty or insufficient, say so honestly.
"""

def generate_answer(question: str, records: list[dict]) -> str:
    """Step 4: LLM takes raw Neo4j records and generates a natural language answer."""
    if not records:
        raw_data = "No results found in the graph."
    else:
        raw_data = "\n".join(str(r) for r in records)

    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": ANSWER_PROMPT.format(question=question, raw_data=raw_data)
        }],
    )
    print("[retrieve] Step 4 ✓ Final answer generated")
    return resp.choices[0].message.content.strip()


# ── Main orchestrator ─────────────────────────────────────────────────────────

def graph_rag(question: str) -> str:
    print(f"\n[retrieve] Query: \"{question}\"")
    print("-" * 60)

    # Step 1 — fetch live schema from Neo4j
    schema = fetch_schema_from_neo4j()

    # Step 2 — LLM generates Cypher
    cypher = generate_cypher(question, schema)

    # Step 3 — Execute on Neo4j
    records = execute_cypher(cypher)

    # Step 4 — LLM generates final answer from raw records
    answer = generate_answer(question, records)

    print("\n[retrieve] ✅ Done")
    return answer


if __name__ == "__main__":
    queries = [
        "what are roles we have in our company?",
        "who is the manager of Alice?"
    ]

    for q in queries:
        answer = graph_rag(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
        print("=" * 60)
