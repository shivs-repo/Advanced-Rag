"""
main.py — Test suite for the HR RAG Agent.
Run: python main.py
"""
from agent import run

TEST_CASES = [
    # --- employee_search tool ---
    ("SEARCH | GenAI / LLM engineers",
     "Find senior candidates with LangChain, LangGraph, and RAG experience"),

    ("SEARCH | DevOps & cloud infra",
     "Who are the best DevOps or SRE candidates with Kubernetes and Terraform skills?"),

    ("SEARCH | Data engineers",
     "Find data engineers experienced with Spark, Airflow, and Snowflake"),

    ("SEARCH | Fintech ML",
     "Show me ML engineers who have worked in fintech or fraud detection"),

    ("SEARCH | Remote senior candidates",
     "Find senior or lead level candidates open to remote work with Python expertise"),

    # --- generate_jd tool ---
    ("JD | ML Engineer",
     "Write a JD for a Mid-Senior Machine Learning Engineer specializing in NLP and LLMs for a healthcare startup"),

    ("JD | Backend Engineer",
     "Generate a job description for a Senior Backend Engineer with Go and distributed systems experience at a fintech company"),

    ("JD | Data Engineer",
     "Create a JD for a Data Engineer role focused on real-time streaming pipelines using Kafka and Spark"),

    # --- agent decides ---
    ("AUTO | Shortlist + JD",
     "I need to hire a computer vision engineer. First find matching candidates, then write a JD for the role"),
]

DIVIDER = "=" * 60

def main():
    print(f"\n{DIVIDER}")
    print("  HR RAG AGENT — TEST RUN")
    print(f"  {len(TEST_CASES)} test cases")
    print(DIVIDER)

    results = []
    for i, (label, query) in enumerate(TEST_CASES, 1):
        print(f"\n[TEST {i}/{len(TEST_CASES)}] {label}")
        answer = run(query)
        results.append((label, query, answer))

    print(f"\n{DIVIDER}")
    print("  SUMMARY")
    print(DIVIDER)
    for i, (label, query, answer) in enumerate(results, 1):
        print(f"\n[{i}] {label}")
        print(f"  Q: {query}")
        print(f"  A: {answer[:250]}{'...' if len(answer) > 250 else ''}")

    print(f"\n{DIVIDER}")
    print(f"  All {len(TEST_CASES)} tests completed.")
    print(DIVIDER)

if __name__ == "__main__":
    main()
