from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


class AgentState(TypedDict):
    query: str
    route: str
    answer: str


# ── Supervisor ────────────────────────────────────────────────────────────────
def supervisor(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a router. Based on the user query, reply with exactly one word: "
            "'finance' if the query is about budgets, investments, salaries, expenses, or financial topics, "
            "'hr' if the query is about hiring, employees, leave, policies, or HR topics."
        )),
        HumanMessage(content=state["query"])
    ])
    route = response.content.strip().lower()
    return {**state, "route": route}


def route_decision(state: AgentState) -> Literal["finance", "hr"]:
    return state["route"] if state["route"] in ("finance", "hr") else "hr"


# ── Specialist nodes ──────────────────────────────────────────────────────────
def finance_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="You are a Finance expert. Answer the user's finance-related question concisely."),
        HumanMessage(content=state["query"])
    ])
    return {**state, "answer": response.content}


def hr_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="You are an HR expert. Answer the user's HR-related question concisely."),
        HumanMessage(content=state["query"])
    ])
    return {**state, "answer": response.content}


# ── Build graph ───────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor)
graph.add_node("finance", finance_node)
graph.add_node("hr", hr_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_decision, {"finance": "finance", "hr": "hr"})
graph.add_edge("finance", END)
graph.add_edge("hr", END)

app = graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "What is the process for annual budget planning?",
        "How many days of sick leave are employees entitled to?",
    ]
    for q in queries:
        result = app.invoke({"query": q, "route": "", "answer": ""})
        print(f"Query  : {result['query']}")
        print(f"Routed : {result['route']}")
        print(f"Answer : {result['answer']}")
        print("-" * 60)
