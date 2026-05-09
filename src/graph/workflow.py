# ==============================================================
# LangGraph Workflow
# ==============================================================

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import router_node, docs_node, sql_node


# ==============================================================
# Graph Builder
# ==============================================================

def build_graph():
    """
    Builds and compiles the LangGraph agent workflow.

    Flow:
        [START]
           ↓
        router_node  ← decides: "docs" or "sql"
           ↓               ↓
        docs_node      sql_node
           ↓               ↓
         [END]           [END]

    Returns:
        Compiled LangGraph app ready to invoke.
    """

    graph = StateGraph(AgentState)

    # --- Register nodes ---
    graph.add_node("router", router_node)
    graph.add_node("docs",   docs_node)
    graph.add_node("sql",    sql_node)

    # --- Entry point ---
    graph.set_entry_point("router")

 
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],  
        {
            "docs": "docs",
            "sql":  "sql"
        }
    )

   
    graph.add_edge("docs", END)
    graph.add_edge("sql",  END)

    return graph.compile()


# ==============================================================
# run_agent — main entry point for the agent
# ==============================================================

def run_agent(question: str) -> dict:
    """
    Runs the full agent pipeline for a given question.

    Args:
        question (str): User's question in Arabic or English.

    Returns:
        dict with keys:
            - answer    (str)  : The final response
            - route     (str)  : Which path was taken ("docs" or "sql")
            - llm_used  (str)  : Which LLM generated the answer
            - is_sensitive (bool): Whether Ollama was used for privacy

    Usage:
        result = run_agent("What is the vacation policy?")
        print(result["answer"])

        result = run_agent("ما هو راتب أحمد؟")
        print(result["answer"])
        print(result["llm_used"])  # → "ollama:llama3"
    """

    app = build_graph()

    initial_state: AgentState = {
        "question":        question,
        "route":           "",
        "is_sensitive":    False,
        "retrieved_nodes": [],
        "sql_results":     [],
        "answer":          "",
        "llm_used":        ""
    }

    result = app.invoke(initial_state)

    return {
        "answer":       result["answer"],
        "route":        result["route"],
        "llm_used":     result["llm_used"],
        "is_sensitive": result["is_sensitive"]
    }