# ==============================================================
# LangGraph Node Functions
# ==============================================================

import os
from dotenv import load_dotenv

from .state import AgentState
from src.tools.mcp_tools import (
    search_docs,
    query_database,
    ask_public_llm,
    ask_private_llm,
    get_schema_info,
)

load_dotenv()

# ==============================================================
#.env
# ==============================================================

OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


_SENSITIVE_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv("SENSITIVE_KEYWORDS", "").split(",")
    if kw.strip() 
]


# ==============================================================
# Helper: Check if question is sensitive
# ==============================================================

def _is_sensitive(question: str) -> bool:
    """
    Returns True if the question contains any sensitive keywords.
    Sensitive questions are routed to Ollama (local processing).

    Keywords are configured in .env → SENSITIVE_KEYWORDS
    Supports Arabic and English keywords.

    Example sensitive: "What is Ahmed's salary?" → True
    Example general:   "What is the leave policy?" → False
    """
    lowered = question.lower()
    return any(keyword in lowered for keyword in _SENSITIVE_KEYWORDS) 


# ==============================================================
# Node 1: Router
# ==============================================================

def router_node(state: AgentState) -> AgentState:
    """
    Analyzes the question and decides:
      1. route       → "docs" (PDFs/manuals) or "sql" (employee database)
      2. is_sensitive → True (use Ollama) or False (use GPT-4o)

    Uses GPT-4o via ask_public_llm() for the routing decision itself
    (fast, and the question text alone is not sensitive).

    Examples:
        "What is the vacation policy?"    → route=docs,  is_sensitive=False
        "Show me Ahmed's salary"          → route=sql,   is_sensitive=True
        "List all employees in IT dept"   → route=sql,   is_sensitive=False
        "ما هي سياسة الإجازات؟"           → route=docs,  is_sensitive=False
    """
    question = state["question"]

    system_prompt = """You are a smart routing assistant for a company knowledge base.

Your job is to classify the user's question into ONE of two categories:
- "docs"  → question is about company policies, procedures, manuals, HR rules, guidelines, or any document/PDF content
- "sql"   → question is about specific employee records, data, numbers, lists, or database queries

Rules:
- Answer ONLY with a single word: docs OR sql
- No explanation, no punctuation, just the word
- Works for both Arabic and English questions"""

    result   = ask_public_llm(system_prompt, question)
    decision = result["response"].lower()

   
    if decision not in ("docs", "sql"):
        decision = "docs"

    
    sensitive = _is_sensitive(question)

    state["route"]        = decision
    state["is_sensitive"] = sensitive

    print(f"🧭 Router → route='{decision}' | sensitive={sensitive}")
    return state


# ==============================================================
# Node 2: Docs Node (Qdrant + LlamaIndex)
# ==============================================================

def docs_node(state: AgentState) -> AgentState:
    """
    Handles document-based questions using Qdrant vector search.

    Flow:
        question → search_docs() → Qdrant similarity search
        → top-k chunks retrieved → LlamaIndex synthesizes answer

    For sensitive doc queries we append a privacy note.
    (LlamaIndex uses GPT-4o internally for synthesis — that part
    is unchanged; we just flag it for the user.)

    Example questions:
        "What is the remote work policy?"
        "ما هي شروط الإجازة السنوية؟"
    """
    question     = state["question"]
    is_sensitive = state.get("is_sensitive", False)

    try:
        doc_result   = search_docs(question, top_k=4)
        answer       = doc_result["answer"]
        source_texts = doc_result["source_texts"]   

        if not doc_result["success"]:
   
            state["answer"]          = answer
            state["retrieved_nodes"] = []
            state["llm_used"]        = "error"
            return state

        if is_sensitive:
            answer   = f"[🔒 Processed with extra care]\n\n{answer}"
            llm_used = "gpt-4o (docs retrieval) + privacy flag"
        else:
            llm_used = "gpt-4o"

        state["retrieved_nodes"] = source_texts
        state["answer"]          = answer
        state["llm_used"]        = llm_used

    except Exception as e:
        state["answer"]          = f"❌ Error searching documents: {str(e)}"
        state["retrieved_nodes"] = []
        state["llm_used"]        = "error"

    return state


# ==============================================================
# Node 3: SQL Node (MySQL + LLM-generated query)
# ==============================================================

def sql_node(state: AgentState) -> AgentState:
    """
    Handles structured data questions by:
      1. Pulling the DB schema from get_schema_info() (single source of truth)
      2. Using ask_public_llm() or ask_private_llm() to generate a SELECT query
      3. Running the query via query_database()
      4. Using the same LLM to format results into a human-readable answer

    Sensitive questions (salary, ID, etc.) go through ask_private_llm()
    so the data never leaves the machine.

    Example questions:
        "How many employees are in the HR department?"
        → SELECT COUNT(*) FROM employees WHERE department = 'HR'

        "ما هو راتب أحمد؟"  (sensitive → Ollama)
        → SELECT salary FROM employees WHERE name LIKE '%Ahmed%'
    """
    question     = state["question"]
    is_sensitive = state.get("is_sensitive", False)

    #
    schema_info = get_schema_info()

    sql_system_prompt = f"""You are an expert SQL assistant for a company HR database.

Generate a safe MySQL SELECT query based on the user's question.

{schema_info["schema"]}

Rules:
- Output ONLY the SQL query — no explanation, no markdown, no backticks
- Always use SELECT — never UPDATE, DELETE, INSERT, DROP, or ALTER
- Use proper WHERE clauses — never return all rows unless explicitly asked
- For Arabic names: use LIKE '%name%' for flexible matching
- Limit results to 20 rows maximum unless a specific number is requested"""

    try:
        #
        if is_sensitive:
            
            print(f"🔒 Sensitive query → using Ollama ({OLLAMA_MODEL})")
            llm_result = ask_private_llm(sql_system_prompt, question)
            llm_used   = f"ollama:{OLLAMA_MODEL}"
        else:
            
            print(f"⚡ General query → using GPT-4o")
            llm_result = ask_public_llm(sql_system_prompt, question)
            llm_used   = "gpt-4o"

        if not llm_result["success"]:
            state["answer"]      = llm_result["response"]   
            state["sql_results"] = []
            state["llm_used"]    = "error"
            return state

        generated_sql = llm_result["response"]
        print(f"📝 Generated SQL: {generated_sql}")

        
        db_result = query_database(generated_sql)

        if not db_result["success"]:
            
            state["answer"]      = db_result["rows"]
            state["sql_results"] = []
            state["llm_used"]    = llm_used
            return state

        
        format_prompt = f"""The user asked: "{question}"

The database returned these results:
{db_result["rows"]}

Write a clear, professional response in the same language the user used (Arabic or English).
Be concise. If results are empty, say so politely."""

        format_system = "You are a helpful HR assistant. Answer professionally and concisely."

        if is_sensitive:
            format_result = ask_private_llm(format_system, format_prompt)
        else:
            format_result = ask_public_llm(format_system, format_prompt)

        final_answer = format_result["response"] if format_result["success"] else str(db_result["rows"])

        state["answer"]      = final_answer
        state["sql_results"] = db_result["rows"]
        state["llm_used"]    = llm_used

    except Exception as e:
        state["answer"]      = f"❌ Error processing your request: {str(e)}"
        state["sql_results"] = []
        state["llm_used"]    = "error"

    return state