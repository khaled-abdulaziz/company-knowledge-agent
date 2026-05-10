---
title: Company Knowledge Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
app_file: app.py
pinned: false
---

# 🤖 Company Knowledge Agent

> **Live Demo** → [Try it on Hugging Face](https://huggingface.co/spaces/khaledxd/company-knowledge-agent)  
> **Built with** LangGraph · LlamaIndex · Qdrant · MySQL · GPT-4o · Ollama · Streamlit

---

## What is this?

A company internal chatbot that answers two types of questions:

- **Document questions** → searches company policy PDFs using RAG
- **Database questions** → generates and runs SQL on a real MySQL database

Supports **Arabic and English** out of the box.

---

## How it works

User asks a question
↓
router_node         ← GPT-4o decides: docs or sql?
↓                      ↓
docs_node            sql_node
↓                      ↓
Qdrant search        LLM generates SQL
(RAG retrieval)      MySQL runs query
↓                      ↓
Answer back to user

**Sensitive questions** (salary, personal data) are detected by keyword and routed to a **local Ollama model** — the data never leaves the machine.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent framework | LangGraph | Controls the workflow between nodes |
| Document search | LlamaIndex + Qdrant | RAG over company PDFs |
| Embeddings | OpenAI text-embedding-3-small | Converts text to vectors |
| Public LLM | GPT-4o | Routing, SQL generation, answers |
| Private LLM | Ollama (llama3) | Sensitive queries — runs locally |
| Database | MySQL + SQLAlchemy | Employee and sales data |
| UI | Streamlit | Chat interface |
| Tool layer | Custom MCP registry | Unified interface for all services |

---

## Project Structure

company_agent_project/
├── app.py                  # Streamlit UI (chat + document upload)
├── main.py                 # CLI mode for local testing
├── requirements.txt
├── .env.example            # Template — copy to .env and fill in your keys
│
├── data/
│   └── manuals/            # Place your PDF/TXT/DOCX files here
│
└── src/
├── graph/
│   ├── state.py        # Shared memory between all nodes
│   ├── nodes.py        # router_node, docs_node, sql_node
│   └── workflow.py     # LangGraph graph assembly
│
├── db/
│   ├── sql_client.py   # MySQL connection + SQL safety guard
│   └── vector_store.py # Qdrant setup + document upload
│
└── tools/
├── mcp_tools.py    # MCP tool registry (all service calls)
└── custom_tools.py # Utility functions

---

## Key Design Decisions

**Why LangGraph?**
The agent needs conditional routing — document questions and database questions require completely different pipelines. LangGraph makes this explicit and easy to extend.

**Why MCP tools layer?**
Instead of calling GPT-4o, Qdrant, and MySQL directly from the nodes, every external call goes through a named tool in `mcp_tools.py`. This means one place to change if a service moves, and every tool returns `{"success": bool, ...}` so errors are handled consistently.

**Why Ollama for sensitive data?**
Salary, national IDs, and personal data should never be sent to a cloud API. Questions containing sensitive keywords are detected locally and routed to a local Ollama model. The data stays on the machine.

**Why upload-once to Qdrant?**
Document vectors are stored permanently in Qdrant's Docker volume. On every restart, the app checks if documents already exist and skips re-uploading. A force re-index option is available when new documents are added.

---

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/khaled-abdulaziz/company-knowledge-agent.git
cd company-knowledge-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Open .env and fill in your keys
```

### 4. Start required services
```bash
# Qdrant (vector database)
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Ollama (local LLM for sensitive queries)
ollama serve
ollama pull llama3
```

### 5. Add your documents

Drop your PDF / TXT / DOCX files into data/manuals/
### 6. Run the app
```bash
streamlit run app.py
```

Or test in terminal:
```bash
python main.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

OPENAI_API_KEY         your OpenAI key
OPENAI_MODEL           gpt-4o
OPENAI_EMBEDDING_MODEL text-embedding-3-small
OLLAMA_BASE_URL        http://localhost:11434
OLLAMA_MODEL           llama3
MYSQL_HOST             localhost
MYSQL_PORT             3306
MYSQL_USER             your_user
MYSQL_PASSWORD         your_password
MYSQL_DB               company_db
MYSQL_SSL              false
QDRANT_HOST            localhost
QDRANT_PORT            6333
QDRANT_COLLECTION      company_docs
SENSITIVE_KEYWORDS     salary,password,national_id,address,marital_status

---

## Example Questions

### 📄 Document questions (PDF policy)
Upload the company policy PDF then try:
- "What is the employee leave policy?"
- "What are the data and security policies?"
- "What are the customer service standards?"
- "What does the company sell and where does it operate?"

### 🗄️ Database questions (MySQL)
- "How many employees do we have?"
- "What is Ahmed Alqahtani's salary?"
- "List all departments"
- "Show me all products and their prices"
- "Which city had the most sales?"

### 🌐 Arabic questions
- "ما هي سياسة الإجازات؟"
- "كم عدد الموظفين لدينا؟"
- "ما هو راتب أحمد القحطاني؟"
- "اعرض لي جميع المنتجات وأسعارها"

---

## Live Demo Notes

The live demo on Hugging Face requires you to bring your own OpenAI API key.  
Enter it in the sidebar before chatting. Your key is never stored or logged.

---

## Author

Built by Khaled Abdulaziz· [LinkedIn](https://www.linkedin.com/in/khaled-abdulaziz/) · [GitHub](https://github.com/khaled-abdulaziz)