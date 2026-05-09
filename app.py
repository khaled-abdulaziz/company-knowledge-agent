# ==============================================================
# app.py — Streamlit UI
# ==============================================================

import streamlit as st
import tempfile
import os
import time

from src.db.vector_store import upload_documents, collection_has_documents
from src.graph.workflow import run_agent

# ==============================================================
# Page config
# ==============================================================

st.set_page_config(
    page_title="Company Knowledge Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# Custom CSS
# ==============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    .msg-user {
        background: #1a1f2e;
        border: 1px solid #2a3045;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0 8px 60px;
        color: #e6edf3;
        font-size: 15px;
        line-height: 1.6;
    }
    .msg-assistant {
        background: #161b27;
        border: 1px solid #1e2d40;
        border-left: 3px solid #2f81f7;
        border-radius: 4px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 60px 8px 0;
        color: #e6edf3;
        font-size: 15px;
        line-height: 1.6;
    }

    .meta-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    .badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
    }
    .badge-route  { background: #1f2d40; color: #58a6ff; border: 1px solid #2f4a6e; }
    .badge-llm    { background: #1f2d1f; color: #3fb950; border: 1px solid #2d5a2d; }
    .badge-lock   { background: #2d1f1f; color: #f85149; border: 1px solid #5a2d2d; }

    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
    .dot-red   { background: #f85149; box-shadow: 0 0 6px #f85149; }
    .dot-amber { background: #d29922; box-shadow: 0 0 6px #d29922; }

    .sidebar-section {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #484f58 !important;
        margin: 20px 0 8px;
    }

    .empty-state {
        text-align: center;
        padding: 80px 20px;
        color: #484f58;
    }
    .empty-state h2 {
        font-size: 22px;
        font-weight: 300;
        color: #8b949e;
        margin-bottom: 8px;
    }
    .empty-state p {
        font-size: 14px;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# Session state initialisation
# ==============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = collection_has_documents()

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# Track how many files are indexed so we can show it
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0


# ==============================================================
# Sidebar
# ==============================================================

with st.sidebar:
    st.markdown("## 🤖 Knowledge Agent")
    st.markdown("---")

    # ── API Key ────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">OpenAI API Key</p>', unsafe_allow_html=True)

    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
        help="Your key is used only for this session and never stored."
    )

    if user_api_key and user_api_key.startswith("sk-"):
        
        os.environ["OPENAI_API_KEY"] = user_api_key

        try:
            from src.tools import mcp_tools as _mcp
            from openai import OpenAI as _OAI
            _mcp._openai_client = _OAI(api_key=user_api_key)
            st.session_state.api_key_set = True
        except Exception:
            st.session_state.api_key_set = True

        st.markdown(
            '<span class="status-dot dot-green"></span> API key accepted',
            unsafe_allow_html=True
        )
    elif user_api_key:
        st.markdown(
            '<span class="status-dot dot-red"></span> Key should start with sk-',
            unsafe_allow_html=True
        )
        st.session_state.api_key_set = False
    else:
        st.markdown(
            '<span style="font-size:12px;color:#484f58">'
            'Enter your key above to start. '
            '<a href="https://platform.openai.com/api-keys" '
            'target="_blank" style="color:#58a6ff">Get one here →</a>'
            '</span>',
            unsafe_allow_html=True
        )
        st.session_state.api_key_set = False

    # ── System status ──────────────────────────────────────────
    st.markdown('<p class="sidebar-section">System status</p>', unsafe_allow_html=True)

    # Qdrant
    try:
        from src.db.vector_store import get_qdrant_client
        get_qdrant_client().get_collections()
        st.markdown(
            '<span class="status-dot dot-green"></span> Qdrant connected',
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown(
            '<span class="status-dot dot-red"></span> Qdrant offline',
            unsafe_allow_html=True
        )

    # Ollama 
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if "localhost" in ollama_url or "127.0.0.1" in ollama_url:
        st.markdown(
            '<span class="status-dot dot-amber"></span> Ollama — local only',
            unsafe_allow_html=True
        )
    else:
        try:
            import httpx
            r = httpx.get(f"{ollama_url}/api/tags", timeout=2)
            if r.status_code == 200:
                st.markdown(
                    '<span class="status-dot dot-green"></span> Ollama running',
                    unsafe_allow_html=True
                )
            else:
                raise Exception()
        except Exception:
            st.markdown(
                '<span class="status-dot dot-red"></span> Ollama offline',
                unsafe_allow_html=True
            )

    # Knowledge base status
    if st.session_state.docs_loaded:
        st.markdown(
            '<span class="status-dot dot-green"></span> Knowledge base ready',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-dot dot-red"></span> No documents indexed',
            unsafe_allow_html=True
        )

    # ── Document upload ────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Upload documents</p>', unsafe_allow_html=True)

    if not st.session_state.api_key_set:
        
        st.caption("Enter your API key above to enable uploads.")
    else:
        uploaded_files = st.file_uploader(
            "PDF, TXT or DOCX",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        force_reindex = st.checkbox("Force re-index (replace existing)", value=False)

        if uploaded_files:
            # Show filenames so user knows the files were received
            st.markdown("**Files ready to upload:**")
            for f in uploaded_files:
                st.markdown(f"📄 `{f.name}`")

            if st.button("📤 Upload & Index", use_container_width=True, type="primary"):
                progress = st.progress(0, text="Preparing files...")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save each file to temp folder and update progress
                    for i, f in enumerate(uploaded_files):
                        dest = os.path.join(tmp_dir, f.name)
                        with open(dest, "wb") as out:
                            out.write(f.read())
                        progress.progress(
                            int((i + 1) / len(uploaded_files) * 50),
                            text=f"Saved {f.name}..."
                        )

                    # Index all files into Qdrant
                    progress.progress(60, text="Indexing into knowledge base...")
                    try:
                        upload_documents(data_path=tmp_dir, force=force_reindex)
                        progress.progress(100, text="✅ Done!")

                        # Update session state so chat unlocks immediately
                        st.session_state.docs_loaded = True
                        st.session_state.indexed_count = len(uploaded_files)

                        st.success(
                            f"✅ {len(uploaded_files)} file(s) uploaded and indexed! "
                            f"You can now start chatting."
                        )
                        time.sleep(1.5)
                        st.rerun()

                    except Exception as e:
                        progress.empty()
                        st.error(f"❌ Upload failed: {str(e)}")

        # Always show current knowledge base status below uploader
        st.markdown("---")
        if st.session_state.docs_loaded:
            st.markdown(
                '<span class="status-dot dot-green"></span> '
                'Knowledge base has documents — ready to chat.',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-dot dot-red"></span> '
                'No documents yet — upload files above to start.',
                unsafe_allow_html=True
            )

    # ── Chat controls ──────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Chat</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Info ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-size:12px;color:#484f58;line-height:1.6">'
        'Supports Arabic & English.<br>'
        'Your API key is used only in this session and never stored.<br><br>'
        '<a href="https://github.com/khaled-abdulaziz/company-knowledge-agent" '
        'target="_blank" style="color:#58a6ff">View source on GitHub →</a>'
        '</p>',
        unsafe_allow_html=True
    )


# ==============================================================
# Main — Chat area
# ==============================================================

st.markdown("### 💬 Ask anything about your company")

# Block everything until API key is entered
if not st.session_state.api_key_set:
    st.markdown("""
    <div class="empty-state">
        <h2>Enter your OpenAI API key to start</h2>
        <p>
            Your key is required to power the AI responses.<br>
            It is used only in this browser session and never stored.<br><br>
            <a href="https://platform.openai.com/api-keys"
               target="_blank" style="color:#3fb950">
               Get an API key from OpenAI →
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Block chat until documents are uploaded
if not st.session_state.docs_loaded:
    st.markdown("""
    <div class="empty-state">
        <h2>Upload documents to start chatting</h2>
        <p>
            Use the sidebar on the left to upload your PDF, TXT, or DOCX files.<br>
            Once indexed, you can ask questions in Arabic or English.<br><br>
            <span style="color:#d29922">
                ⬅️ Click "Upload & Index" in the sidebar after selecting your files.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Chat history ───────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h2>Knowledge base ready — ask anything</h2>
        <p>
            Ask about company policies, employee data, leave rules,<br>
            attendance records, or anything in your uploaded documents.<br><br>
            <span style="color:#3fb950">Supports Arabic and English.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            meta      = msg.get("meta", {})
            route     = meta.get("route", "")
            llm_used  = meta.get("llm_used", "")
            sensitive = meta.get("is_sensitive", False)

            badge_route = f'<span class="badge badge-route">⚡ {route}</span>' if route else ""
            badge_llm   = f'<span class="badge badge-llm">🤖 {llm_used}</span>' if llm_used else ""
            badge_lock  = '<span class="badge badge-lock">🔒 private</span>' if sensitive else ""

            st.markdown(f"""
            <div class="msg-assistant">
                {msg["content"]}
                <div class="meta-row">
                    {badge_route}{badge_llm}{badge_lock}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Chat input ─────────────────────────────────────────────────

question = st.chat_input("Ask a question in Arabic or English...")

if question:
    # Add user message to history
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
    })

    # Run the full agent pipeline
    with st.spinner("Thinking..."):
        try:
            result = run_agent(question)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["answer"],
                "meta": {
                    "route":        result["route"],
                    "llm_used":     result["llm_used"],
                    "is_sensitive": result["is_sensitive"],
                }
            })

        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"❌ Something went wrong: {str(e)}",
                "meta":    {}
            })

    st.rerun()