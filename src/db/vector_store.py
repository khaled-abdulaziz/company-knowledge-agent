# ==============================================================
# vector_store.py — Qdrant + LlamaIndex RAG
#
# Supports both:
#   - Local Docker   → no API key needed
#   - Qdrant Cloud   → QDRANT_API_KEY set in .env or HF secrets
# ==============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

# File readers
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader

# Embeddings + LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Qdrant
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ==============================================================
# Constants from .env
# ==============================================================

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")        # None when running locally
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "company_docs")

# text-embedding-3-small outputs 1536-dim vectors
EMBEDDING_DIMENSION = 1536


# ==============================================================
# Configure LlamaIndex global models
# ==============================================================

def _configure_models():
    """
    Sets the global LlamaIndex LLM and embedding model.
    Reads OPENAI_API_KEY fresh every call so it picks up the key
    injected by the user in the Streamlit sidebar at runtime.
    """
    # Read key fresh — not from module-level constant —
    # because the user may have entered it after the module loaded
    api_key = os.getenv("OPENAI_API_KEY")

    Settings.llm = LlamaOpenAI(
        model=OPENAI_MODEL,
        api_key=api_key,
        temperature=0.1
    )
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )


# ==============================================================
# Qdrant Client (singleton)
# ==============================================================

_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    """
    Returns a singleton Qdrant client.

    Two modes depending on whether QDRANT_API_KEY is set:
      - Cloud mode → connects to Qdrant Cloud using URL + API key
      - Local mode → connects to local Docker using host + port

    Cloud URL example:
      1bdce6fc-6841-41a1-a366-b78543dc8aeb.eu-west-2-0.aws.cloud.qdrant.io
    """
    global _qdrant_client
    if _qdrant_client is None:
        api_key = os.getenv("QDRANT_API_KEY")

        if api_key:
            # ☁️ Cloud mode — QDRANT_HOST is the cluster URL without https://
            # Example: 1bdce6fc-xxxx.eu-west-2-0.aws.cloud.qdrant.io
            _qdrant_client = QdrantClient(
                url=f"https://{QDRANT_HOST}",
                api_key=api_key,
            )
            print(f"☁️ Qdrant Cloud connected: {QDRANT_HOST}")
        else:
            # 🐳 Local mode — connects to Docker on localhost
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
            print(f"🐳 Qdrant Local connected: {QDRANT_HOST}:{QDRANT_PORT}")

    return _qdrant_client


# ==============================================================
# Collection Manager
# ==============================================================

def _ensure_collection_exists(client: QdrantClient):
    """
    Creates the Qdrant collection ONLY if it does not already exist.
    Never wipes existing data on restart.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created new Qdrant collection: '{COLLECTION_NAME}'")
    else:
        print(f"📦 Collection '{COLLECTION_NAME}' already exists — skipping creation.")


# ==============================================================
# Check if collection has documents
# ==============================================================

def collection_has_documents() -> bool:
    """
    Returns True if the Qdrant collection already has vectors stored.
    Used to skip re-uploading documents that are already indexed.
    Returns False safely if Qdrant is unreachable.
    """
    try:
        client = get_qdrant_client()
        info   = client.get_collection(COLLECTION_NAME)
        count  = info.points_count
        return count is not None and count > 0
    except Exception:
        # Qdrant offline or collection doesn't exist yet — not an error
        return False


# ==============================================================
# Document Loader — supports PDF + TXT + DOCX
# ==============================================================

def _load_documents(data_path: str) -> list:
    """
    Loads all supported files from the given folder.
    Supports: PDF, TXT, DOCX — Arabic + English content.

    Args:
        data_path (str): Path to folder containing company documents.

    Returns:
        list: LlamaIndex Document objects ready for indexing.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"❌ Data folder not found: {data_path}")

    # SimpleDirectoryReader handles multiple file types automatically
    loader = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,                           # scan subfolders too
        required_exts=[".pdf", ".txt", ".docx"]   # supported formats
    )

    documents = loader.load_data()

    if not documents:
        raise ValueError(f"⚠️ No documents found in: {data_path}")

    print(f"📄 Loaded {len(documents)} document chunks from '{data_path}'")
    return documents


# ==============================================================
# Upload Documents to Qdrant (run ONCE per new document set)
# ==============================================================

def upload_documents(data_path: str = "data/manuals", force: bool = False):
    """
    Uploads and indexes company documents into Qdrant.

    Behavior:
        - If collection already has data AND force=False → skips upload.
        - If collection is empty OR force=True → uploads documents.

    Args:
        data_path (str): Folder with your PDF/TXT/DOCX files.
        force (bool): Set True to re-upload even if data exists.

    Usage:
        upload_documents()              # safe — skips if already uploaded
        upload_documents(force=True)    # force re-index after adding new docs
    """
    _configure_models()
    client = get_qdrant_client()
    _ensure_collection_exists(client)

    # Skip if already uploaded and not forcing
    if collection_has_documents() and not force:
        print("✅ Documents already in Qdrant. Skipping upload. (Use force=True to re-index)")
        return

    # Load files from disk
    documents = _load_documents(data_path)

    # Chunk documents for better retrieval
    # chunk_size=512  → ~400 words per chunk, good for Arabic + English
    # chunk_overlap=64 → overlap preserves context across chunk boundaries
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64
    )

    # Build and store index in Qdrant
    vector_store    = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        storage_context=storage_context,
        show_progress=True
    )

    print(f"✅ Successfully uploaded and indexed {len(documents)} chunks into Qdrant.")


# ==============================================================
# Query Engine — used by the agent for every question
# ==============================================================

def get_query_engine(top_k: int = 4):
    """
    Returns a LlamaIndex query engine connected to the existing Qdrant collection.
    Does NOT reload or re-upload documents — reads from what's already stored.

    Args:
        top_k (int): Number of document chunks to retrieve per query.
                     Higher = more context but slower. Default 4 is balanced.

    Returns:
        QueryEngine: Ready to call .query("your question") on.

    Usage:
        engine = get_query_engine()
        response = engine.query("What is the vacation policy?")
        print(response)
    """
    _configure_models()
    client = get_qdrant_client()
    _ensure_collection_exists(client)

    vector_store    = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="tree_summarize"   # best for multi-chunk summarization
    )