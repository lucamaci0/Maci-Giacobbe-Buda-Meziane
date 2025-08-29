from __future__ import annotations

"""Utility functions for building and querying a simple RAG pipeline.

This module centralizes helpers to:

- Initialize embeddings and a chat model (Azure OpenAI compatible)
- Load or simulate documents
- Split documents into chunks
- Build or load a FAISS vector store
- Configure a retriever and format contexts for prompts
- Execute basic RAG retrieval flows

Notes
-----
These utilities expect Azure OpenAI environment variables to be configured
(``AZURE_API_BASE``, ``AZURE_API_KEY``, ``AZURE_API_VERSION``, and an embedding
deployment). When running locally, secrets may be prompted via ``getpass``.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import getpass

# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    """Configuration values for RAG utilities.

    Attributes
    ----------
    persist_dir : str
        Directory where the FAISS index is stored.
    chunk_size : int
        Maximum characters per chunk during splitting.
    chunk_overlap : int
        Overlap between adjacent chunks to preserve context.
    search_type : str
        Retrieval mode, ``"mmr"`` or ``"similarity"``.
    k : int
        Number of final retrieved documents.
    fetch_k : int
        Initial candidate pool size for MMR.
    mmr_lambda : float
        Trade-off for MMR, 0=max diversity, 1=max relevance.
    lmstudio_model_env : str
        Environment variable name holding the Azure OpenAI deployment name.
    """

    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 1000
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 1                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 1         # 0 = diversificazione massima, 1 = pertinenza massima
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "MODEL"  # nome del modello in LM Studio, via env var



SETTINGS = Settings()


# =========================
# Componenti di base
# =========================

def get_embeddings():
    """Initialize Azure OpenAI embeddings client.

    Prompts for a key if ``AZURE_API_KEY`` is not set.

    Returns
    -------
    AzureOpenAIEmbeddings
        Configured embeddings instance.
    """

    if not os.getenv("AZURE_API_KEY"):
        os.environ["AZURE_API_KEY"] = getpass.getpass(
            "Enter your AzureOpenAI API key: "
        )
    
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
    )


def get_llm_from_lmstudio(settings: Settings):
    """Initialize a chat model pointing to Azure OpenAI.

    Parameters
    ----------
    settings : Settings
        The settings object providing the model deployment env var.

    Returns
    -------
    Any
        A chat model instance compatible with LangChain interfaces.
    """
    base_url = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    model_name = os.getenv(settings.lmstudio_model_env)

    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY devono essere impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings.lmstudio_model_env} con il nome del modello caricato in LM Studio."
        )

    return init_chat_model(model_name, model_provider="azure_openai", api_key=api_key, api_version=api_version)

def load_documents(file_format, file_path):
    """Load documents from disk by format.

    Parameters
    ----------
    file_format : str
        Short format specifier, currently only ``"md"`` supported.
    file_path : str
        Path to the file.

    Returns
    -------
    list of Document
        Parsed documents.

    Raises
    ------
    ValueError
        If an unsupported format is requested.
    """
    if file_format == "md":
        return load_md_documents(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
def load_md_documents(file_path: str) -> List[Document]:
    """Read a Markdown file into LangChain ``Document`` objects.

    Parameters
    ----------
    file_path : str
        Path to the Markdown file.

    Returns
    -------
    list of Document
        One document per section, split on "---".

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    documents = []
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    documents = [
        Document(
            page_content=section,
            metadata={"source": os.path.basename(file_path), "section": i}
        )
        for i, section in enumerate(content.split("---"), start=1)
        if section.strip()
    ]
    
    return documents
    

def simulate_corpus() -> List[Document]:
    """Create a small English corpus with metadata and ``source`` for citations."""
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Apply robust splitting to optimize retrieval.

    Parameters
    ----------
    docs : list of Document
        Input documents to split.
    settings : Settings
        Chunking configuration.

    Returns
    -------
    list of Document
        The resulting chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings, persist_dir: str) -> FAISS:
    """Build and persist a FAISS index from document chunks.

    Parameters
    ----------
    chunks : list of Document
        Documents to index.
    embeddings : Any
        Embeddings model used by FAISS.
    persist_dir : str
        Directory to persist the index.

    Returns
    -------
    FAISS
        The created vector store instance.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, docs: List[Document]) -> FAISS:
    """Load a persisted FAISS index or build one from documents.

    Parameters
    ----------
    settings : Settings
        Configuration including ``persist_dir``.
    embeddings : Any
        Embeddings model for FAISS.
    docs : list of Document
        Documents from which to build the index if missing.

    Returns
    -------
    FAISS
        Loaded or newly built vector store.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """Configure a retriever, optionally using MMR for diversity.

    Parameters
    ----------
    vector_store : FAISS
        The vector store to wrap as a retriever.
    settings : Settings
        Retrieval configuration.

    Returns
    -------
    Any
        A retriever object compatible with LangChain.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Prepare a prompt context string with ``[source:...]`` citations."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def rag_answer(question: str, chain) -> str:
    """Execute a RAG chain for a single question.

    Parameters
    ----------
    question : str
        The user query.
    chain : Any
        A LangChain chain or runnable supporting ``invoke``.

    Returns
    -------
    str
        The generated answer text.
    """
    return chain.invoke(question)

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the contents of the top-k retrieved chunks.

    Parameters
    ----------
    retriever : Any
        The retriever to query.
    question : str
        Query text.
    k : int
        Number of contexts to return.

    Returns
    -------
    dict
        Mapping from ``source`` to ``page_content``.
    """
    docs = docs = retriever.invoke(question)[:k]
    return {d.metadata.get("source", f"doc{d.id}") : d.page_content for d in docs}

def rag_search(question: str, k: int):
    """Perform a simple RAG retrieval flow and return contexts.

    Parameters
    ----------
    question : str
        The user query.
    k : int
        Number of contexts to retrieve.

    Returns
    -------
    dict
        Mapping from ``source`` to ``page_content`` of retrieved chunks.
    """
    settings = SETTINGS
    
    settings.k = k  # aggiorna k dinamicamente

    # 1) Componenti
    embeddings = get_embeddings()
    llm = get_llm_from_lmstudio(settings)

    # 2) Dati simulati e indicizzazione (load or build)
    docs = simulate_corpus()
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)
    
    retrieved_docs = get_contexts_for_question(retriever, question, k)
    
    return retrieved_docs
       