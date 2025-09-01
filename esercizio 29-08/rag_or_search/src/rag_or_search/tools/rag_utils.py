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
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

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

    Creates and configures an Azure OpenAI embeddings client using environment
    variables. Prompts the user for an API key if not already set.

    Returns
    -------
    AzureOpenAIEmbeddings
        Configured embeddings instance ready for use.

    Raises
    ------
    ValueError
        If required environment variables are not set.

    Examples
    --------
    >>> embeddings = get_embeddings()
    >>> print(type(embeddings))
    <class 'langchain_openai.embeddings.AzureOpenAIEmbeddings'>
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

    Creates a chat model instance configured to use Azure OpenAI services
    based on the provided settings and environment variables.

    Args
    ----
    settings : Settings
        The settings object providing the model deployment environment variable name.

    Returns
    -------
    Any
        A chat model instance compatible with LangChain interfaces.

    Raises
    ------
    RuntimeError
        If required Azure OpenAI environment variables are not set.

    Examples
    --------
    >>> settings = Settings()
    >>> llm = get_llm_from_lmstudio(settings)
    >>> print(type(llm))
    <class 'langchain.chat_models.base.ChatOpenAI'>
    """
    base_url = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    model_name = os.getenv(settings.lmstudio_model_env)

    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY devono essere "
            "impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings.lmstudio_model_env} con il nome "
            f"del modello caricato in LM Studio."
        )

    return init_chat_model(
        model_name, model_provider="azure_openai",
        api_key=api_key, api_version=api_version
    )

def load_documents(file_format, file_path):
    """Load documents from disk by format.

    Loads documents from a file using the specified format. Currently supports
    Markdown files which are split by "---" separators.

    Args
    ----
    file_format : str
        Short format specifier, currently only "md" supported.
    file_path : str
        Path to the file to load.

    Returns
    -------
    list of Document
        List of parsed Document objects with metadata.

    Raises
    ------
    ValueError
        If an unsupported format is requested.
    FileNotFoundError
        If the specified file path does not exist.

    Examples
    --------
    >>> docs = load_documents("md", "example.md")
    >>> print(len(docs))
    3
    >>> print(type(docs[0]))
    <class 'langchain.schema.document.Document'>
    """
    if file_format == "md":
        return load_md_documents(file_path)
    raise ValueError(f"Unsupported file format: {file_format}")

def load_md_documents(file_path: str) -> List[Document]:
    """Read a Markdown file into LangChain Document objects.

    Parses a Markdown file and splits it into Document objects based on
    "---" separators, with each section becoming a separate document.

    Args
    ----
    file_path : str
        Path to the Markdown file to read.

    Returns
    -------
    list of Document
        List of Document objects, one per section split on "---".

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.

    Examples
    --------
    >>> docs = load_md_documents("example.md")
    >>> print(len(docs))
    2
    >>> print(docs[0].metadata['source'])
    example.md
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
    """Create a small English corpus with metadata and source for citations.

    Generates a predefined set of Document objects containing information about
    LangChain, FAISS, sentence transformers, RAG pipelines, and MMR retrieval.
    Each document includes metadata with an ID and source for citation purposes.

    Returns
    -------
    list of Document
        List of 5 Document objects with predefined content about AI/ML topics.

    Examples
    --------
    >>> docs = simulate_corpus()
    >>> print(len(docs))
    5
    >>> print(docs[0].metadata['id'])
    doc1
    >>> print('LangChain' in docs[0].page_content)
    True
    """
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
                "FAISS is a library for efficient similarity search and clustering of "
                "dense vectors. It supports exact and approximate nearest neighbor "
                "search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings "
                "suitable for semantic search, clustering, and information retrieval. "
                "The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the "
                "LLM produces an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during "
                "retrieval. It helps avoid redundant chunks and improves coverage of "
                "different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Apply robust splitting to optimize retrieval.

    Splits documents into smaller chunks using RecursiveCharacterTextSplitter
    with configurable chunk size and overlap to optimize retrieval performance.

    Args
    ----
    docs : list of Document
        Input documents to split into chunks.
    settings : Settings
        Chunking configuration including chunk_size and chunk_overlap.

    Returns
    -------
    list of Document
        List of Document objects representing the resulting chunks.

    Examples
    --------
    >>> docs = simulate_corpus()
    >>> settings = Settings(chunk_size=500, chunk_overlap=50)
    >>> chunks = split_documents(docs, settings)
    >>> print(len(chunks) > len(docs))
    True
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

    Creates a FAISS vector store from document chunks and saves it to disk
    for future retrieval operations.

    Args
    ----
    chunks : list of Document
        Document chunks to index in the vector store.
    embeddings : Any
        Embeddings model used to create vector representations.
    persist_dir : str
        Directory path where the FAISS index will be saved.

    Returns
    -------
    FAISS
        The created and persisted FAISS vector store instance.

    Examples
    --------
    >>> docs = simulate_corpus()
    >>> embeddings = get_embeddings()
    >>> vs = build_faiss_vectorstore(docs, embeddings, "test_index")
    >>> print(type(vs))
    <class 'langchain_community.vectorstores.faiss.FAISS'>
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

    Attempts to load an existing FAISS index from disk. If no index exists,
    creates a new one from the provided documents and saves it for future use.

    Args
    ----
    settings : Settings
        Configuration including persist_dir for index storage.
    embeddings : Any
        Embeddings model for creating vector representations.
    docs : list of Document
        Documents to use for building the index if it doesn't exist.

    Returns
    -------
    FAISS
        Either the loaded existing vector store or a newly built one.

    Examples
    --------
    >>> settings = Settings(persist_dir="test_index")
    >>> embeddings = get_embeddings()
    >>> docs = simulate_corpus()
    >>> vs = load_or_build_vectorstore(settings, embeddings, docs)
    >>> print(type(vs))
    <class 'langchain_community.vectorstores.faiss.FAISS'>
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization'
        # per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """Configure a retriever, optionally using MMR for diversity.

    Creates a retriever from a FAISS vector store with configurable search
    parameters. Supports both similarity search and MMR (Maximal Marginal Relevance)
    for diversity in retrieved results.

    Args
    ----
    vector_store : FAISS
        The FAISS vector store to wrap as a retriever.
    settings : Settings
        Retrieval configuration including search type, k, fetch_k, and mmr_lambda.

    Returns
    -------
    Any
        A retriever object compatible with LangChain interfaces.

    Examples
    --------
    >>> vs = FAISS.from_documents(docs, embeddings)
    >>> settings = Settings(search_type="mmr", k=3)
    >>> retriever = make_retriever(vs, settings)
    >>> print(type(retriever))
    <class 'langchain_community.vectorstores.base.VectorStoreRetriever'>
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda
            },
        )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.k},
    )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Prepare a prompt context string with [source:...] citations.

    Formats a list of Document objects into a single string with source
    citations for use in prompt templates.

    Args
    ----
    docs : list of Document
        List of Document objects to format.

    Returns
    -------
    str
        Formatted string with each document's content prefixed by its source.

    Examples
    --------
    >>> docs = simulate_corpus()
    >>> formatted = format_docs_for_prompt(docs[:2])
    >>> print('[source:' in formatted)
    True
    >>> print('LangChain' in formatted)
    True
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def rag_answer(question: str, chain) -> str:
    """Execute a RAG chain for a single question.

    Runs a RAG (Retrieval-Augmented Generation) chain to answer a question
    using retrieved context and a language model.

    Args
    ----
    question : str
        The user query to answer.
    chain : Any
        A LangChain chain or runnable that supports the invoke method.

    Returns
    -------
    str
        The generated answer text from the RAG chain.

    Examples
    --------
    >>> # Assuming chain is a properly configured RAG chain
    >>> answer = rag_answer("What is LangChain?", chain)
    >>> print(type(answer))
    <class 'str'>
    """
    return chain.invoke(question)

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the contents of the top-k retrieved chunks.

    Retrieves the top-k most relevant document chunks for a given question
    and returns them as a mapping from source to content.

    Args
    ----
    retriever : Any
        The retriever to query for relevant documents.
    question : str
        Query text to search for.
    k : int
        Number of contexts to retrieve and return.

    Returns
    -------
    dict
        Dictionary mapping source names to page content strings.

    Examples
    --------
    >>> retriever = make_retriever(vs, settings)
    >>> contexts = get_contexts_for_question(retriever, "What is FAISS?", 2)
    >>> print(len(contexts))
    2
    >>> print(type(contexts))
    <class 'dict'>
    """
    docs = retriever.invoke(question)[:k]
    return {d.metadata.get("source", f"doc{d.id}"): d.page_content for d in docs}

def rag_search(question: str, k: int):
    """Perform a simple RAG retrieval flow and return contexts.

    Executes a complete RAG retrieval pipeline including document loading,
    vector store creation/loading, and context retrieval for a given question.

    Args
    ----
    question : str
        The user query to search for.
    k : int
        Number of contexts to retrieve and return.

    Returns
    -------
    dict
        Dictionary mapping source names to page content of retrieved chunks.

    Examples
    --------
    >>> contexts = rag_search("What is LangChain?", 3)
    >>> print(len(contexts))
    3
    >>> print(type(contexts))
    <class 'dict'>
    >>> print('LangChain' in str(contexts.values()))
    True
    """
    settings = SETTINGS

    settings.k = k  # aggiorna k dinamicamente

    # 1) Componenti
    embeddings = get_embeddings()

    # 2) Dati simulati e indicizzazione (load or build)
    docs = simulate_corpus()
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    retrieved_docs = get_contexts_for_question(retriever, question, k)

    return retrieved_docs
