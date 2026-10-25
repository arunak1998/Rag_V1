# Basic RAG with LangChain + Qdrant

A simple Retrieval-Augmented Generation (RAG) app that answers questions about
**Meta's Privacy Policy** and **Instagram's Terms of Use**.

Plain LangChain, local Qdrant, and a Streamlit UI — no LangGraph, no extra layers.

## Pipeline

| Step | What happens | Tool |
|---|---|---|
| 1. Loader | Read every PDF in `data/`, one Document per page | `PyPDFLoader` |
| 2. Chunking | Split pages into 1000-char chunks with 200-char overlap | `RecursiveCharacterTextSplitter` |
| 3. Vector store | Embed chunks and save them on disk | `QdrantVectorStore` + `gemini-embedding-001` |
| 4. Retriever | Find the 4 chunks most similar to the question | `vectorstore.as_retriever(k=4)` |
| 5. Answer | Send question + chunks to the LLM, answer only from context | `ChatOpenAI` (`gemini-3.1-flash-lite`) |

All five steps live in [`rag.py`](rag.py), one function per step.

## Files

| File | Purpose |
|---|---|
| `rag.py` | The RAG pipeline (loader, chunking, vector store, retriever, answer) |
| `ingest.py` | Run once: load → chunk → embed → store in Qdrant |
| `verify.py` | Ask test questions and print answers with source pages |
| `app.py` | Streamlit UI |
| `export_demo_data.py` | Record answers with and without RAG as JSON for the demo website |
| `data/` | The source PDFs |

## Setup

Requires [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).

**1. Clone the repo**

```bash
git clone https://github.com/arunak1998/Rag_V1.git
cd Rag_V1
```

**2. Install dependencies** (uv creates the `.venv` for you)

```bash
uv sync
```

**3. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and replace `your-api-key-here` with your key. The models are called
through an OpenAI-compatible gateway, configured in `.env`.

## Run

```bash
uv run ingest.py              # build the vector store (once, or when PDFs change)
uv run verify.py              # test questions in the terminal
uv run streamlit run app.py   # UI at http://localhost:8501
```

> Local Qdrant allows one process at a time on `qdrant_data/`, so stop the
> Streamlit app before running `ingest.py` or `verify.py`.
