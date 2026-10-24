"""Basic RAG pipeline: Loader -> Chunking -> Vector store -> Retriever -> Answer."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "meta_policies"


# ---------- Models (Tiger AI gateway, OpenAI-compatible) ----------

def get_embeddings():
    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=os.environ["TIGER_API_KEY"],
        base_url=os.environ["TIGER_BASE_URL"],
        check_embedding_ctx_length=False,  # send raw text, not OpenAI token ids
        chunk_size=100,  # texts per embedding request
    )


def get_llm():
    return ChatOpenAI(
        model=os.environ["CHAT_MODEL"],
        api_key=os.environ["TIGER_API_KEY"],
        base_url=os.environ["TIGER_BASE_URL"],
        temperature=0,
    )


# ---------- Step 1: Loader ----------

def load_documents():
    """Load every PDF in data/ — one Document per page."""
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf_path)).load())
    return docs


# ---------- Step 2: Chunking ----------

def split_documents(docs):
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


# ---------- Step 3: Vector store ----------

def create_vectorstore(chunks):
    """Embed chunks and save them in a local on-disk Qdrant collection."""
    return QdrantVectorStore.from_documents(
        chunks,
        embedding=get_embeddings(),
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )


def load_vectorstore():
    """Open the collection created by create_vectorstore()."""
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )


# ---------- Step 4: Retriever ----------

def get_retriever(vectorstore, k=4):
    """Return the top-k most similar chunks for a question."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ---------- Step 5: Answer ----------

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about Meta's Privacy Policy
and Instagram's Terms of Use. Use ONLY the context below. If the answer is not
in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(retriever, question):
    """Retrieve relevant chunks and ask the LLM. Returns (answer, source_docs)."""
    docs = retriever.invoke(question)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs
