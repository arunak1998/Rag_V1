"""Record real answers (without RAG vs with RAG) and vector data for the website.

Writes frontend/src/data/demo.json. Run after ingest.py:  uv run export_demo_data.py
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np

from rag import (
    PROMPT,
    format_docs,
    get_llm,
    load_documents,
    load_vectorstore,
    split_documents,
)

OUTPUT = Path("frontend/src/data/demo.json")

QUESTIONS = [
    "What is the effective date of Meta's current Privacy Policy?",
    "What does section 4.2 of Instagram's Terms of Use cover?",
    "What is the minimum age to use Instagram?",
    "How long does Meta keep my data?",
    "Who won the FIFA World Cup in 2022?",
]


def short_source(metadata):
    name = Path(metadata["source"]).name
    return "Instagram Terms of Use" if name.startswith("Terms") else "Meta Privacy Policy"


def estimate_tokens(text):
    return len(text) // 4  # rough rule of thumb: ~4 characters per token


# ---------- Without RAG vs With RAG ----------

def record_question(vectorstore, llm, question):
    """Ask the LLM alone, then ask it with retrieved chunks. Return both."""
    without_rag = llm.invoke(question).content

    results = vectorstore.similarity_search_with_score(question, k=4)
    docs = [doc for doc, _ in results]
    context = format_docs(docs)
    with_rag = (PROMPT | llm).invoke({"context": context, "question": question}).content

    return {
        "question": question,
        "without_rag": {"answer": without_rag, "tokens": estimate_tokens(question)},
        "with_rag": {
            "answer": with_rag,
            "tokens": estimate_tokens(PROMPT.format(context=context, question=question)),
            "chunks": [
                {
                    "id": doc.metadata["_id"],
                    "score": round(score, 4),
                    "source": short_source(doc.metadata),
                    "page": doc.metadata["page"] + 1,
                    "text": doc.page_content,
                }
                for doc, score in results
            ],
        },
    }


# ---------- Pipeline stats ----------

def find_overlap(first, second):
    """Longest end of `first` that is also the start of `second`."""
    for size in range(min(len(first), len(second)), 0, -1):
        if first[-size:] == second[:size]:
            return first[-size:]
    return ""


def pipeline_stats():
    pages = load_documents()
    chunks = split_documents(pages)

    pages_per_file = {}
    for page in pages:
        name = short_source(page.metadata)
        pages_per_file[name] = pages_per_file.get(name, 0) + 1

    # First pair of neighbouring chunks that share overlapping text
    overlap_example = None
    for first, second in zip(chunks, chunks[1:]):
        overlap = find_overlap(first.page_content, second.page_content)
        if len(overlap) > 80:
            overlap_example = {
                "first": first.page_content,
                "second": second.page_content,
                "overlap": overlap,
            }
            break

    return {
        "files": [{"name": n, "pages": c} for n, c in pages_per_file.items()],
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "avg_chunk_chars": round(sum(len(c.page_content) for c in chunks) / len(chunks)),
        "sample_page": pages[0].page_content[:600],
        "overlap_example": overlap_example,
    }


# ---------- Vector DB view ----------

def vector_map(vectorstore, embeddings, recorded):
    """Project all stored vectors (and each question) to 2D with PCA."""
    points, _ = vectorstore.client.scroll(
        vectorstore.collection_name, limit=1000, with_vectors=True, with_payload=True
    )
    matrix = np.array([p.vector for p in points])
    mean = matrix.mean(axis=0)
    _, _, components = np.linalg.svd(matrix - mean, full_matrices=False)
    project = lambda vecs: (np.array(vecs) - mean) @ components[:2].T

    coords = project(matrix)
    map_points = [
        {
            "id": p.id,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "source": short_source(p.payload["metadata"]),
            "page": p.payload["metadata"]["page"] + 1,
            "preview": p.payload["page_content"][:140],
        }
        for p, (x, y) in zip(points, coords)
    ]

    for item in recorded:
        qx, qy = project([embeddings.embed_query(item["question"])])[0]
        item["query_point"] = {"x": round(float(qx), 4), "y": round(float(qy), 4)}

    sample = points[0]
    sample_point = {
        "id": sample.id,
        "vector": [round(v, 4) for v in sample.vector],
        "payload": sample.payload,
    }
    return map_points, sample_point, len(sample.vector)


def main():
    vectorstore = load_vectorstore()
    llm = get_llm()

    recorded = []
    for question in QUESTIONS:
        print(f"Recording: {question}")
        recorded.append(record_question(vectorstore, llm, question))

    print("Collecting pipeline stats...")
    stats = pipeline_stats()

    print("Projecting vectors to 2D...")
    map_points, sample_point, dimensions = vector_map(vectorstore, vectorstore.embeddings, recorded)

    data = {
        "captured_on": date.today().isoformat(),
        "chat_model": os.environ["CHAT_MODEL"],
        "embedding_model": os.environ["EMBEDDING_MODEL"],
        "collection": vectorstore.collection_name,
        "dimensions": dimensions,
        "prompt_template": PROMPT.messages[0].prompt.template,
        "questions": recorded,
        "pipeline": stats,
        "vector_map": map_points,
        "sample_point": sample_point,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
