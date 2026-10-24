"""Ask a few test questions and print answers with their source pages."""

from pathlib import Path

from rag import answer_question, get_retriever, load_vectorstore

QUESTIONS = [
    "What information does Meta collect about users?",
    "How long does Meta keep my data?",
    "What is the minimum age to use Instagram?",
    "What is the capital of France?",  # not in the docs -> should say "I don't know"
]

retriever = get_retriever(load_vectorstore())

for question in QUESTIONS:
    print("=" * 80)
    print(f"Q: {question}")

    print("\nStep 4: Retrieved chunks:")
    answer, docs = answer_question(retriever, question)
    for doc in docs:
        source = Path(doc.metadata["source"]).name[:40]
        print(f"  - {source} (page {doc.metadata['page'] + 1}): {doc.page_content[:80]!r}")

    print(f"\nStep 5: Answer:\n{answer}\n")
