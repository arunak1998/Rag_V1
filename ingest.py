"""Run once: load PDFs, chunk them, and store embeddings in Qdrant."""

from rag import create_vectorstore, load_documents, split_documents

print("Step 1: Loading PDFs...")
docs = load_documents()
print(f"  Loaded {len(docs)} pages")

print("Step 2: Chunking...")
chunks = split_documents(docs)
print(f"  Created {len(chunks)} chunks")
print(f"  Sample chunk:\n  {chunks[0].page_content[:200]!r}")

print("Step 3: Embedding + storing in Qdrant (this can take a minute)...")
vectorstore = create_vectorstore(chunks)
count = vectorstore.client.count(vectorstore.collection_name).count
print(f"  Stored {count} vectors in collection '{vectorstore.collection_name}'")

print("Done! Now run: uv run verify.py  or  uv run streamlit run app.py")
