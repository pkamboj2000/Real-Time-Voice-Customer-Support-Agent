"""
Build (or rebuild) the FAISS vector index from knowledge base files.

Run this whenever you update the documents in knowledge_base/.
It reads all JSON files, chunks them, embeds them, and saves the
index to data/vector_index/.

Usage:
    python scripts/build_index.py
    # or
    make build-index
"""

import sys
import os
import time

# make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.knowledge_loader import KnowledgeBaseLoader
from src.retrieval.vector_store import VectorStore
from configs.settings import settings


def main():
    print("=" * 60)
    print("  Building vector index from knowledge base")
    print("=" * 60)

    start = time.time()

    # load documents
    loader = KnowledgeBaseLoader(kb_directory="knowledge_base")
    documents = loader.load_all()

    if not documents:
        print("\nNo documents found in knowledge_base/ directory.")
        print("Make sure you have .json files with the right format.")
        sys.exit(1)

    print(f"\nLoaded {len(documents)} document chunks from knowledge base")

    # show a breakdown by source file
    source_counts = {}
    for doc in documents:
        source_counts[doc.source_file] = source_counts.get(doc.source_file, 0) + 1

    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} chunks")

    # build index
    print("\nGenerating embeddings and building index...")
    store = VectorStore()
    store.build_index(documents)

    # persist to disk
    save_path = settings.vector_store_path
    store.save(save_path)

    elapsed = time.time() - start
    print(f"\nIndex built and saved to {save_path}/")
    print(f"Total documents indexed: {store.document_count}")
    print(f"Time taken: {elapsed:.1f} seconds")

    # quick sanity check — run a test query
    print("\n--- Sanity check ---")
    test_queries = [
        "How do I reset my password?",
        "What are your pricing plans?",
        "I want to cancel my subscription",
    ]

    for query in test_queries:
        results = store.search(query, top_k=2)
        if results:
            top_doc, top_score = results[0]
            print(f"  Q: {query}")
            print(f"  A: [{top_score:.3f}] {top_doc.title} ({top_doc.source_file})")
            print()
        else:
            print(f"  Q: {query}")
            print(f"  A: (no results)")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
