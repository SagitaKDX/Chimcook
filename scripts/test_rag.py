#!/usr/bin/env python3
"""RAG diagnostic: shows exact FAISS similarity scores for test queries."""
import sys
sys.path.insert(0, '/app')

from core.rag import RAGPipeline

rag = RAGPipeline()

queries = [
    'Greenwich Vietnam history',
    'tuition fees scholarships',
    'scholarship criteria GPA',
    'how much does it cost',
    'admissions requirements',
    'established 2009',
]

print("\n=== FAISS Score Diagnostics (lower L2 score = better match) ===\n")
for q in queries:
    results = rag.vector_store.similarity_search_with_score(q, k=3)
    print(f"Query: {q!r}")
    for doc, score in results:
        marker = "✅" if score <= 0.8 else "❌"
        print(f"  {marker} score={score:.4f}  chunk={doc.page_content[:90]!r}")
    print()
