import os
import re
import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
# Prevent PyTorch from hogging cores needed by LLM and STT
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "Missing RAG dependencies. Please install: "
        "pip install langchain langchain-community langchain-text-splitters faiss-cpu"
    )

import numpy as np
from langchain_core.embeddings import Embeddings


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Embedding Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NativeONNXEmbeddings(Embeddings):
    """Dependency-free LangChain wrapper for pure ONNX Runtime embeddings."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_file = f"{model_path}/model_quantized.onnx"
        if not os.path.exists(onnx_file):
            onnx_file = f"{model_path}/model.onnx"

        self.session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64),
        }
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]

        # Mean pooling over attention mask
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.repeat(
            attention_mask[:, :, np.newaxis], token_embeddings.shape[2], axis=2
        )
        sum_emb = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_emb / sum_mask

        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Keyword Scorer (no extra dependencies — pure Python)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Scorer:
    """
    Lightweight BM25 implementation for keyword re-ranking.

    Why hybrid search for FAQ?
    - Semantic search catches meaning ("how much does a course cost?")
    - BM25 catches exact terms ("IELTS 4.5", "14 million", "GPA 9.0")
    - Combining both gives best of both worlds for FAQ with numbers/scores.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_freqs: List[dict] = []
        self.idf: dict = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[str]) -> None:
        self.corpus = [self._tokenize(d) for d in documents]
        self.n_docs = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.n_docs, 1)

        # Document frequency per term
        df: dict = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # IDF (log smoothed)
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

        # Term frequency per document
        self.doc_freqs = [
            {t: doc.count(t) for t in set(doc)} for doc in self.corpus
        ]

    def score(self, query: str) -> List[float]:
        """Return BM25 score for each document in corpus."""
        if self.n_docs == 0:
            return []
        tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            dl = len(doc)
            s = 0.0
            for term in tokens:
                tf = self.doc_freqs[i].get(term, 0)
                idf = self.idf.get(term, 0.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / max(den, 1e-9)
            scores.append(s)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Offline Retrieval-Augmented Generation (RAG) Module.

    Search strategy: HYBRID (FAISS semantic + BM25 keyword re-ranking)
    - Semantic FAISS catches meaning-level matches ("fees", "costs")
    - BM25 catches exact tokens ("IELTS 4.5", "14 million VND", "GPA 9.0")
    - Final score = 0.6 * semantic + 0.4 * BM25 (normalized)

    Chunking strategy: Section-aware with heading injection
    - Splits on ## markdown headings (each section = independent chunk)
    - Injects parent heading prefix into every sub-chunk
    - FAISS never matches a bare heading — it always sees the full context
    """

    # Direct-answer confidence threshold (L2 distance).
    # Below this → speak chunk directly without LLM.
    DIRECT_ANSWER_THRESHOLD = 0.75

    def __init__(
        self,
        docs_dir: str = "data/docs",
        faiss_index_path: str = "data/faiss_index",
        embedding_model_name: str = "models/embed_onnx",
        chunk_size: int = 350,    # Reduced from 600 — finer granularity
        chunk_overlap: int = 80,  # Reduced from 150
    ):
        self.docs_dir = Path(docs_dir)
        self.faiss_index_path = Path(faiss_index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("[RAG] Initializing RAG Pipeline (Hybrid Mode)...")

        # 1. Embedding model
        print(f"[RAG] Loading embedding model: {embedding_model_name}")
        self.embeddings = NativeONNXEmbeddings(model_path=embedding_model_name)

        # 2. FAISS index
        self.vector_store = None
        self._chunk_texts: List[str] = []  # Parallel list for BM25
        self._bm25 = BM25Scorer()
        self._load_or_create_index()

        # 3. Prompt template (for LLM fallback path)
        self.prompt_template = PromptTemplate(
            template=(
                "You are a helpful voice assistant at Greenwich Vietnam University.\n"
                "Answer ONLY using the provided context. Be concise (2-3 sentences max).\n"
                "If the context doesn't contain the answer, say: "
                "'I don't have that information in my knowledge base.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )
        print("[RAG] Ready. Strategy: Hybrid (FAISS + BM25).")

    # ── Index management ──────────────────────────────────────────────────────

    def _load_or_create_index(self) -> None:
        """Load FAISS index from disk if it exists, and rebuild BM25 scorer."""
        if self.faiss_index_path.exists():
            print(f"[RAG] Loading FAISS index from {self.faiss_index_path}...")
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.faiss_index_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Rebuild BM25 from stored docstore
                self._rebuild_bm25_from_docstore()
                print(f"[RAG] FAISS loaded. BM25 fitted on {self._bm25.n_docs} chunks.")
            except Exception as e:
                print(f"[RAG] Failed to load index: {e}. Will re-ingest.")
                self.vector_store = None
        else:
            print("[RAG] No FAISS index found. Run ingest_documents() to build it.")

    def _rebuild_bm25_from_docstore(self) -> None:
        """Re-fit BM25 scorer using all texts in the FAISS docstore."""
        if self.vector_store is None:
            return
        try:
            self._chunk_texts = [
                doc.page_content
                for doc in self.vector_store.docstore._dict.values()
            ]
            self._bm25.fit(self._chunk_texts)
        except Exception as e:
            print(f"[RAG] BM25 rebuild warning: {e}")

    # ── Section-aware chunking ────────────────────────────────────────────────

    def _section_aware_split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents by markdown ## sections and inject parent heading
        into every sub-chunk so FAISS always sees full context.

        Example output chunk:
            [Greenwich Vietnam > Tuition Fee Structure]
            * English Level Fee: Each level takes ~2 months and costs 14 million VND.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()

            # Extract document title (# heading)
            doc_title = ""
            title_match = re.match(r"^#\s+(.+)", text, re.MULTILINE)
            if title_match:
                doc_title = title_match.group(1).strip()

            # Split on ## headings (keep the heading with its content)
            sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Extract this section's heading
                heading_match = re.match(r"^#{1,4}\s+(.+)", section)
                section_heading = heading_match.group(1).strip() if heading_match else ""

                def _make_chunk(
                    content: str,
                    part: int = 0,
                    _title: str = doc_title,
                    _heading: str = section_heading,
                ) -> Document:
                    """Closure uses default-arg capture to avoid Python late-binding."""
                    if _title and _heading:
                        prefix = f"[{_title} > {_heading}]\n"
                    elif _title:
                        prefix = f"[{_title}]\n"
                    else:
                        prefix = ""
                    return Document(
                        page_content=prefix + content,
                        metadata={**metadata, "section": _heading, "part": part},
                    )

                if len(section) <= self.chunk_size:
                    all_chunks.append(_make_chunk(section))
                else:
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=["\n\n", "\n", ". ", " "],
                    )
                    sub_chunks = sub_splitter.split_text(section)
                    for i, sub in enumerate(sub_chunks):
                        all_chunks.append(_make_chunk(sub, part=i))

        return all_chunks

    def ingest_documents(self) -> None:
        """
        Read all .md and .txt files from docs_dir, chunk with section-aware
        splitter, build FAISS index, and fit BM25 scorer.

        Delete data/faiss_index/ before calling to force full re-index.
        """
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RAG] Created empty docs dir: {self.docs_dir}")
            return

        file_paths = list(self.docs_dir.glob("**/*.md")) + list(
            self.docs_dir.glob("**/*.txt")
        )
        if not file_paths:
            print(f"[RAG] No documents found in {self.docs_dir}.")
            return

        documents: List[Document] = []
        print(f"[RAG] Found {len(file_paths)} documents. Loading...")
        for fp in file_paths:
            try:
                loader = TextLoader(str(fp), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"[RAG] Error loading {fp}: {e}")

        # Section-aware chunking with heading injection
        chunks = self._section_aware_split(documents)
        print(f"[RAG] Section-aware split: {len(chunks)} chunks. Building FAISS index...")

        # Build FAISS — always fresh (never append; re-ingest = full rebuild)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Persist
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.faiss_index_path))

        # Fit BM25
        self._chunk_texts = [c.page_content for c in chunks]
        self._bm25.fit(self._chunk_texts)

        # Clear stale lru_cache entries from the old index
        self._cached_context.cache_clear()

        print(f"[RAG] Index saved to {self.faiss_index_path}. BM25 fitted. Ready.")

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def _hybrid_search(
        self, query: str, top_k: int = 4, semantic_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search: FAISS semantic + BM25 keyword, score-fused.

        Args:
            semantic_weight: 0.6 = 60% semantic, 40% BM25. Tune if needed.
                             Increase toward 1.0 for more meaning-based results.
                             Decrease toward 0.5 for more exact-keyword results.

        Returns:
            List of (Document, combined_score) sorted best-first (lower = better).
        """
        if self.vector_store is None:
            return []

        # 1. Semantic search (L2 distance — lower is better)
        sem_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        if not sem_results:
            return []

        # 2. BM25 scores for the same retrieved documents (higher is better)
        bm25_scores_raw = self._bm25.score(query)

        # Build a lookup: chunk text → BM25 score
        bm25_lookup: dict = {}
        if bm25_scores_raw and self._chunk_texts:
            max_bm25 = max(bm25_scores_raw) or 1.0
            for text, score in zip(self._chunk_texts, bm25_scores_raw):
                bm25_lookup[text[:200]] = score / max_bm25  # normalize 0→1

        # 3. Normalize semantic scores (0→1, lower L2 = higher quality)
        sem_scores = [s for _, s in sem_results]
        max_sem = max(sem_scores) or 1.0

        combined: List[Tuple[Document, float]] = []
        for doc, sem_score in sem_results:
            sem_norm = sem_score / max_sem  # 0=best, 1=worst
            bm25_norm = bm25_lookup.get(doc.page_content[:200], 0.0)  # 0=worst, 1=best

            # Combined score: lower = better
            # semantic contributes its normalized distance; BM25 inverts (1-bm25)
            combined_score = (
                semantic_weight * sem_norm
                + (1 - semantic_weight) * (1.0 - bm25_norm)
            )
            combined.append((doc, combined_score))

        # Sort best-first and deduplicate using a normalized content key
        # (first 150 chars of stripped lowercase text) to catch cross-doc duplicates
        combined.sort(key=lambda x: x[1])
        seen: set = set()
        deduped: List[Tuple[Document, float]] = []
        for doc, score in combined:
            # Normalize: strip markdown prefix bracket, lowercase, first 150 chars
            raw = doc.page_content
            norm = re.sub(r"^\[.*?\]\s*", "", raw).strip().lower()[:150]
            if norm not in seen:
                seen.add(norm)
                deduped.append((doc, score))
            if len(deduped) >= top_k:
                break

        return deduped

    # ── Public API ────────────────────────────────────────────────────────────

    def answer_direct(
        self, user_query: str, top_k: int = 1
    ) -> Tuple[str, float]:
        """
        FAST PATH: Return the single best-matching chunk directly WITHOUT LLM.

        Always returns only ONE chunk — combining multiple chunks produces
        out-of-order, rambling answers when the query is vague.
        The LLM path (get_context) handles multi-chunk synthesis correctly.

        Returns:
            (answer_text, combined_score)
            score < 0.50 → excellent specific match, speak directly
            score < 0.75 → good match, speak directly
            score >= 0.75 → poor/vague match, fall through to LLM
        """
        if self.vector_store is None or not user_query.strip():
            return "", 999.0

        results = self._hybrid_search(user_query, top_k=1)  # always single best
        if not results:
            return "", 999.0

        best_doc, best_score = results[0]

        if best_score >= self.DIRECT_ANSWER_THRESHOLD:
            return "", best_score  # Not confident enough — let LLM handle it

        return best_doc.page_content, best_score

    @lru_cache(maxsize=128)
    def _cached_context(self, user_query: str, top_k: int) -> str:
        """Cached context retrieval for repeated queries."""
        results = self._hybrid_search(user_query, top_k=top_k)
        valid = [doc.page_content for doc, score in results if score < 0.9]
        # Use plain blank-line separator — "---" can bleed into LLM output as "dash dash dash"
        return "\n\n".join(valid) if valid else ""

    def get_context(self, user_query: str, top_k: int = 2) -> str:
        """
        Return retrieved context as a string (for injection into LLM prompt).
        Uses hybrid search + LRU cache for repeated queries.
        """
        if self.vector_store is None or not user_query:
            return ""
        # Cache on original stripped query (not lowercased) — lowercasing can
        # reduce FAISS semantic similarity for models trained with mixed-case input.
        return self._cached_context(user_query.strip(), top_k)

    def query(self, user_query: str, top_k: int = 3) -> str:
        """
        Full RAG query: hybrid search → prompt → LLM.
        Used as fallback when answer_direct() score is too low.
        """
        if self.vector_store is None:
            return "My knowledge base is empty. Please ingest documents first."
        if not user_query or not user_query.strip():
            return "I didn't hear a question."

        try:
            results = self._hybrid_search(user_query, top_k=top_k)
            valid = [doc.page_content for doc, score in results if score < 0.9]
            context_text = "\n---\n".join(valid) if valid else "No relevant context found."

            formatted_prompt = self.prompt_template.format(
                context=context_text, question=user_query
            )
            # Note: self.llm is injected externally (llama-cpp LLM via inference_worker)
            # This method returns the prompt if no llm attached, so caller can use it.
            if not hasattr(self, "llm") or self.llm is None:
                return context_text  # Return raw context — caller will prompt the LLM

            return self.llm.invoke(formatted_prompt).strip()

        except Exception as e:
            print(f"[RAG] Query error: {e}")
            return "An internal error occurred while searching my knowledge base."


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    # Ensure project root is on path when run as: python core/rag.py
    _root = Path(__file__).parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    rag = RAGPipeline(
        docs_dir=str(_root / "data" / "docs"),
        faiss_index_path=str(_root / "data" / "faiss_index"),
        embedding_model_name=str(_root / "models" / "embed_onnx"),
    )

    # Only ingest if no index exists — never contaminate with test data
    if not (Path(rag.faiss_index_path)).exists():
        rag.ingest_documents()

    for q in [
        "What is the IELTS score required?",
        "How much does each English level cost?",
        "What is the admission fee?",
        "What scholarships are available?",
    ]:
        answer, score = rag.answer_direct(q)
        print(f"\n[Q] {q}")
        print(f"[A] (score={score:.3f}) {answer[:120]}")
