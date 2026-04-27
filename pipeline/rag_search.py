"""
pipeline/rag_search.py
======================
RAG context retrieval helper for the inference pipeline.

Extracted from inference_worker.py so it can be tested and debugged
independently without running the full audio pipeline.

Responsibilities
----------------
• Strip conversational filler from the user query so the embedding model
  focuses on the actual topic (e.g. "Greenwich Vietnam tuition fees" not
  "can you tell me a little bit about the fees")
• Try the fast-path direct answer first (no LLM needed, <100ms)
• Fall back to get_context() for LLM prompt injection when confidence is low

Debugging tips
--------------
• Run this file directly to test retrieval against the live FAISS index:
      python pipeline/rag_search.py "what is the IELTS requirement"

• If the direct answer fires too often (low-quality chunks speaking directly),
  raise RAGSearch.DIRECT_SCORE_THRESHOLD toward 0.6.

• If direct answers never fire (LLM always used), lower the threshold toward 0.8.

• To inspect raw scores: set DEBUG_SCORES = True below.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

# ── Filler words to strip before embedding ────────────────────────────────────
_FILLER_PATTERN = re.compile(
    r"\b("
    r"can you|could you|please|tell me|i want to know"
    r"|what (is|are|was|were)|do you know"
    r"|i('d| would) like to know|a little bit about"
    r"|give me|explain|describe|talk about|say something about"
    r")\b",
    flags=re.IGNORECASE,
)

# ── University alias normalisation ────────────────────────────────────────────
_UNI_PATTERN = re.compile(r"\b(?:my )?university\b", flags=re.IGNORECASE)

# Set True to print raw FAISS + BM25 scores on every query (useful when tuning)
DEBUG_SCORES: bool = False


class RAGSearch:
    """
    Thin wrapper around RAGPipeline that handles query cleaning and the
    fast-path / fallback decision.

    Usage (inside InferenceWorker)::

        searcher = RAGSearch(assistant)

        # Fast path — no LLM
        direct, score = searcher.direct_answer(user_text)
        if direct:
            speak(direct)
            return

        # Slow path — inject context into LLM prompt
        context = searcher.get_context(user_text)
        prompt  = build_prompt(context, user_text)
        reply   = llm.generate(prompt)
        speak(reply)
    """

    # L2-distance threshold for the direct answer fast path.
    # score < this  → answer is confident enough, skip LLM entirely.
    # score >= this → fall through to LLM.
    DIRECT_SCORE_THRESHOLD: float = 0.75

    def __init__(self, assistant) -> None:
        # Accepts the VoiceAssistant instance (avoids circular import)
        self._rag = getattr(assistant._components, "rag", None)

    # ── Public API ────────────────────────────────────────────────────────────

    def clean_query(self, text: str) -> str:
        """
        Strip conversational filler and normalise university aliases so the
        embedding model receives a clean keyword-focused query.

        Examples
        --------
        "Can you tell me about the tuition fees?"
            → "tuition fees"
        "What are the IELTS requirements at my university?"
            → "IELTS requirements at Greenwich Vietnam"
        """
        q = _UNI_PATTERN.sub("Greenwich Vietnam", text)
        q = _FILLER_PATTERN.sub(" ", q)
        q = " ".join(q.split())          # collapse whitespace
        return q if q.strip() else text  # never return empty

    def direct_answer(self, text: str) -> Tuple[str, float]:
        """
        Try to answer directly from the knowledge base without calling the LLM.

        Returns
        -------
        (answer_text, score)
            If score < DIRECT_SCORE_THRESHOLD: answer_text is ready to speak.
            If score >= DIRECT_SCORE_THRESHOLD: answer_text is empty, use LLM.
        """
        if self._rag is None:
            return "", 999.0

        query = self.clean_query(text)
        t0 = time.time()
        answer, score = self._rag.answer_direct(query)
        elapsed_ms = int((time.time() - t0) * 1000)

        if DEBUG_SCORES:
            print(f"[RAG direct] query='{query}' score={score:.3f} ({elapsed_ms}ms)")

        if answer and score < self.DIRECT_SCORE_THRESHOLD:
            print(
                f"[RAG] ⚡ Direct answer — score={score:.3f}, {elapsed_ms}ms "
                f"(skipping LLM)"
            )
            return answer, score

        return "", score

    def get_context(self, text: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context chunks for injection into the LLM prompt.

        Returns empty string if the knowledge base has nothing useful.
        """
        if self._rag is None:
            return ""

        query = self.clean_query(text)
        t0 = time.time()
        context = self._rag.get_context(query, top_k=top_k)
        elapsed_ms = int((time.time() - t0) * 1000)

        if context:
            print(
                f"[RAG] ✅ Context found — {len(context)} chars, {elapsed_ms}ms"
            )
        else:
            print(
                f"[RAG] ⚠️  No relevant context for: '{text[:60]}'"
            )

        if DEBUG_SCORES:
            print(f"[RAG context] query='{query}'\n{context[:200]}")

        return context


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test — run: python pipeline/rag_search.py "your question here"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _root = Path(__file__).parent.parent
    sys.path.insert(0, str(_root))

    from core.rag import RAGPipeline

    rag = RAGPipeline()

    # Minimal stub so RAGSearch works without a full VoiceAssistant
    class _Stub:
        class _components:
            pass
    stub = _Stub()
    stub._components.rag = rag  # type: ignore[attr-defined]

    searcher = RAGSearch(stub)  # type: ignore[arg-type]

    questions = sys.argv[1:] or [
        "What is the IELTS requirement?",
        "How much does each English level cost?",
        "What scholarships are available?",
        "Tell me about AI major requirements",
    ]

    for q in questions:
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"   cleaned → '{searcher.clean_query(q)}'")

        answer, score = searcher.direct_answer(q)
        if answer:
            print(f"   DIRECT  (score={score:.3f}): {answer[:120]}")
        else:
            ctx = searcher.get_context(q)
            print(f"   CONTEXT (score={score:.3f}): {ctx[:120]}")
