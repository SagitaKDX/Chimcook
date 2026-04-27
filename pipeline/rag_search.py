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

# ── Markdown stripping patterns ───────────────────────────────────────────────
# Removes all markdown syntax so TTS speaks clean prose instead of
# "hashtag hashtag Greenwich Vietnam" or "asterisk asterisk bold asterisk asterisk"
_MD_HEADING   = re.compile(r"^#{1,6}\s+", re.MULTILINE)          # ## Heading
_MD_BOLD_IT   = re.compile(r"\*{1,3}(.+?)\*{1,3}")               # **bold** / *italic*
_MD_BRACKET   = re.compile(r"^\[.*?\]\s*", re.MULTILINE)          # [Title > Section] prefix
_MD_BULLET    = re.compile(r"^[\*\-]\s+", re.MULTILINE)           # * bullet or - bullet
_MD_LINK      = re.compile(r"\[([^\]]+)\]\([^)]+\)")              # [text](url)
_MD_BACKTICK  = re.compile(r"`+(.+?)`+")                          # `inline code`
_MD_SECTION_NUM = re.compile(r"^\d+(\.\d+)*\s+", re.MULTILINE)   # "1.3 " / "2.1.4 " prefixes
_MULTI_NL     = re.compile(r"\n{3,}")

# Set True to print raw FAISS + BM25 scores on every query (useful when tuning)
DEBUG_SCORES: bool = False


def _strip_markdown(text: str) -> str:
    """
    Convert raw markdown chunk text to clean TTS-ready prose.

    Removes:
      - [Title > Section] chunk prefixes injected by section-aware splitter
      - # ## ### headings markers
      - **bold**, *italic*, ***both*** markers (keeps the inner text)
      - * bullet / - bullet list markers
      - [text](url) links (keeps link text)
      - `inline code` backticks (keeps inner text)
      - Excess blank lines

    Example
    -------
    "### 3.2 Scholarship Opportunities\n* **Green Talent:** 50% fee reduction."
        → "Scholarship Opportunities\nGreen Talent: 50% fee reduction."
    """
    t = _MD_BRACKET.sub("", text)      # remove [Title > Section] prefixes
    t = _MD_HEADING.sub("", t)         # remove ## markers (keep heading text)
    t = _MD_SECTION_NUM.sub("", t)     # remove "1.3 " / "2.1.4 " section numbering
    t = _MD_LINK.sub(r"\1", t)         # [text](url) → text
    t = _MD_BOLD_IT.sub(r"\1", t)      # **bold** / *italic* → plain text
    t = re.sub(r"\*+", "", t)          # mop up any remaining lone asterisks (**Key:** edge case)
    t = _MD_BULLET.sub("", t)          # remove bullet markers
    t = _MD_BACKTICK.sub(r"\1", t)     # `code` → code
    t = "\n".join(line.lstrip() for line in t.splitlines())  # remove indent left by bullet removal
    t = _MULTI_NL.sub("\n\n", t)       # collapse 3+ blank lines
    return t.strip()


# ── Spoken-language formatter ─────────────────────────────────────────────────
# Converts numbered list items to ordinal words for natural TTS output
_NUMBERED_STEP = re.compile(r"^(\d+)\.\s+", re.MULTILINE)
_ORDINALS = {"1": "First, ", "2": "Next, ", "3": "Then, ",
             "4": "After that, ", "5": "Also, ", "6": "Finally, "}


def _format_for_speech(raw_chunk: str) -> str:
    """
    Convert a raw markdown chunk into natural spoken language.

    Pipeline
    --------
    1. Strip all markdown syntax
    2. Extract the first line as a topic title and make it a natural intro
    3. Remove standalone sub-section header lines (short title-case labels)
    4. Trim chunk bleed — drop a trailing line that is a new top-level heading
    5. Convert '1. text' → 'First, text', '2. text' → 'Next, text' ...
    6. Join into flowing prose (single space between lines)

    Examples
    --------
    Raw:  'Scholarship Opportunities\nThe 2026 fund is 250 Billion.\nGreen Talent: 20-100%.\n4. Registration Steps'
    Out:  'Regarding scholarship opportunities: The 2026 fund is 250 Billion. Green Talent: 20 to 100 percent.'

    Raw:  'Online Registration\n1. Prepare your ID.\n2. Pay the fee.\n3. Confirmation within 2 weeks.'
    Out:  'Here is how to register online: First, prepare your I D. Next, pay the fee. Then, confirmation within 2 weeks.'
    """
    clean = _strip_markdown(raw_chunk)
    if not clean:
        return ""

    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return ""

    def _is_heading_like(line: str) -> bool:
        """True if the line looks like a section title rather than a sentence."""
        words = line.split()
        return (
            1 <= len(words) <= 7
            and not line.endswith(".")
            and not line.endswith("?")
            and not line.endswith(",")
            and not _NUMBERED_STEP.match(line)
        )

    # Step 2: use first line as natural intro if it looks like a heading
    first, rest = lines[0], lines[1:]
    if _is_heading_like(first) and rest:
        intro = f"Regarding {first.lower().rstrip(':')}: "
        lines = rest
    else:
        intro = ""
        lines = [first] + rest

    # Step 3: remove standalone sub-section headers inside the body
    # e.g. "Online Registration", "Required Documents" on their own line
    def _is_sub_header(line: str) -> bool:
        words = line.split()
        return (
            1 <= len(words) <= 5
            and not line.endswith(".")
            and not line.endswith(",")
            and not re.search(r"\d", line)
            and not _NUMBERED_STEP.match(line)
            and all(w[0].isupper() for w in words if w)
        )

    lines = [ln for ln in lines if not _is_sub_header(ln)] or lines

    # Step 4: trim chunk bleed — drop trailing heading-like line from the next section
    if len(lines) > 1 and _is_heading_like(lines[-1]):
        lines = lines[:-1]

    # Step 5: convert numbered steps to ordinal words
    result: list = []
    for ln in lines:
        m = _NUMBERED_STEP.match(ln)
        if m:
            num = m.group(1)
            body = ln[m.end():]
            result.append(_ORDINALS.get(num, f"Step {num}, ") + body)
        else:
            result.append(ln)

    # Step 6: join into flowing prose
    return (intro + " ".join(result)).strip()


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
            # Format as natural spoken prose — no LLM, zero latency
            spoken = _format_for_speech(answer)
            print(
                f"[RAG] ⚡ Direct answer — score={score:.3f}, {elapsed_ms}ms "
                f"(skipping LLM)"
            )
            return spoken, score

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
