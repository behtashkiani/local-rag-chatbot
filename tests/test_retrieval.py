"""
Standalone demonstration and sanity-check for the retrieval pipeline.

This script needs NO Anthropic API key. It loads the knowledge base, builds the
TF-IDF index, runs a battery of representative customer questions, prints the
top retrieved chunks with their scores, and asserts that the expected source
sections surface. Run it to verify the retrieval math end-to-end:

    python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly (python tests/test_retrieval.py) by adding the repo
# root to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.document_processor import load_and_chunk  # noqa: E402
from app.retrieval_engine import build_retriever  # noqa: E402

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"

# (question, substring expected to appear in the top result's section id)
TEST_CASES: list[tuple[str, str]] = [
    ("What time does the airport store open on Sunday?", "airport"),
    ("Is there a restocking fee for opened items?", "returns"),
    ("How much does the Premium service tier cost per month?", "premium"),
    ("Do you offer free shipping?", "shipping"),
    ("Can I return opened earbuds?", "returns-exceptions"),
    ("What payment methods do you accept?", "payment"),
    ("Are you open on Christmas Day?", "holiday"),
    ("How do I track my order?", "faq-account"),
]


def main() -> int:
    kb, chunks = load_and_chunk(KB_PATH, window=3, stride=2)
    retriever = build_retriever(chunks)

    print(f"Knowledge base: '{kb.get('company')}'")
    print(f"Chunks indexed: {retriever.num_chunks} | Vocabulary: {retriever.vocab_size}")
    print("=" * 78)

    failures = 0
    top1_hits = 0
    recall_hits = 0
    for question, expected_section in TEST_CASES:
        results = retriever.retrieve(question, top_k=3, min_score=0.05)
        print(f"\nQ: {question}")
        if not results:
            print("   (no results above threshold)")
            failures += 1
            continue
        for rank, r in enumerate(results, start=1):
            print(
                f"   {rank}. score={r.score:.3f} "
                f"[{r.chunk.section_id}] {r.chunk.section_title}"
            )
        top_section = results[0].chunk.section_id
        in_topk = any(expected_section in r.chunk.section_id for r in results)
        top1 = expected_section in top_section
        top1_hits += int(top1)
        # recall@k is the figure that matters: every retrieved chunk is fed to
        # the model, so a relevant chunk anywhere in the top-k grounds the answer.
        recall_hits += int(in_topk)
        if not in_topk:
            failures += 1
        marker = "rank-1" if top1 else ("in top-k" if in_topk else "MISSING")
        print(f"   -> expected section '{expected_section}': {marker}")

    print("\n" + "=" * 78)
    # An empty / whitespace query must return nothing (guardrail check).
    assert retriever.retrieve("   ", top_k=3) == [], "empty query must return []"
    # A totally out-of-domain query should return little or nothing.
    oov = retriever.retrieve("quantum chromodynamics lagrangian", top_k=3, min_score=0.05)
    print(f"Out-of-domain query returned {len(oov)} result(s) above threshold.")

    total = len(TEST_CASES)
    print(f"\nRecall@3 (relevant chunk retrieved): {recall_hits}/{total}")
    print(f"Top-1 accuracy (best chunk ranked first): {top1_hits}/{total}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
