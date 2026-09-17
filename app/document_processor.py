"""
Data layer: load the knowledge base, clean its text, and split it into
retrieval-ready chunks.

Design notes
------------
* The knowledge base is a structured JSON of titled *sections*. Keeping the
  raw data structured (rather than one flat blob) lets every chunk carry
  honest provenance metadata (section title + category) that we surface to
  the user as "sources".
* Chunking uses a *sentence sliding window*. Window/stride are configurable.
  Small, overlapping windows give the retriever precise, self-contained units
  to match against while the overlap avoids splitting an answer across a
  boundary where neither half scores well on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base cannot be loaded or is malformed."""


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of knowledge with its provenance."""

    id: str
    text: str
    section_id: str
    section_title: str
    category: str

    def preview(self, max_chars: int = 240) -> str:
        if len(self.text) <= max_chars:
            return self.text
        return self.text[: max_chars - 1].rstrip() + "\u2026"


# Matches sentence terminators (. ! ?) followed by whitespace. A small set of
# common abbreviations is protected so we don't split "9:00 A.M." style tokens
# or "i.e." mid-sentence. This is intentionally lightweight and dependency-free.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "vs",
    "etc",
    "inc",
    "co",
    "ltd",
    "no",
    "approx",
    "i.e",
    "e.g",
    "a.m",
    "p.m",
}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters.

    Collapses runs of whitespace (including newlines and tabs) into single
    spaces and trims the result. Idempotent.
    """
    if not text:
        return ""
    # Drop non-printable control chars except standard whitespace.
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences, guarding common abbreviations."""
    text = clean_text(text)
    if not text:
        return []

    raw_parts = _SENTENCE_BOUNDARY.split(text)
    sentences: list[str] = []
    buffer = ""

    for part in raw_parts:
        candidate = f"{buffer} {part}".strip() if buffer else part
        # If the fragment ends in a protected abbreviation, keep accumulating
        # so the next fragment is glued on rather than treated as a new line.
        last_token = candidate.split()[-1].lower().rstrip(".") if candidate.split() else ""
        if last_token in _ABBREVIATIONS:
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


def load_knowledge_base(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the knowledge base JSON file.

    Raises
    ------
    KnowledgeBaseError
        If the file is missing, is not valid JSON, or lacks a non-empty
        ``sections`` list.
    """
    kb_path = Path(path)
    if not kb_path.exists():
        raise KnowledgeBaseError(
            f"Knowledge base file not found at '{kb_path.resolve()}'. "
            "Set KNOWLEDGE_BASE_PATH or place the file at the expected location."
        )
    if not kb_path.is_file():
        raise KnowledgeBaseError(f"'{kb_path}' exists but is not a file.")

    try:
        raw = kb_path.read_text(encoding="utf-8")
    except OSError as exc:  # permissions, decoding, etc.
        raise KnowledgeBaseError(f"Could not read knowledge base: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(
            f"Knowledge base is not valid JSON ({exc.msg} at line {exc.lineno})."
        ) from exc

    if not isinstance(data, dict):
        raise KnowledgeBaseError("Knowledge base root must be a JSON object.")
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        raise KnowledgeBaseError(
            "Knowledge base must contain a non-empty 'sections' array."
        )
    return data


def _chunk_section(
    section: dict[str, Any], window: int, stride: int
) -> list[Chunk]:
    """Turn one section into one or more overlapping sentence-window chunks."""
    section_id = str(section.get("id") or "unknown")
    section_title = clean_text(str(section.get("title") or section_id))
    category = clean_text(str(section.get("category") or "general"))
    content = clean_text(str(section.get("content") or ""))

    if not content:
        return []

    sentences = split_sentences(content)
    if not sentences:
        return []

    # If the whole section fits in one window, keep it intact.
    if len(sentences) <= window:
        windows = [sentences]
    else:
        windows = [
            sentences[i : i + window]
            for i in range(0, len(sentences), stride)
            if sentences[i : i + window]  # skip trailing empties
        ]
        # Ensure the tail sentences are always covered by the final window.
        if windows and windows[-1][-1] != sentences[-1]:
            windows.append(sentences[-window:])

    chunks: list[Chunk] = []
    seen_texts: set[str] = set()
    for idx, win in enumerate(windows):
        text = " ".join(win).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        chunks.append(
            Chunk(
                id=f"{section_id}#{idx}",
                text=text,
                section_id=section_id,
                section_title=section_title,
                category=category,
            )
        )
    return chunks


def build_chunks(
    knowledge_base: dict[str, Any], window: int = 3, stride: int = 2
) -> list[Chunk]:
    """Build the full ordered list of chunks from a loaded knowledge base."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")

    chunks: list[Chunk] = []
    for section in knowledge_base.get("sections", []):
        if not isinstance(section, dict):
            continue
        chunks.extend(_chunk_section(section, window=window, stride=stride))

    if not chunks:
        raise KnowledgeBaseError(
            "No chunks were produced from the knowledge base; check that "
            "sections contain non-empty 'content'."
        )
    return chunks


def load_and_chunk(
    path: str | Path, window: int = 3, stride: int = 2
) -> tuple[dict[str, Any], list[Chunk]]:
    """Convenience: load the KB and return ``(raw_kb, chunks)``."""
    kb = load_knowledge_base(path)
    chunks = build_chunks(kb, window=window, stride=stride)
    return kb, chunks
