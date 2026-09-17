"""
API data contracts (Pydantic v2 models).

These models define and validate the boundary between the outside world and
the application. Validation here means malformed input is rejected with a 422
before it ever reaches business logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming payload for the chat endpoint."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Opaque client-supplied conversation identifier.",
        examples=["web-user-42"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's natural-language question.",
        examples=["What is your return policy for opened items?"],
    )

    @field_validator("session_id", "message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace only")
        return cleaned


class Source(BaseModel):
    """A single retrieved knowledge-base chunk surfaced for transparency."""

    chunk_id: str
    section_title: str
    category: str
    score: float = Field(description="Cosine similarity in [0, 1].")
    text_preview: str = Field(description="Truncated chunk text.")


class ChatResponse(BaseModel):
    """Structured chat reply returned to the client."""

    session_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieved_count: int = Field(
        description="Number of chunks that passed the relevance threshold."
    )
    degraded: bool = Field(
        default=False,
        description="True if the answer is a fallback due to an LLM error.",
    )
    model: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    knowledge_base_loaded: bool
    chunk_count: int
    model: str
    llm_configured: bool


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool
