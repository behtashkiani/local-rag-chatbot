"""
Application configuration.

All tunable parameters live here and are sourced from environment variables
(optionally via a local ``.env`` file). Nothing else in the codebase reads
``os.environ`` directly, so configuration stays in one auditable place.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read (in order of precedence) from: explicit constructor
    arguments, environment variables, then a ``.env`` file. Unknown env vars
    are ignored so the same environment can host other services.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM / Anthropic ---------------------------------------------------
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Loaded from ANTHROPIC_API_KEY.",
    )
    model: str = Field(
        default="claude-opus-4-8",
        description="Anthropic model identifier used for generation.",
    )
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    # Temperature 0.0 keeps answers deterministic and tightly grounded, which
    # is what we want for a policy-answering support bot.
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout: float = Field(
        default=30.0, gt=0, description="Per-request HTTP timeout in seconds."
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="SDK-level automatic retries for transient API errors.",
    )

    # --- Knowledge base / retrieval ---------------------------------------
    knowledge_base_path: str = Field(default="data/knowledge_base.json")
    top_k: int = Field(default=3, ge=1, le=20)
    # Chunks scoring below this cosine similarity are treated as irrelevant
    # and dropped, so the model is never fed noise that invites hallucination.
    min_similarity: float = Field(default=0.05, ge=0.0, le=1.0)
    chunk_window: int = Field(
        default=3, ge=1, description="Sentences per chunk (sliding window size)."
    )
    chunk_stride: int = Field(
        default=2, ge=1, description="Sentence step between consecutive chunks."
    )

    # --- Session management ------------------------------------------------
    max_history_messages: int = Field(
        default=10,
        ge=0,
        description="Max stored messages per session (user + assistant combined).",
    )
    max_sessions: int = Field(
        default=1000,
        ge=1,
        description="Max concurrent sessions kept in memory (LRU eviction).",
    )

    # --- HTTP / server -----------------------------------------------------
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins. Lock this down in production.",
    )

    @field_validator("chunk_stride")
    @classmethod
    def _stride_not_larger_than_window(cls, v: int, info) -> int:
        window = info.data.get("chunk_window")
        if window is not None and v > window:
            raise ValueError("chunk_stride must be <= chunk_window")
        return v

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
