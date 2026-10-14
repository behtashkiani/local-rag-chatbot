from __future__ import annotations

__version__ = "1.0.0"

"""
FastAPI application layer mapped to local Ollama execution logic.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .document_processor import KnowledgeBaseError, load_and_chunk
from .llm_service import LLMService

from .models import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    HealthResponse,
    Source,
)
from .retrieval_engine import TfidfRetriever, build_retriever
from .session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger("techstore.rag")


@dataclass
class AppState:
    """Container for the long-lived components shared across requests."""
    settings: Settings
    retriever: TfidfRetriever | None
    llm_service: LLMService | None
    sessions: SessionManager
    startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all components on startup; clean them up on shutdown."""
    settings = get_settings()
    retriever: TfidfRetriever | None = None
    llm_service: LLMService | None = None
    startup_error: str | None = None

    # 1) Knowledge base + retrieval index.
    try:
        _, chunks = load_and_chunk(
            settings.knowledge_base_path,
            window=settings.chunk_window,
            stride=settings.chunk_stride,
        )
        retriever = build_retriever(chunks)
        logger.info(
            "Knowledge base loaded: %d chunks, vocab size %d.",
            retriever.num_chunks,
            retriever.vocab_size,
        )
    except KnowledgeBaseError as exc:
        startup_error = f"Knowledge base unavailable: {exc}"
        logger.critical(startup_error)

    # 2) Initialize local Ollama Service (No API key check required)
    try:
        llm_service = LLMService()
    except Exception as exc:
        logger.critical("LLM service init failed: %s", exc)

    app.state.app_state = AppState(
        settings=settings,
        retriever=retriever,
        llm_service=llm_service,
        sessions=SessionManager(
            max_history_messages=settings.max_history_messages,
            max_sessions=settings.max_sessions,
        ),
        startup_error=startup_error,
    )

    try:
        yield
    finally:
        logger.info("Shutdown complete.")


app = FastAPI(
    title="TechStore Co. RAG Support Assistant",
    description=(
        "A retrieval-augmented customer support chatbot that answers strictly "
        "from the company knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# allow_credentials=True combined with a wildcard allow_origins makes
# Starlette reflect back whatever Origin header the request sent (since a
# literal "*" can't be paired with credentials per the CORS spec), which
# effectively allows any site to make credentialed requests. This API
# doesn't use cookies, so credentials aren't needed; keep it off unless a
# future cookie-based auth scheme is added alongside a real origin allowlist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_state(request: Request) -> AppState:
    """Dependency that exposes the shared application state."""
    return request.app.state.app_state


# --- Endpoints ------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "TechStore Co. RAG Support Assistant",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Report readiness of the knowledge base and LLM configuration."""
    kb_loaded = state.retriever is not None
    return HealthResponse(
        status="ok" if (kb_loaded and state.llm_service is not None) else "degraded",
        knowledge_base_loaded=kb_loaded,
        chunk_count=state.retriever.num_chunks if kb_loaded else 0,
        model="llama3.2",
        llm_configured=state.llm_service is not None,
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    payload: ChatRequest, state: AppState = Depends(get_state)
) -> ChatResponse:
    """Answer a user message using retrieval-augmented generation."""
    if state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.startup_error or "Knowledge base is not available.",
        )
    if state.llm_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local LLM assistant service is not initialized.",
        )

    settings = state.settings

    # 1) Retrieve chunks from local matrix.
    results = state.retriever.retrieve(
        query=payload.message,
        top_k=settings.top_k,
        min_score=settings.min_similarity,
    )

    # Convert results array to list of dictionaries for internal service format mapping
    retrieved_chunks_dict = [
        {
            "section_title": r.chunk.section_title,
            "text_preview": r.chunk.preview()
        }
        for r in results
    ]

    # 2) Generate response via local Ollama instance
    history = state.sessions.get_history(payload.session_id)
    try:
        raw_text_answer = await state.llm_service.generate_answer(
            query=payload.message,
            retrieved_chunks=retrieved_chunks_dict,
            history=history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    # 3) Persist the exchange so follow-ups have context.
    state.sessions.add_exchange(
        payload.session_id, payload.message, raw_text_answer
    )

    # 4) Build the transparent source list.
    sources = [
        Source(
            chunk_id=r.chunk.id,
            section_title=r.chunk.section_title,
            category=r.chunk.category,
            score=round(r.score, 4),
            text_preview=r.chunk.preview(),
        )
        for r in results
    ]

    return ChatResponse(
        session_id=payload.session_id,
        answer=raw_text_answer,
        sources=sources,
        retrieved_count=len(results),
        degraded=False,
        model="llama3.2",
    )


@app.delete(
    "/api/sessions/{session_id}",
    response_model=ClearSessionResponse,
    tags=["chat"],
)
async def clear_session(
    session_id: str, state: AppState = Depends(get_state)
) -> ClearSessionResponse:
    """Forget a conversation's in-memory history."""
    cleared = state.sessions.clear(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=cleared)


# --- Global error handling ------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last line of defense: log the real error, return a safe generic message."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )
