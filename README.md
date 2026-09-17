# TechStore Co. — RAG Customer Support Assistant

A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers
customer questions **strictly** from a company knowledge base. Built with
FastAPI, a dependency-light NumPy TF-IDF retriever, and the Anthropic Claude
API (`claude-opus-4-8`).

The defining property of this system is **grounding**: the model is given only
the retrieved knowledge-base passages and is instructed to refuse anything not
supported by them, which keeps answers truthful and prevents hallucinated
policies, prices, or hours.

---

## Architecture

```
techstore-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py              # Typed settings from env / .env (single source)
│   ├── models.py              # Pydantic request/response schemas + validation
│   ├── document_processor.py  # Load KB, clean text, sentence-window chunking
│   ├── retrieval_engine.py    # NumPy TF-IDF index + cosine-similarity search
│   ├── llm_service.py         # Grounded prompt + async Claude call + error handling
│   ├── session_manager.py     # Bounded, LRU in-memory conversation history
│   └── main.py                # FastAPI app: lifespan wiring + endpoints
├── data/
│   └── knowledge_base.json    # Mock "TechStore Co." knowledge base
├── tests/
│   └── test_retrieval.py      # Runnable retrieval sanity check (no API key)
├── requirements.txt
├── .env.example
└── README.md
```

### Request flow (`POST /api/chat`)

```
client → validate (Pydantic) → retrieve top-k chunks (TF-IDF + cosine)
       → build grounded prompt (system rules + history + <knowledge_base>)
       → Claude (claude-opus-4-8) → persist exchange → JSON answer + sources
```

The clean separation matters: `retrieval_engine` knows nothing about HTTP or
the LLM; `llm_service` knows nothing about retrieval internals (it just takes
already-retrieved chunks); `main` orchestrates. You can replace the retriever
with a vector DB, or the model provider, by touching exactly one file.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
uvicorn app.main:app --reload
```

- Interactive API docs: <http://127.0.0.1:8000/docs>
- Health/readiness: <http://127.0.0.1:8000/health>

### Verify retrieval without an API key

```bash
python tests/test_retrieval.py
```

This loads the knowledge base, builds the index, and runs representative
questions. On the bundled data it reports **Recall@3 = 8/8** (the relevant
section is always among the retrieved chunks) and confirms the empty-query and
out-of-domain guardrails.

---

## Example requests

```bash
# Ask a question
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"web-1","message":"Is there a fee to return an opened laptop?"}'

# Follow-up in the same session (history is used to resolve the reference)
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"web-1","message":"And what about custom-built ones?"}'

# Out-of-scope question → polite, grounded refusal
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"web-1","message":"What is the capital of France?"}'
```

A response looks like:

```json
{
  "session_id": "web-1",
  "answer": "Opened but undamaged items can be returned within 30 days...",
  "sources": [
    {"chunk_id": "returns#0", "section_title": "Return and Refund Policy",
     "category": "returns", "score": 0.30, "text_preview": "..."}
  ],
  "retrieved_count": 3,
  "degraded": false,
  "model": "claude-opus-4-8"
}
```

---

## Key design decisions

- **Grounding is enforced in three layers.** (1) A strict system prompt; (2)
  the current question's context injected inside explicit `<knowledge_base>`
  delimiters so the model can separate company facts from the user's words and
  from its own training data; (3) a similarity threshold (`MIN_SIMILARITY`) so
  irrelevant chunks are dropped rather than fed in as noise. When nothing
  relevant is found, the model is told so and refuses politely.
- **TF-IDF with unigrams + bigrams.** Portable (NumPy only), deterministic, and
  fast. Bigrams let multi-word concepts ("return policy", "restocking fee")
  match as phrases, recovering recall that pure single-word matching loses.
  Vectors are L2-normalized so cosine similarity is a single matrix–vector dot
  product. For a much larger corpus, swap the dense matrix for `scipy.sparse`
  or an embedding-backed vector store behind the same `retrieve()` interface.
- **Resilient LLM layer.** The Anthropic SDK auto-retries transient errors
  (connection issues, 429, ≥500) with backoff; on top of that, the service
  catches auth errors (surfaced as 503 for operator attention) and transient
  failures (returned as a polite fallback message with `degraded: true`) so the
  API never returns a 500 for a reachable-but-busy upstream.
- **Bounded memory.** Sessions keep only the most recent N messages, and the
  store keeps only the most recent M sessions (LRU), so a long-running process
  cannot grow without limit.
- **Boots even when misconfigured.** A missing knowledge base or API key is
  logged and reported via `/health` rather than crashing; affected endpoints
  return a clear `503`.

## Endpoints

| Method | Path                        | Purpose                              |
|--------|-----------------------------|--------------------------------------|
| POST   | `/api/chat`                 | Ask a question (RAG answer + sources)|
| GET    | `/health`                   | Readiness of KB + LLM configuration  |
| DELETE | `/api/sessions/{id}`        | Forget a conversation's history      |

## Configuration (env / `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required** to answer. |
| `MODEL` | `claude-opus-4-8` | Generation model. |
| `TOP_K` | `3` | Chunks retrieved per query. |
| `MIN_SIMILARITY` | `0.05` | Drop chunks below this cosine score. |
| `CHUNK_WINDOW` / `CHUNK_STRIDE` | `3` / `2` | Sentence window + overlap. |
| `MAX_HISTORY_MESSAGES` | `10` | Messages retained per session. |
| `MAX_SESSIONS` | `1000` | Sessions retained (LRU). |
| `MAX_TOKENS` / `TEMPERATURE` | `1024` / `0.0` | Generation limits. |
| `REQUEST_TIMEOUT` / `MAX_RETRIES` | `30` / `2` | HTTP timeout + SDK retries. |

## Notes / next steps for true production

- Swap the in-memory session store for Redis to support multiple workers.
- For semantic recall beyond lexical matching, add an embeddings index
  (e.g. a local sentence-transformer or a hosted embedding model) and combine
  it with TF-IDF (hybrid retrieval).
- Add request authentication/rate limiting at the edge and structured request
  logging/metrics.
