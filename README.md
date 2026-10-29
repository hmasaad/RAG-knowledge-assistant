# RAG Knowledge Assistant

Upload files, search the indexed chunks, and get answers only from those chunks. Answers include citations (`[1]`, filename, heading, page). If the collection does not cover the question, the reply is `I don't have enough evidence to answer.`

Open `http://127.0.0.1:8000/` after you start the API. Use **Library** to index, **Ask** for one question, **Research** for a multi-step report, **Engineering** for run status.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

`.env.example` uses local embeddings and a local answerer (no API key). For production answers, set `OPENAI_API_KEY` and switch both providers to `openai`.

Open the app at `http://127.0.0.1:8000/`, not as a `file://` page. Keep **User id** as `local` unless a step says otherwise. PDFs need selectable text.

```bash
pytest
```

## Walkthrough

Index these as **Shared**: `samples/api-errors.md`, `samples/database.md`, `evaluation/fixtures/retry-limit-three.md`, `evaluation/fixtures/retry-limit-five.md`. Wait until each is indexed.

**Ask**

| Question | Expect |
|---|---|
| `Which file implements the retry logic?` | Cites `api-errors.md`, names `app/http/retry.py` |
| `Where are connect and read timeouts configured?` | Names `app/http/client.py` |
| `What is this document about?` | Quotes a retrieved sentence |
| `What is the office wifi password?` | Refuses |
| `help` | Asks which document or topic to search |

Use words from the file for a specific fact. The same **User id** as the upload, or search looks empty.

**Research** — mode **Cycle**, halt unchecked unless noted.

| Question | Expect |
|---|---|
| `Which file implements the retry logic and where are timeouts configured?` | Two searches, then reverify and synthesize; cites both files |
| Same question, **Stop after the first committed step** | `halted` after one search. Resume finishes without repeating it |
| `What is the retry limit?` | Keeps both 3 and 5 |
| `Which file implements the retry logic and what is the office wifi password?` | Retry cited; wifi refused |
| `Which documents or files do I have?` | Lists your uploads |

Refresh **Sessions** to reopen a report. Optional: stop uvicorn after a halt, start it again, resume — the plan stays on disk. Switch **User id** to `ada`: private files owned by `local` are hidden; shared files are not.

**Engineering** should list those runs. Halted or failed is not marked completed.

## API

```bash
curl -F "file=@samples/api-errors.md" http://127.0.0.1:8000/api/documents
curl -X POST http://127.0.0.1:8000/api/retrieve -H "content-type: application/json" \
  -d '{"query":"which file implements the retry logic","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/answer -H "content-type: application/json" \
  -d '{"question":"which file implements the retry logic","top_k":3}'
```

| Method | Path | Role |
|---|---|---|
| `POST` | `/api/documents` | Upload (`visibility`: `private` or `shared`) |
| `POST` | `/api/retrieve` | Ranked passages |
| `POST` | `/api/answer` | Cited answer or refuse |
| `POST` | `/api/research/cycle` | Full research session |
| `POST` | `/api/research` | Planner only |
| `GET` / `POST` | `/api/research/sessions/{id}` / `.../resume` | Load or continue |
| `GET` | `/health` `/ready` `/metrics` | Liveness, readiness, counters |

Send `X-User-Id` (default `local`). Private docs are owner-only; shared docs are readable by everyone.

## How it works

1. **Index** — parse, chunk, embed, store vectors and original bytes.
2. **Ask** — retrieve, then answer only from passages that cover the question.
3. **Research** — split the question, search clauses in parallel, verify claims, fill gaps, keep contradictions, synthesize a report. Sessions survive a restart.
4. **Guardrails** — no citation or a bad citation becomes a refuse. Prompt-injection lines are stripped. Secrets from another user's docs are not returned.

```
Library → chunks + vectors
              ↓
Ask: retrieve → cite or refuse
Research: plan → search / verify / fill → synthesize
```

Background indexing: `INDEXING_MODE=background` and `python -m app.worker` (share `SQLITE_PATH` and `BLOB_PATH`). One host: `docker compose up --build`.

## Configuration

Copy `.env.example` and change only what you need. Defaults below match the code.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` / `LLM_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | empty | Required for OpenAI |
| `CHROMA_PATH` / `SQLITE_PATH` / `BLOB_PATH` | `data/…` | Vectors, metadata, original bytes |
| `INDEXING_MODE` | `inline` | `inline` or `background` |
| `DEFAULT_USER_ID` | `local` | Caller when `X-User-Id` is missing |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_STRATEGY` | `800` / `120` / `structure` | How files are split |
| `RESEARCH_SEARCH_BUDGET` | `16` | Knowledge searches per request |
| `RESEARCH_TASK_CONCURRENCY` | `2` | Parallel DAG tasks in one cycle |
| `QUERY_REWRITE` / `QUERY_DECOMPOSE` / `MULTI_QUERY` | `false` | Optional query transforms |
| `PARENT_RETRIEVAL` / `CONTEXTUAL_RETRIEVAL` / `KNOWLEDGE_GRAPH` | `false` | Optional retrieval; re-index after turning on |

All settings live in `.env.example`.
