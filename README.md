# RAG Knowledge Assistant

Upload PDF, Markdown, or text documents, search the indexed chunks, and ask questions that are answered only from those chunks.

An answer includes citation markers such as `[1]` and the source file, heading, and page. If the retrieved passages do not cover the question, the API returns `I don't have enough evidence to answer.`

## Layout

- `app/api` upload, update, list, fetch, delete, retrieve, and answer
- `app/ingestion` parsing, cleaning, chunking, embeddings, versions, and the index queue
- `app/retrieval` hybrid vector and lexical search, reranking, and evidence inspection
- `app/generation` grounded answers, citation checks, and an answer cache
- `app/security` upload validation, document visibility, and stripping of instructions inside passages
- `app/observability` JSON logs and a `/metrics` endpoint
- `app/storage` content-addressed bytes for each document version
- `evaluation` retrieval, answer, and quality cases, plus the metric baseline
- `samples` example knowledge-base documents

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` to use OpenAI embeddings and `gpt-4o-mini` for answers.

For an offline demo, set both providers to local:

```bash
EMBEDDING_PROVIDER=local
LLM_PROVIDER=local
```

The local embedder is a hashed bag-of-words, and the local answerer quotes the best supporting sentence. Neither is a production model. They let the pipeline run without an API key.

## Run

```bash
uvicorn app.main:app --reload
```

Index a document, inspect the retrieved passages, then ask a question:

```bash
curl -F "file=@samples/api-errors.md" http://127.0.0.1:8000/api/documents

curl -X POST http://127.0.0.1:8000/api/retrieve \
  -H "content-type: application/json" \
  -d '{"query":"which file implements the retry logic","top_k":3}'

curl -X POST http://127.0.0.1:8000/api/answer \
  -H "content-type: application/json" \
  -d '{"question":"which file implements the retry logic","top_k":3}'
```

`POST /api/retrieve` embeds the question, fuses vector search with BM25, reranks by how many question terms each passage contains, and returns ranked passages with source metadata. `sufficient_evidence` is false when none of those passages cover the question.

`POST /api/answer` sends those passages to the language model. The model must cite them, and each citation has to be supported by the words in that passage. A reply with no valid citation, or a citation that does not match the passage, is replaced by the refusal sentence. When two visible documents state different numbers, `conflict` is true and the answer cites both sides.

Passages are untrusted data. Lines that try to change the assistant's rules, including requests to reveal a secret phrase or skip citations, are removed before ranking and before the model sees the text. The stored chunk keeps the original text, and `instruction_detected` marks those passages.

`POST /api/documents` accepts an optional `visibility` field: `private` (the default) or `shared`. Send `X-User-Id` to choose the owner. A missing header uses `DEFAULT_USER_ID`. Private documents are readable and deletable only by the owner. Shared documents are readable by every caller and deletable only by the owner. Another caller's documents look the same as a missing id (`404`) and are left out of search. The same file bytes can be indexed once for each owner.

`POST /api/documents` returns `201` the first time a file's bytes are indexed for that owner and `200` with `already_indexed: true` when those same bytes were already stored for them. `PUT /api/documents/{id}` stores a new version. The previous indexed version stays searchable until the new one is indexed, and `GET /api/documents/{id}/versions` lists that history. `GET /api/documents/{id}/versions/{version}/file` returns the original bytes. `GET /api/documents/{id}` returns the chunks of the indexed version. `DELETE /api/documents/{id}` removes every version, its chunks, and the stored bytes.

Chunk offsets are positions in the cleaned page text. PDF chunks include a 1-based `page`. Markdown chunks include the heading that introduced the section. Headings inside fenced code blocks are not treated as section titles.

## Operate

`INDEXING_MODE=inline` embeds during the request. `INDEXING_MODE=background` stores the bytes, records a job, and returns `202` with `status: "queued"`. With `WORKER_IN_PROCESS=true` the API process drains that queue. For a separate worker:

```bash
INDEXING_MODE=background WORKER_IN_PROCESS=false uvicorn app.main:app
python -m app.worker
```

The API and the worker share `SQLITE_PATH` and `BLOB_PATH`. A local Chroma directory is safe for one process. Set `CHROMA_HOST` when more than one process reads or writes vectors. A job left in `running` longer than `JOB_TIMEOUT_SECONDS` is queued again.

Repeated question and passage text is cached. Answers are cached until a document is indexed, deleted, or its visibility changes. `GET /health` reports queue depth and the catalog generation. `GET /ready` checks that metadata and vectors can be read. `GET /metrics` exposes counters and timings.

One container is enough for a single host. It keeps SQLite, the original bytes, and Chroma on one volume, and indexes in a background thread:

```bash
docker compose up --build
```

The local providers in that file are for an offline deploy. Set the OpenAI variables in the environment for production answers.

## Tests

```bash
pytest
```

`evaluation/datasets/phase1_retrieval.json` checks retrieval against the sample documents. `evaluation/datasets/phase3_answers.json` checks cited answers and the refusal when the collection does not contain the answer.

`evaluation/datasets/phase4_quality.json` checks relevant questions, irrelevant questions, missing information, conflicting documents, incorrect citations, retrieval failures, and instructions hidden in a document. `pytest` compares the local pipeline with the floors in `evaluation/baselines/phase4_quality.json`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `LLM_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | empty | Required for the OpenAI providers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHROMA_PATH` | `data/chroma` | Local vector collection |
| `SQLITE_PATH` | `data/metadata.db` | Document metadata |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between neighboring chunks |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload limit |
| `DEFAULT_USER_ID` | `local` | Caller used when `X-User-Id` is absent |
| `BLOB_PATH` | `data/blobs` | Original bytes, stored by checksum |
| `CHROMA_HOST` | empty | Chroma server host; empty uses `CHROMA_PATH` |
| `CHROMA_PORT` | `8000` | Chroma server port |
| `INDEXING_MODE` | `inline` | `inline` or `background` |
| `WORKER_IN_PROCESS` | `true` | Drain the queue inside the API process |
| `WORKER_POLL_SECONDS` | `1` | Delay between empty queue checks |
| `JOB_TIMEOUT_SECONDS` | `900` | Requeue a job left in `running` |
| `EMBEDDING_CACHE_SIZE` | `4096` | Cached passage and query embeddings |
| `ANSWER_CACHE_SIZE` | `256` | Cached answers for an unchanged catalog |
| `RETRIEVAL_CANDIDATES` | `20` | Vector and lexical candidates before fusion |
| `LOG_LEVEL` | `INFO` | Log level |

Retrieval logs include `hit_count`, `top_score`, `sufficient_evidence`, and `duration_ms`. Answer logs include `citation_count`, `sufficient_evidence`, and `duration_ms`.
