# RAG Knowledge Assistant

Phase 1 indexes your documents into a searchable knowledge collection. Upload a PDF, Markdown, or text file. The app extracts the text, splits it into chunks, stores metadata in SQLite, and stores embeddings in a local Chroma collection. Search returns the matching chunks and the file they came from.

Later phases can add answer generation, citations, and a Streamlit UI on top of this collection.

## Layout

- `app/api` upload, list, fetch, delete, and search
- `app/ingestion` parsing, cleaning, chunking, embeddings, and indexing
- `app/retrieval` vector search
- `app/security` upload validation
- `app/observability` JSON logs with latency and retrieval metrics
- `evaluation` retrieval cases that the tests run against the sample documents
- `samples` example knowledge-base documents

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to use OpenAI embeddings (`text-embedding-3-small` by default).

For an offline demo, set `EMBEDDING_PROVIDER=local`. That provider is a deterministic hashed bag-of-words so the pipeline can be tested without an API key. It is not a semantic embedding model.

## Run

```bash
uvicorn app.main:app --reload
```

Upload a document and search it:

```bash
curl -F "file=@samples/api-errors.md" http://127.0.0.1:8000/api/documents

curl -X POST http://127.0.0.1:8000/api/search \
  -H "content-type: application/json" \
  -d '{"query":"which file implements the retry logic","top_k":3}'
```

`POST /api/documents` returns `201` the first time a file's bytes are indexed and `200` with `already_indexed: true` when those same bytes were already stored. `GET /api/documents/{id}` returns the stored chunks. `DELETE /api/documents/{id}` removes the document and its chunks.

Chunk offsets are positions in the cleaned page text. PDF chunks include a 1-based `page`. Markdown chunks include the heading that introduced the section. Headings inside fenced code blocks are not treated as section titles.

## Tests

```bash
pytest
```

The retrieval dataset in `evaluation/datasets/phase1_retrieval.json` checks that questions about the sample documents hit the expected source text.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | empty | Required for the OpenAI provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_PATH` | `data/chroma` | Local vector collection |
| `SQLITE_PATH` | `data/metadata.db` | Document metadata |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between neighboring chunks |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload limit |
| `LOG_LEVEL` | `INFO` | Log level |

Indexed logs include `duration_ms` and `chunk_count`. Search logs include `hit_count`, `top_score`, and `duration_ms`.
