# RAG Knowledge Assistant

Upload PDF, Markdown, text, AsciiDoc architecture docs, or source files, search the indexed chunks, and ask questions that are answered only from those chunks.

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

`POST /api/retrieve` embeds the question, ranks passages with vector similarity and with BM25, and fuses those two rankings. It then reranks the candidates. The default reranker raises passages that contain more of the question's terms. Set `RERANK_PROVIDER=cross_encoder` to score each question and passage together, preferring terms that sit next to each other in the same order as the question. Set `RERANK_PROVIDER=api` to send those candidates to a Cohere-compatible rerank endpoint (`RERANK_API_URL` and `RERANK_API_KEY`). The response includes `vector_score`, `lexical_score`, and `rerank_score`. `queries` lists each search that was run. `sufficient_evidence` is false when none of those passages cover the question.

`POST /api/answer` sends those passages to the language model. The model must cite them, and each citation has to be supported by the words in that passage. A reply with no valid citation, or a citation that does not match the passage, is replaced by the refusal sentence. When two visible documents state different numbers, `conflict` is true and the answer cites both sides.

Passages are untrusted data. Lines that try to change the assistant's rules, including requests to reveal a secret phrase or skip citations, are removed before ranking and before the model sees the text. The stored chunk keeps the original text, and `instruction_detected` marks those passages.

`POST /api/documents` accepts an optional `visibility` field: `private` (the default) or `shared`. Send `X-User-Id` to choose the owner. A missing header uses `DEFAULT_USER_ID`. Private documents are readable and deletable only by the owner. Shared documents are readable by every caller and deletable only by the owner. Another caller's documents look the same as a missing id (`404`) and are left out of search. The same file bytes can be indexed once for each owner.

`POST /api/documents` returns `201` the first time a file's bytes are indexed for that owner and `200` with `already_indexed: true` when those same bytes were already stored for them. `PUT /api/documents/{id}` stores a new version. The previous indexed version stays searchable until the new one is indexed, and `GET /api/documents/{id}/versions` lists that history. `GET /api/documents/{id}/versions/{version}/file` returns the original bytes. `GET /api/documents/{id}` returns the chunks of the indexed version. `DELETE /api/documents/{id}` removes every version, its chunks, and the stored bytes.

Chunk offsets are positions in the cleaned page text. PDF chunks include a 1-based `page`. Markdown and AsciiDoc chunks include the heading that introduced the section. Source chunks include the top-level function, class, or other symbol, and a nested definition stays inside that symbol. Headings inside fenced code blocks are not treated as section titles. Each chunk is embedded and stored in the vector collection with its filename, content type, and those source locations.

## Operate

`INDEXING_MODE=inline` embeds during the request. `INDEXING_MODE=background` stores the bytes, records a job, and returns `202` with `status: "queued"`. With `WORKER_IN_PROCESS=true` the API process drains that queue. For a separate worker:

```bash
INDEXING_MODE=background WORKER_IN_PROCESS=false uvicorn app.main:app
python -m app.worker
```

The API and the worker share `SQLITE_PATH` and `BLOB_PATH`. A local Chroma directory is safe for one process. Set `CHROMA_HOST` when more than one process reads or writes vectors. A job left in `running` longer than `JOB_TIMEOUT_SECONDS` is queued again.

Repeated question and passage text is cached. Answers are cached until a document is indexed, deleted, or its visibility changes. Unchanged sections of an updated document reuse the embedding cache, so a new version does not re-embed text that did not change. `GET /health` reports queue depth and the catalog generation. `GET /ready` checks that metadata and vectors can be read. `GET /metrics` exposes counters and timings.

Each answer records a trace. The response includes `action` and `trace_id`. `action` is `clarify` when the question does not name a topic, `retry` when a shorter query was required, `answer` when the first search was enough, and `refuse` when the evidence is still missing. `GET /api/traces/{trace_id}` returns those steps.

`POST /api/research` investigates a question in steps. The planner turns one request into independent subquestions, and each subquestion has a completion criterion: a cited claim must be supported by a retrieved passage. `knowledge` searches the indexed collection through `search(query, filters, top_k)`, which returns ranked passages, document ids, source locations, and relevance scores. `catalog` lists visible documents when the question asks which documents, files, or sources exist. `graph` follows a file path to the paths recorded with it. Each search counts against `RESEARCH_SEARCH_BUDGET`. A weak search is retried once when budget remains. A timeout or a malformed retrieval response becomes an evidence gap for that subquestion. Accepted claims are stored with an evidence id, a source reference, a timestamp, a support score, and a link to the supporting passage. A claim is kept when at least 60% of its content words appear in that passage. Claims from different documents that state different numbers are returned in `conflicts`, with both filenames and both values. Disagreements are also listed under contradictions. The report's `action` is `clarify`, `research`, or `refuse`.

`POST /api/claims/judge` retrieves supporting and opposing passages for one claim through the same search interface. The production judge is `LLMClaimJudge`: it asks the chat model for a schema-checked JSON verdict and keeps only evidence ids that were retrieved. A contradiction is kept only when the passages share scope, dates, versions, units, populations, conditions, and qualifiers and then disagree on a quantity or one passage denies the claim. A different date, version, unit, population, condition, or qualifier is a different fact. Instruction lines in a source document are stripped before the model sees the text, and each cited passage keeps its filename and character offsets. `evaluation/datasets/claim_judge.json` is the verification benchmark, and the baseline reports precision and recall separately for verification and for contradiction detection.

Each research report and claim verdict includes an evidence graph. Nodes are claims, passages, documents, versions, and sources. Edges record that a passage supports or contradicts a claim, that a passage was extracted from a document version, and the exact source offsets. That graph is what later report steps use to reason across documents.

`POST /api/operations/recover` queues failed versions again and rebuilds any indexed version whose vectors are missing. Inline mode finishes that work in the request. Background mode leaves it on the worker queue. The previous indexed version stays searchable until the replacement finishes.

One container is enough for a single host. It keeps SQLite, the original bytes, and Chroma on one volume, and indexes in a background thread:

```bash
docker compose up --build
```

The local providers in that file are for an offline deploy. Set the OpenAI variables in the environment for production answers.

## Chunking and embeddings

`CHUNK_STRATEGY` selects how a document is split:

- `fixed` cuts every `CHUNK_SIZE` characters
- `recursive` prefers paragraph, line, and sentence breaks
- `structure` does that inside markdown sections and keeps the heading
- `semantic` groups neighboring sentences while their cosine similarity stays at or above `SEMANTIC_SIMILARITY`

The collection records the embedding model and vector width used to index it. Search refuses to run when the active model differs, because those vectors are not comparable. `GET /api/embeddings` reports both identities. `POST /api/embeddings/migrate` rebuilds the vector collection and re-embeds every indexed version with the active model, including when the vector width changes. `POST /api/embeddings/evaluate` scores whether related passages outrank unrelated ones. The cosine retrieval score is `1 - cosine distance` for normalized vectors.

## Query transformation

These options are off by default, so a question is searched as it was asked.

- `QUERY_REWRITE` removes politeness such as "can you tell me" before the search
- `QUERY_DECOMPOSE` splits a question on "and" when both clauses still have two content words. "connect and read timeouts" stays one query
- `MULTI_QUERY` searches the full question, each clause, and the question's keywords, then fuses those rankings

`queries` in the retrieve response shows the searches that ran. Coverage is still judged against the rewritten question, or the original question when rewriting is off.

## Advanced retrieval

These options are also off by default. Turn one on and index again so the stored chunks match it.

- `PARENT_RETRIEVAL` embeds slices of `CHILD_CHUNK_SIZE` and returns the parent section that contained the matching slice
- `CONTEXTUAL_RETRIEVAL` embeds a prefix naming the file and heading with the chunk. The text stored for citation stays the original chunk
- `KNOWLEDGE_GRAPH` records file paths that share a sentence, such as `alpha/retry.py calls beta/client.py`. A later question that names one path also searches the path it links to

## Tests

```bash
pytest
```

`evaluation/datasets/phase1_retrieval.json` checks retrieval against the sample documents. `evaluation/datasets/phase3_answers.json` checks cited answers and the refusal when the collection does not contain the answer.

`evaluation/datasets/phase4_quality.json` checks relevant questions, irrelevant questions, missing information, conflicting documents, incorrect citations, retrieval failures, and instructions hidden in a document. `pytest` compares the local pipeline with the floors in `evaluation/baselines/phase4_quality.json`.

`evaluation/datasets/rag_metrics.json` scores Recall@K, MRR, nDCG, faithfulness, answer relevance, and citation correctness. Faithfulness is the share of a cited claim that appears in that passage. Answer relevance is the share of question words the passages can support that also appear in the answer. A refusal is faithful, and it is relevant when the question should be refused. Citation correctness requires the marker to name a relevant passage and clear the faithfulness bar.

Indirect prompt injection is stripped before ranking and answering. Private documents are left out of search for other callers. An answer that repeats a canary, API key, password, or secret from a document the caller cannot read is replaced by the refusal. A chunk that repeats one word enough to crowd out ordinary prose is left out of ranking, so a poisoned document cannot win on keyword stuffing.

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
| `CHUNK_STRATEGY` | `structure` | `fixed`, `recursive`, `semantic`, or `structure` |
| `SEMANTIC_SIMILARITY` | `0.35` | Minimum cosine similarity for joining sentences |
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
| `AGENT_ENABLED` | `true` | Clarify, search, or retry before answering |
| `RESEARCH_SEARCH_BUDGET` | `16` | Knowledge searches one research request may run |
| `RESEARCH_TIMEOUT_SECONDS` | `30` | Limit for one research search |
| `RETRIEVAL_CANDIDATES` | `20` | Vector and lexical candidates before fusion |
| `RRF_K` | `60` | Reciprocal-rank fusion constant |
| `RERANK_COVERAGE_WEIGHT` | `0.35` | Weight of term coverage in the default score |
| `RERANK_PROVIDER` | `coverage` | `coverage`, `cross_encoder`, or `api` |
| `RERANK_MODEL` | empty | Cross-encoder name, or the API model (`rerank-v3.5` when empty) |
| `RERANK_API_URL` | empty | Cohere-compatible rerank endpoint |
| `RERANK_API_KEY` | empty | Bearer token for that endpoint |
| `QUERY_REWRITE` | `false` | Strip politeness before searching |
| `QUERY_DECOMPOSE` | `false` | Split a question into clauses |
| `MULTI_QUERY` | `false` | Search each clause and fuse the rankings |
| `PARENT_RETRIEVAL` | `false` | Search child slices and return the parent |
| `CHILD_CHUNK_SIZE` | `240` | Child slice size when parent retrieval is on |
| `CONTEXTUAL_RETRIEVAL` | `false` | Embed the file and heading with each chunk |
| `KNOWLEDGE_GRAPH` | `false` | Expand a query along linked file paths |
| `LOG_LEVEL` | `INFO` | Log level |

Retrieval logs include `hit_count`, `top_score`, `sufficient_evidence`, and `duration_ms`. Answer logs include `citation_count`, `sufficient_evidence`, and `duration_ms`.
