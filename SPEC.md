# Multilingual RAG Customer Support Chatbot - Project Specification

English | [한국어](SPEC.ko.md)

**Date:** 2026-07-20
**Status:** M0–M8 technical implementation complete; **M8-B staging BGE validation PASS**; production cutover **planning may begin**; production deployment is **not** approved or complete; **M9** production packaging is **not started**

**Language Support:** Korean, English, Japanese, Chinese (extensible)

**Living status:** [MILESTONES.md](MILESTONES.md) · **Ops validation:** [docs/staging-cutover-bge.md](docs/staging-cutover-bge.md)

## 1. Project Overview

**Target:** a production-ready RAG-based customer support chatbot that can be
embedded into an existing website, with high-quality multilingual answers, low
latency, and strong observability.

**Current state (2026-07-20):** the M8-B technical path is complete and **M8-B
staging BGE validation has passed**. Production packaging (**M9**) and
**production deployment approval** remain future work. Detailed milestone status
lives in `MILESTONES.md`; operational procedures and validation evidence live in
`docs/staging-cutover-bge.md`.

Key objectives:
- Excellent performance across Korean, English, Japanese, and Chinese
- Hybrid retrieval + reranking + configurable contextual enrichment
- Adaptive/agentic routing using LangGraph
- Real-time SSE streaming
- Easy integration via lightweight embed widget
- Automated evaluation with RAGAS
- Optional observability with Langfuse when credentials and configuration are
  present
- Docker-based packaging and deployment suitable for production (target; see M9)

## 2. Fixed Technology Stack

- **Backend**: FastAPI (async) + LangGraph
- **Retrieval**: Custom hybrid pipeline over **Qdrant** (dense + sparse); not
  LlamaIndex-as-sole-retriever
- **Indexing / helpers**: LlamaIndex core and related helpers may support
  ingestion/indexing utilities where used in the codebase
- **Embedding**: **BGE-M3** dense when `PREFER_BGE=true` (optional `bge`
  dependency group); **hash dense** offline/rollback when `PREFER_BGE=false`
- **Reranker**: **BGE-reranker-v2-m3** (CrossEncoder) in BGE mode; **lexical**
  reranker in hash mode or when BGE reranker is unavailable
- **Ingestion enhancement**: Contextual Retrieval for generic documents when
  enabled; FAQ atomic path (see §3.2)
- **Evaluation**: RAGAS-style runner and multilingual datasets
- **Observability**: **Optional Langfuse** integration (no-op without keys)
- **Widget**: Lightweight JavaScript embed widget with SSE streaming
- **Dependency Management**: uv + `pyproject.toml`
- **Python Version**: 3.12

## 3. Core Technical Requirements

### 3.1 Multilingual & CJK Handling

- First-class support for Korean, English, Japanese, and Chinese.
- Query language detection (auto or explicit parameter).
- Answers should be returned in the query language by default.
- **CJK Chunking Strategy**: Must implement proper language-aware or semantic
  chunking. Do not rely on default English-centric splitters. Justify the chosen
  approach (e.g. semantic breakpoint chunking, language-aware tokenizers).

### 3.2 Contextual Retrieval and FAQ Ingestion

**Generic documents**

- When contextualization is enabled, use an LLM to generate a concise contextual
  description for each chunk (including document title and surrounding context
  where available).
- Contextual text participates in embedding / index preparation.

**FAQ documents**

- Use **atomic Q+A chunks** so that question and answer remain co-located for
  retrieval.
- Preserve FAQ metadata (including `faq_id`, language, and question text where
  present).
- FAQ contextualization is **disabled by default** and requires explicit
  enablement; do not require LLM contextual generation for every FAQ chunk.

### 3.3 Retrieval Pipeline

Staged pipeline (defaults are configurable, not fixed protocol constants):

1. **Retrieve** a broad candidate set from Qdrant hybrid search (example default:
   top_n ≈ 40).
2. **Rerank** to an internal top-k (example default: top_k ≈ 6).
3. **Score-prune** so generation and client-facing sources receive **1–3**
   high-signal chunks: **Top-1 is always retained**; Top-2/Top-3 only when
   absolute/relative score thresholds pass.

Mode behavior:

- **BGE mode** (`PREFER_BGE=true`): BGE-M3 dense + sparse hybrid; BGE CrossEncoder
  reranking. Active collection: **`onlybook_faq_bge_m3_v1`** (dense dim **1024**).
- **Hash rollback / offline** (`PREFER_BGE=false`): hash dense + sparse hybrid;
  lexical reranking. Rollback collection: **`onlybook_faq`** (dense dim **384**).
- **`PREFER_BGE` and `QDRANT_COLLECTION` must remain a valid paired
  configuration** (never mix BGE collection with hash mode or the reverse).

Additional requirements:

- Support metadata filtering (language, category, source, etc.), including
  retrieval language filtering when enabled.
- Client-facing source identifiers must be language-unique public IDs (e.g.
  `faq-{lang}-Q<number>`) and must **not** expose internal filesystem paths;
  titles should prefer natural FAQ questions where available.

### 3.4 Agentic / Adaptive Routing (LangGraph)

- Build a LangGraph workflow with the following capabilities:
  - Query analysis (intent, complexity, language, ambiguity)
  - Conditional routing (simple retrieval, multi-hop retrieval, clarification
    needed, out-of-scope, etc.)
  - Support for conversation history
  - Structured output using Pydantic models

### 3.5 Streaming API

- FastAPI endpoint that streams responses using Server-Sent Events (SSE).
- Stream tokens progressively along with metadata (sources, scores, language
  info).
- Source field rules in §3.3 apply to SSE payloads.

### 3.5.1 Health, Readiness, and Model Lifecycle

- **`/health`**: liveness (process up).
- **`/ready`**: lightweight readiness for deployment probes.
- Readiness validates configuration pairing (embedder mode vs collection),
  collection availability, expected dense vector dimension, and the **startup
  embedder snapshot**.
- Readiness probes **must not** reconstruct or reload the dense embedder.
- Equivalent dense-embedder configuration **must reuse one process-level cached
  instance**.
- The BGE reranker **may lazy-load** on first BGE retrieval; an **operational
  warm-up** may be performed after deployment.
- **`/ready` does not automatically load or validate the reranker**.
- Model-cache persistence is an operational deployment concern (host/volume
  layout), not a request-path responsibility.

### 3.6 Embed Widget

- Lightweight, self-contained JavaScript widget.
- Support floating button and inline modes.
- Real-time SSE streaming UI.
- Markdown rendering, source citations, language handling.
- Easy to integrate into existing websites via a single script tag.

### 3.7 Evaluation & Quality Gates

- Implement automated RAGAS-style evaluation.
- Maintain cross-lingual test sets (parallel queries across four languages).
- Faithfulness remains a useful quality signal; **answer relevancy is diagnostic
  only**, not a hard release gate when multilingual metric noise is high.
- Every major milestone must include runnable Gate Verification commands and PASS
  criteria.

**Staging validation gates** (procedures and evidence in
`docs/staging-cutover-bge.md`):

- Fixed EN / KO / JA / ZH smoke (message + session_id; auto language detection).
- **BGE → Hash → BGE** rollback drill.
- Representative soak under agreed latency, memory, model-lifecycle, and quality
  gates.

**Release posture:**

- **M8-B staging validation PASS** permits **production cutover planning** to
  begin.
- **Production deployment approval** remains a separate decision and is not
  granted by staging validation alone.

## 4. Development & Workflow Rules

- Work exclusively in **Plan Mode** until the human approves execution.
- Organize work into clear milestones (M0, M1, M2, ...).
- Every milestone must end with explicit **Gate Verification** commands and PASS
  criteria.
- Use `uv` for dependency management (edit `pyproject.toml` and run `uv sync` /
  `uv add`).
- Maintain clean diffs after plan approval.
- Include type hints, proper error handling, logging, and tests from the start.
- Make the system easily extensible for adding new languages in the future.

## 5. Target Deliverables and Current Implementation Status

Statuses:

- **COMPLETE** — repository implementation and applicable validation complete
  (does **not** mean production deployment approval)
- **PARTIAL** — implemented in part or not fully production-hardened
- **PLANNED** — not yet implemented

| Deliverable | Status |
|-------------|--------|
| FastAPI + LangGraph backend | COMPLETE |
| Multilingual ingestion, FAQ atomic chunking, configurable contextualization | COMPLETE |
| Hybrid retrieval, reranking, pruning, and safe public sources | COMPLETE |
| Lightweight embeddable chat widget | COMPLETE |
| RAGAS evaluation scripts with cross-lingual test sets | COMPLETE |
| Optional Langfuse integration | PARTIAL |
| Staging BGE Docker/image path and staging validation | COMPLETE (staging) |
| Production packaging and production-ready deployment topology | PLANNED (M9) |
| Production cutover approval / deployment | PLANNED |
| Comprehensive website integration documentation | PARTIAL |

---

**Terminology:** M8-B technical implementation · M8-B staging validation ·
production cutover planning · production deployment approval · M9 production
packaging · BGE collection (`onlybook_faq_bge_m3_v1`) · hash rollback collection
(`onlybook_faq`) · paired `PREFER_BGE` / `QDRANT_COLLECTION`.
