# Multilingual RAG Customer Support Chatbot - Project Specification

**Date:** 2026-07-16  
**Status:** Implementation advanced (M0–M8 technical); staging BGE cutover in progress  
**Language Support:** Korean, English, Japanese, Chinese (extensible)

## 1. Project Overview
Build a production-ready RAG-based customer support chatbot that can be embedded into an existing website. The system must deliver high-quality, accurate answers in multiple languages with low latency and strong observability.

Key objectives:
- Excellent performance across Korean, English, Japanese, and Chinese
- Hybrid retrieval + reranking + contextual enrichment
- Adaptive/agentic routing using LangGraph
- Real-time SSE streaming
- Easy integration via lightweight embed widget
- Automated evaluation with RAGAS and observability with Langfuse
- Docker-based deployment ready for production

## 2. Fixed Technology Stack
- **Backend**: FastAPI (async) + LangGraph 1.0
- **Retrieval & Indexing**: LlamaIndex + Qdrant (hybrid search)
- **Embedding Model**: BGE-M3 (multilingual)
- **Reranker**: BGE-reranker-v2-m3
- **Ingestion Enhancement**: Contextual Retrieval technique
- **Evaluation**: RAGAS
- **Observability**: Langfuse (preferred) or LangSmith
- **Widget**: Lightweight JavaScript embed widget with SSE streaming
- **Dependency Management**: uv + pyproject.toml
- **Python Version**: 3.12

## 3. Core Technical Requirements

### 3.1 Multilingual & CJK Handling
- First-class support for Korean, English, Japanese, and Chinese.
- Query language detection (auto or explicit parameter).
- Answers should be returned in the query language by default.
- **CJK Chunking Strategy**: Must implement proper language-aware or semantic chunking. Do not rely on default English-centric splitters. Justify the chosen approach (recommended options: LlamaIndex `SemanticSplitterNodeParser` with BGE-M3, kiwipiepy for Korean, SudachiPy for Japanese, etc.).

### 3.2 Contextual Retrieval (Ingestion)
- During document ingestion, use an LLM to generate a concise contextual description for each chunk (including document title and surrounding context).
- The contextual text must be included when creating embeddings to improve retrieval quality.

### 3.3 Retrieval Pipeline
- Qdrant hybrid search (dense BGE-M3 + sparse/BM25); hash dense path for offline/rollback.
- Apply BGE-reranker-v2-m3 after initial retrieval (e.g., retrieve top 20–50 → rerank to top 5–8), with lexical fallback when BGE reranker is unavailable.
- After rerank, apply score-based pruning so generation and client sources receive a small set of high-signal chunks (Top-1 always; limited Top-2/3 by absolute/relative thresholds).
- Support metadata filtering (language, category, source, etc.).
- FAQ documents use atomic Q+A chunking so question and answer stay co-located for retrieval.

### 3.4 Agentic / Adaptive Routing (LangGraph)
- Build a LangGraph workflow with the following capabilities:
  - Query analysis (intent, complexity, language, ambiguity)
  - Conditional routing (simple retrieval, multi-hop retrieval, clarification needed, out-of-scope, etc.)
  - Support for conversation history
  - Structured output using Pydantic models

### 3.5 Streaming API
- FastAPI endpoint that streams responses using Server-Sent Events (SSE).
- Stream tokens progressively along with metadata (sources, scores, language info).
- Client-facing source identifiers must be language-unique and must not expose internal filesystem paths; user-facing titles should prefer natural FAQ questions where available.

### 3.5.1 Health and readiness
- Provide a lightweight liveness endpoint and a readiness endpoint suitable for deployment probes.
- Readiness should confirm configuration consistency (embedder mode vs collection), collection availability, and expected dense vector dimension without unnecessary model reloads on every probe.

### 3.6 Embed Widget
- Lightweight, self-contained JavaScript widget.
- Support floating button and inline modes.
- Real-time SSE streaming UI.
- Markdown rendering, source citations, language handling.
- Easy to integrate into existing websites via a single script tag.

### 3.7 Evaluation & Quality Gates
- Implement automated RAGAS evaluation.
- Create or generate cross-lingual test sets (parallel queries across 4 languages).
- Faithfulness remains a primary quality signal; answer relevancy may be retained as a **diagnostic** metric rather than a hard gate when multilingual metric noise is high.
- Every major milestone must include runnable Gate Verification commands.

## 4. Development & Workflow Rules
- Work exclusively in **Plan Mode** until the human approves execution.
- Organize work into clear milestones (M0, M1, M2, ...).
- Every milestone must end with explicit **Gate Verification** commands and PASS criteria.
- Use `uv` for dependency management (edit `pyproject.toml` and run `uv sync` / `uv add`).
- Maintain clean diffs after plan approval.
- Include type hints, proper error handling, logging, and tests from the start.
- Make the system easily extensible for adding new languages in the future.

## 5. Expected Deliverables
- Well-structured FastAPI backend with LangGraph orchestration
- Robust ingestion pipeline with Contextual Retrieval and CJK support
- Production-ready Docker Compose setup
- Lightweight embeddable chat widget
- RAGAS evaluation scripts with cross-lingual test sets
- Langfuse integration for tracing and monitoring
- Comprehensive documentation and integration guide for existing websites

