# Multilingual RAG Customer Support Chatbot - Project Specification

> **한국어 번역본**
>
> 기준 영문 문서: [SPEC.md](SPEC.md)
> 번역 기준 리포지토리 커밋:
> `d0ef3f025f7f6a940c224f5de5759bdaed2f1def` (M9 packaging baseline; follow-up docs sync)
> 동기화 날짜: 2026-07-21
>
> 영문 문서와 내용이 충돌할 경우 영문 문서를 우선합니다.

[English](SPEC.md) | 한국어

**Date:** 2026-07-21
**Status:** **M0–M9 리포지토리 구현 완료**; **M8-B staging BGE validation PASS**; **M9 리포지토리 프로덕션 패키징 완료** 및 로컬 검증됨; **실제 프로덕션 배포는 수행되지 않음**; **프로덕션 트래픽 컷오버는 승인되지 않음**; 플랫폼 의사결정 필요

**Language Support:** Korean, English, Japanese, Chinese (extensible)

**Living status:** [MILESTONES.md](MILESTONES.md) · **Ops validation:** [docs/staging-cutover-bge.md](docs/staging-cutover-bge.md) · **Production packaging:** [docs/production-deployment.md](docs/production-deployment.md)

## 1. Project Overview

**Target:** 기존 웹사이트에 임베드할 수 있는 production-ready RAG 기반 고객 지원 챗봇으로, 고품질 다국어 응답, 낮은 지연, 강한 관측성을 목표로 합니다.

**Current state (2026-07-21):** M0–M9 리포지토리 작업이 완료되었습니다. **M8-B staging BGE validation PASS**와 **M9 프로덕션 패키징**(템플릿, Compose override, 시작 정책, 런북, 로컬 검증)을 포함합니다. **실제 프로덕션 배포는 수행되지 않았고** **프로덕션 트래픽 컷오버는 승인되지 않았습니다**. TLS/ingress, 시크릿, 레지스트리, HA, 공식 트래픽 전환 등 플랫폼 의사결정이 여전히 필요합니다. 상세 마일스톤 상태는 `MILESTONES.md`를 참고하세요.

주요 목표:
- 한국어, 영어, 일본어, 중국어에서 우수한 성능
- Hybrid retrieval + reranking + 구성 가능한 contextual enrichment
- LangGraph 기반 adaptive/agentic routing
- 실시간 SSE 스트리밍
- 경량 임베드 위젯으로 쉬운 통합
- RAGAS를 통한 자동 평가
- 자격 증명과 구성이 있을 때 Langfuse를 통한 선택적 관측성
- 프로덕션에 적합한 Docker 기반 패키징/배포 (target; M9 참고)

## 2. Fixed Technology Stack

- **Backend**: FastAPI (async) + LangGraph
- **Retrieval**: **Qdrant** 위 custom hybrid pipeline (dense + hashed log-TF sparse, RRF 융합; 고전 BM25 아님); LlamaIndex를 유일한 검색 엔진으로 보지 않음
- **Indexing / helpers**: LlamaIndex core 및 관련 헬퍼가 코드베이스에서 ingestion/indexing 유틸을 지원할 수 있음
- **Embedding**: `PREFER_BGE=true` 일 때 **BGE-M3** dense (optional `bge` dependency group); `PREFER_BGE=false` 일 때 **hash dense** offline/rollback
- **Reranker**: BGE mode에서 **BGE-reranker-v2-m3** (CrossEncoder); hash mode 또는 BGE reranker 불가 시 **lexical** reranker
- **Ingestion enhancement**: 활성화 시 generic 문서에 대한 Contextual Retrieval; FAQ atomic path (§3.2)
- **Evaluation**: RAGAS-style runner 및 multilingual datasets
- **Observability**: **Optional Langfuse** integration (no-op without keys)
- **Widget**: SSE 스트리밍이 있는 경량 JavaScript 임베드 위젯
- **Dependency Management**: uv + `pyproject.toml`
- **Python Version**: 3.12

## 3. Core Technical Requirements

### 3.1 Multilingual & CJK Handling

- 한국어, 영어, 일본어, 중국어 1급 지원.
- 쿼리 언어 감지 (자동 또는 명시 파라미터).
- 기본적으로 쿼리 언어로 답변.
- **CJK Chunking Strategy**: 언어 인지 또는 시맨틱 청킹을 구현해야 함. 기본 영어 중심 splitter에만 의존하지 말 것. 선택한 접근을 정당화할 것 (예: semantic breakpoint chunking, language-aware tokenizers).

### 3.2 Contextual Retrieval and FAQ Ingestion

**Generic documents**

- Contextualization이 켜져 있으면 LLM으로 각 청크에 대한 간결한 contextual description 생성 (문서 제목 및 가능한 경우 주변 컨텍스트 포함).
- Contextual text는 embedding / index preparation에 참여.

**FAQ documents**

- 질문과 답이 검색을 위해 함께 있도록 **atomic Q+A chunks** 사용.
- FAQ metadata 보존 (`faq_id`, language, question text 등이 있을 때 포함).
- FAQ contextualization은 **기본적으로 비활성**이며 명시적 enablement가 필요; 모든 FAQ chunk에 대해 LLM contextual generation을 요구하지 않음.

### 3.3 Retrieval Pipeline

단계적 파이프라인 (defaults는 configurable이며 고정 프로토콜 상수가 아님):

1. **Retrieve**: Qdrant hybrid search에서 넓은 후보 집합 (example default: top_n ≈ 40).
2. **Rerank**: internal top-k로 축소 (example default: top_k ≈ 6).
3. **Score-prune**: generation 및 client-facing sources가 **1–3**개의 high-signal chunk를 받도록 함: **Top-1은 항상 유지**; Top-2/Top-3는 absolute/relative score thresholds를 통과할 때만.

Mode behavior:

- **BGE mode** (`PREFER_BGE=true`): BGE-M3 dense + sparse hybrid; BGE CrossEncoder reranking. Active collection: **`onlybook_faq_bge_m3_v1`** (dense dim **1024**).
- **Hash rollback / offline** (`PREFER_BGE=false`): hash dense + sparse hybrid; lexical reranking. Rollback collection: **`onlybook_faq`** (dense dim **384**).
- **`PREFER_BGE`와 `QDRANT_COLLECTION`은 유효한 paired configuration을 유지해야 함** (BGE collection과 hash mode를 섞거나 그 반대를 하지 말 것).

Additional requirements:

- language, category, source 등 metadata filtering 지원 (활성화 시 retrieval language filtering 포함).
- Client-facing source identifiers는 언어-고유 public ID (예: `faq-{lang}-Q<number>`)여야 하며 내부 filesystem path를 **노출하지 않아야** 함; titles는 가능하면 자연스러운 FAQ 질문을 선호.

### 3.4 Agentic / Adaptive Routing (LangGraph)

- 다음 능력을 갖춘 LangGraph 워크플로:
  - Query analysis (intent, complexity, language, ambiguity)
  - Conditional routing (simple retrieval, multi-hop retrieval, clarification needed, out-of-scope 등)
  - Support for conversation history
  - Structured output using Pydantic models

### 3.5 Streaming API

- Server-Sent Events (SSE)로 응답을 스트리밍하는 FastAPI 엔드포인트.
- sources, scores, language info 등 메타데이터와 함께 토큰을 점진적으로 스트림.
- SSE payload에는 §3.3 source field 규칙이 적용됨.

### 3.5.1 Health, Readiness, and Model Lifecycle

- **`/health`**: liveness (process up).
- **`/ready`**: deployment probes용 lightweight readiness.
- Readiness는 configuration pairing (embedder mode vs collection), collection availability, expected dense vector dimension, 그리고 **startup embedder snapshot**을 검증.
- Readiness probes는 dense embedder를 reconstruct 하거나 reload **해서는 안 됨**.
- Equivalent dense-embedder configuration은 **process-level cached instance 하나를 재사용**해야 함.
- BGE reranker는 first BGE retrieval 시 **lazy-load 할 수 있음**; deployment 후 **operational warm-up**을 수행할 수 있음.
- **`/ready`는 reranker를 자동으로 load 하거나 validate 하지 않음**.
- Model-cache persistence는 operational deployment concern (host/volume layout)이며 request-path 책임이 아님.

### 3.6 Embed Widget

- Lightweight, self-contained JavaScript widget.
- floating button 및 inline modes 지원.
- Real-time SSE streaming UI.
- Markdown rendering, source citations, language handling.
- 단일 script tag로 기존 사이트에 쉽게 통합.

### 3.7 Evaluation & Quality Gates

- Automated RAGAS-style evaluation 구현.
- 교차 언어 test sets 유지 (4개 언어 병렬 쿼리).
- **Faithfulness**는 기본 **하드 릴리스 게이트**입니다.
- **Answer relevancy**는 기본 **진단 전용**입니다(항상 보고; 러너 실패 조건 아님). 명시적 옵션(예: `--gate-answer-relevancy`)으로만 하드 게이트가 될 수 있습니다.
- 모든 major milestone은 runnable Gate Verification commands와 PASS criteria를 포함해야 함.

**Staging validation gates** (절차와 증거는 `docs/staging-cutover-bge.md`):

- Fixed EN / KO / JA / ZH smoke (message + session_id; auto language detection).
- **BGE → Hash → BGE** rollback drill.
- 합의된 latency, memory, model-lifecycle, quality gates 하의 representative soak.

**Release posture:**

- **M8-B staging validation PASS**는 **production cutover planning** 시작을 허용함.
- **Production deployment approval**은 별도 결정이며 staging validation만으로 부여되지 않음.

## 4. Development & Workflow Rules

- 사람이 실행을 승인하기 전까지 **Plan Mode**에서만 작업.
- 명확한 milestones로 조직 (M0, M1, M2, ...).
- 모든 milestone은 명시적 **Gate Verification** commands와 PASS criteria로 끝나야 함.
- 의존성 관리는 `uv` 사용 (`pyproject.toml` 수정 후 `uv sync` / `uv add`).
- plan 승인 후 clean diffs 유지.
- 처음부터 type hints, 적절한 error handling, logging, tests 포함.
- 향후 새 언어 추가가 쉽도록 시스템 설계.

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
| Production packaging (repository templates, Compose, startup policy, runbook) | COMPLETE (M9; 컷오버 승인 아님) |
| Production cutover approval / deployment | PLANNED (미승인) |
| Comprehensive website integration documentation | PARTIAL |

---

**Terminology:** M8-B technical implementation · M8-B staging validation ·
production cutover planning · production deployment approval · M9 production
packaging · BGE collection (`onlybook_faq_bge_m3_v1`) · hash rollback collection
(`onlybook_faq`) · paired `PREFER_BGE` / `QDRANT_COLLECTION`.
