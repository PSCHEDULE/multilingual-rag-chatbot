# Project milestones status

Living status for the multilingual RAG customer-support chatbot
(KO / EN / JA / ZH).

**Last updated:** 2026-07-21

| Milestone | Status |
|-----------|--------|
| M0–M6 (bootstrap → widget) | Complete |
| **M7 Evaluation** | **Complete** (pipeline stable; answer_relevancy diagnostic only) |
| **M8-A FAQ atomic chunking** | **FINAL PASS** |
| **M8-B BGE-M3 embedding migration** | **Technical implementation complete** |
| **M8-B staging cutover validation** | **PASS (2026-07-20)** |
| **M9 Docker / production packaging** | **REPOSITORY PRODUCTION PACKAGING: PASS (2026-07-21)** |

**LLM (current):** OpenAI **`gpt-4o-mini`** (`LLM_PROVIDER=openai`).

**Operational runbooks:**

- Staging: [docs/staging-cutover-bge.md](docs/staging-cutover-bge.md) (§10.2 staging validation record)
- Production packaging: [docs/production-deployment.md](docs/production-deployment.md)

---

## Current decision snapshot

| Topic | Decision |
|-------|----------|
| Active quality work | Production cutover remains **not approved**; packaging is in-repo |
| Staging validation prerequisite | **Complete** (2026-07-20) — smoke, lifecycle, BGE→Hash→BGE drill, soak |
| Hash collection `onlybook_faq` | Keep as rollback until retention policy after prod BGE stabilizes |
| BGE collection | `onlybook_faq_bge_m3_v1` (dense **1024**, hybrid sparse) |
| Cutover mechanism | Config only: `QDRANT_COLLECTION` + `PREFER_BGE` (must stay paired) |
| RAGAS answer_relevancy | Diagnostic only — not a hard release gate |
| **M9 repository production packaging** | **PASS** (2026-07-21) |
| **Actual production cutover** | **NOT APPROVED** |
| **Platform decisions** | **REQUIRED** (TLS/ingress, secrets platform, registry, Qdrant HA, …) |

---

## 1. M7 (evaluation) — complete

- Cross-lingual eval set and stable RAGAS-style runner in place.
- Live OnlyBook 120-row run: pipeline stable; **faithfulness** is the default hard
  release gate; **answer_relevancy** is diagnostic by default (optional
  `--gate-answer-relevancy` hard gate).
- Weak retrieval drivers identified: FAQ Q/A split under generic chunking + weak
  hash dense embeddings → led to M8-A / M8-B split.

---

## 2. M8-A (FAQ atomic chunking) — FINAL PASS

- FAQ docs route to atomic Q+A chunks (multi-entry, oversized answers with question
  repeated on every part).
- FAQ contextualization off by default; metadata (`faq_id`, language, question, …)
  preserved into Qdrant payload.
- Generic documents do not use `faq_atomic`.
- Operational re-index of live hash corpus was **not** required for M8-A acceptance
  (code + tests); BGE re-index uses M8-A path.

---

## 3. M8-B (BGE-M3) — technical implementation complete

High-level outcomes (not an implementation log):

| Area | Outcome |
|------|---------|
| Embeddings | Optional `bge` dependency group; BGE-M3 dense path when `PREFER_BGE=true` |
| Index | Versioned collection **`onlybook_faq_bge_m3_v1`** (1024-d); hash baseline retained |
| Retrieval quality | Side-by-side checks favored BGE + atomic FAQ, especially paraphrase / CJK |
| Pruning | Post-rerank keep Top-1; Top-2/3 only if score thresholds pass; max 3 sources |
| UX sources | Prefer question as title; public id `faq-{lang}-Q<number>`; no internal paths in SSE |
| Ops probes | `/health` liveness; `/ready` lightweight readiness (startup snapshot + Qdrant metadata) |
| Safety | Collection/embedder pairing guards; protected hash collections on recreate |

**Explicitly not done under “technical complete”:**

- Production cutover approval or production traffic switch

Staging validation of the technical path is recorded separately in §4 / runbook
§10.2 (PASS 2026-07-20).

---

## 4. Staging cutover validation — PASS (2026-07-20)

| Item | Status |
|------|--------|
| Local BGE collection + preflight | Done |
| Staging env (`PREFER_BGE`, collection, `APP_ENV=staging`, …) | Done (staging BGE mode) |
| Image / pipeline includes `bge` group | Done for staging image in use |
| Multilingual fixed smoke (EN/KO/JA/ZH) | **PASS** |
| BGE / reranker lifecycle reuse | **PASS** |
| Staging BGE → Hash → BGE rollback drill | **PASS** |
| 30-minute / 100-request soak | **PASS** (see runbook §10.2 for metrics) |

Source of truth for operators: **`docs/staging-cutover-bge.md`** (§10.2 for the
validation record and evidence caveat).

---

## 5. M9 — repository production packaging — PASS (2026-07-21)

### 5.1 M9-A inspection (evidence-only; closed)

Boundary and packaging inspection on master `35c434d…` established that staging
packaging existed, production-named runtime policy paths were largely **absent**,
and production + mock **startup hard rejection** was **ABSENT** prior to the
implementation commit. Capability for `APP_ENV` / `MOCK_LLM` existed in application
and packaging configuration. This closed the inspection phase; it was **not**
production approval.

### 5.2 Implemented packaging (this milestone)

| Item | Status |
|------|--------|
| `APP_ENV=production` packaging template | `.env.production.example` |
| `MOCK_LLM=false` production configuration | example + Compose override |
| Production + mock hard rejection | `Settings` model_validator + lifespan re-assert |
| Selected-provider secret validation | Required when production + openai/xai/grok |
| Production Compose override + external `APP_IMAGE` | `docker-compose.production.yml` |
| restart / stop_grace / healthcheck | on production override |
| No literal secrets in repo templates | placeholders only |
| Production deploy/rollback runbook | `docs/production-deployment.md` |
| Targeted unit tests | `tests/unit/test_production_config.py` |

### 5.3 Explicit non-approvals

| Decision | Status |
|----------|--------|
| **M9 REPOSITORY PRODUCTION PACKAGING** | **PASS** |
| **ACTUAL PRODUCTION CUTOVER** | **NOT APPROVED** |
| **PLATFORM DECISIONS REQUIRED** | TLS/ingress/DNS, secret manager, registry/signing, Qdrant HA/auth/backups, monitoring, formal traffic switch |

---

## 6. Known remaining issues (decision-oriented)

1. **Production cutover** still needs platform work, approval, and deploy — packaging
   does **not** approve production traffic.
2. **Hash collection** remains the rollback path; do not delete while it is the
   rollback target.
3. **`/ready`** validates config, embedder snapshot, and collection dim;
   sparse/payload corpus inventory remains **manual preflight**.
4. **Platform decisions** listed in §5.3 remain outside repository packaging.

---

## 7. Next priority

1. Platform owners: complete decisions in §5.3.
2. Formal **production cutover** approval process (separate from M9 packaging).
3. Retain hash rollback until retention policy after prod BGE stabilizes.

---

## Document history

| Date | Note |
|------|------|
| 2026-07-15 | M8-A FINAL PASS; M8-B plan drafted |
| 2026-07-16 | M8-B technical implementation complete; staging cutover prep in progress |
| 2026-07-20 | M8-B staging cutover validation PASS (smoke, lifecycle, rollback drill, soak); production planning eligible; M9 still not started |
| 2026-07-21 | M9 repository production packaging PASS; actual production cutover NOT APPROVED; platform decisions required |
