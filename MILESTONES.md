# Project milestones status

Living status for the multilingual RAG customer-support chatbot
(KO / EN / JA / ZH).

**Last updated:** 2026-07-20

| Milestone | Status |
|-----------|--------|
| M0–M6 (bootstrap → widget) | Complete |
| **M7 Evaluation** | **Complete** (pipeline stable; answer_relevancy diagnostic only) |
| **M8-A FAQ atomic chunking** | **FINAL PASS** |
| **M8-B BGE-M3 embedding migration** | **Technical implementation complete** |
| **M8-B staging cutover validation** | **PASS (2026-07-20)** |
| M9 Docker / production packaging | Not started |

**LLM (current):** OpenAI **`gpt-4o-mini`** (`LLM_PROVIDER=openai`).

**Operational runbook:** [docs/staging-cutover-bge.md](docs/staging-cutover-bge.md)
(including §10.2 staging validation record).

---

## Current decision snapshot

| Topic | Decision |
|-------|----------|
| Active quality work | Production cutover **planning** and M9 packaging preparation |
| Staging validation prerequisite | **Complete** (2026-07-20) — smoke, lifecycle, BGE→Hash→BGE drill, soak |
| Hash collection `onlybook_faq` | Keep as rollback until retention policy after prod BGE stabilizes |
| BGE collection | `onlybook_faq_bge_m3_v1` (dense **1024**, hybrid sparse) |
| Cutover mechanism | Config only: `QDRANT_COLLECTION` + `PREFER_BGE` (must stay paired) |
| RAGAS answer_relevancy | Diagnostic only — not a hard release gate |
| Production cutover | **Planning may begin**; not approved or deployed (see runbook §10.2) |

---

## 1. M7 (evaluation) — complete

- Cross-lingual eval set and stable RAGAS-style runner in place.
- Live OnlyBook 120-row run: pipeline stable; **faithfulness** useful as a signal;
  **answer_relevancy** kept as diagnostic only.
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
- Full production packaging sign-off (M9)

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

## 5. Known remaining issues (decision-oriented)

1. **Production cutover** still needs planning, approval, and deploy — staging
   validation does **not** approve production.
2. **Hash collection** remains the rollback path; do not delete while it is the
   rollback target.
3. **M9** (Docker / production packaging hardening) is **not started**.
4. **`/ready`** validates config, embedder snapshot, and collection dim;
   sparse/payload corpus inventory remains **manual preflight**.
5. Production environment, deployment approval, and final packaging remain
   future work.

---

## 6. Next priority

1. Begin **production cutover planning** (same checklist as staging; not
   approval).
2. Start **M9** Docker / production packaging as needed.
3. Retain hash rollback until retention policy after prod BGE stabilizes.

---

## Document history

| Date | Note |
|------|------|
| 2026-07-15 | M8-A FINAL PASS; M8-B plan drafted |
| 2026-07-16 | M8-B technical implementation complete; staging cutover prep in progress |
| 2026-07-20 | M8-B staging cutover validation PASS (smoke, lifecycle, rollback drill, soak); production planning eligible; M9 still not started |
