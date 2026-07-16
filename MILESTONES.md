# Project milestones status

Living status for the multilingual RAG customer-support chatbot
(KO / EN / JA / ZH).

**Last updated:** 2026-07-16

| Milestone | Status |
|-----------|--------|
| M0–M6 (bootstrap → widget) | Complete |
| **M7 Evaluation** | **Complete** (pipeline stable; answer_relevancy diagnostic only) |
| **M8-A FAQ atomic chunking** | **FINAL PASS** |
| **M8-B BGE-M3 embedding migration** | **Technical implementation complete** |
| M8-B staging cutover | **In progress** (env, image, deploy pipeline) |
| M9 Docker / production packaging | Not started (after staging sign-off) |

**LLM (current):** OpenAI **`gpt-4o-mini`** (`LLM_PROVIDER=openai`).

**Operational runbook:** [docs/staging-cutover-bge.md](docs/staging-cutover-bge.md)

---

## Current decision snapshot

| Topic | Decision |
|-------|----------|
| Active quality work | Staging BGE cutover validation (not more core M8-B feature work) |
| Hash collection `onlybook_faq` | Keep as rollback until retention policy after prod BGE stabilizes |
| BGE collection | `onlybook_faq_bge_m3_v1` (dense **1024**, hybrid sparse) |
| Cutover mechanism | Config only: `QDRANT_COLLECTION` + `PREFER_BGE` (must stay paired) |
| RAGAS answer_relevancy | Diagnostic only — not a hard release gate |
| Production cutover | Only after staging BGE → Hash → BGE drill + soak |

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

- Staging/production env flipped as the permanent active path
- Dockerfile / CI always baking `uv sync --group bge` (in progress)
- Full production cutover sign-off

---

## 4. Staging cutover — in progress

| Item | Status |
|------|--------|
| Local BGE collection + preflight | Done in lab (see runbook) |
| Staging env vars (`PREFER_BGE`, collection, `APP_ENV=staging`, …) | Prepare per runbook |
| Image / pipeline includes `bge` group | In progress |
| Staging BGE → Hash → BGE rollback drill + soak | Required before production planning |

Source of truth for operators: **`docs/staging-cutover-bge.md`**.

---

## 5. Known remaining issues (decision-oriented)

1. **Staging/production traffic** still needs formal cutover (config + deploy), not more embedding features.  
2. **Hash collection** remains the rollback path; do not delete while it is the rollback target.  
3. **Image/deps:** ensure staging/prod builds install the optional **`bge`** group.  
4. **`/ready`** validates config, embedder snapshot, and collection dim; sparse/payload corpus inventory remains **manual preflight**.  
5. **M9** (prod packaging hardening) after staging sign-off.

---

## 6. Next priority

1. Finish staging deployment pipeline (env + image with `bge` group).  
2. Execute runbook: preflight → BGE start → `/ready` → multilingual smoke → **rollback drill** → soak.  
3. Only then plan production cutover.  
4. M9 as needed for production packaging.

---

## Document history

| Date | Note |
|------|------|
| 2026-07-15 | M8-A FINAL PASS; M8-B plan drafted |
| 2026-07-16 | M8-B technical implementation complete; staging cutover prep in progress |
