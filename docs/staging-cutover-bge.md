# Staging cutover guide — BGE-M3 (`onlybook_faq_bge_m3_v1`)

Practical operational runbook for deploying retrieval to the BGE-M3 collection in
**staging**. Design goals: safe soft cutover, reversible hash rollback, and
verifiable readiness before production planning.

**Rollback target:** collection `onlybook_faq` (hash dense dim **384**).
Do not delete or recreate it while it remains the active rollback collection.
Retain it until production BGE stabilization is complete and the agreed
rollback-retention period has expired.

---

## 1. Prerequisites

| Check | Notes |
|-------|--------|
| Staging Qdrant reachable | Matches `QDRANT_URL` |
| Collection `onlybook_faq_bge_m3_v1` exists | Dense dim **1024**, hybrid sparse present, FAQ points loaded |
| Collection `onlybook_faq` present | Hash dim **384**; rollback only |
| Host capacity | Disk/RAM for BGE-M3 + torch; CPU works but is slower |
| App cutover code deployed | `PREFER_BGE`, guards, pruning, display titles, public source ids, `/health` + `/ready` |

### Manual Qdrant preflight (before deploy)

`/ready` does **not** validate sparse-vector config, payload fields, or corpus
completeness. Complete this **operator preflight** before cutover (or as step 1
of the cutover order):

| Manual preflight item | Expected |
|-----------------------|----------|
| `onlybook_faq_bge_m3_v1` exists | Yes |
| Dense vector dimension | **1024** |
| Sparse-vector config | Hybrid sparse vector present (as used at ingest) |
| Corpus size / manifest | Point count matches expected FAQ corpus (or recorded baseline) |
| Sample points include payload fields | `language`, `faq_id`, `title` (and typically `question`) |
| Language × faq_id coverage | Expected EN/KO/JA/ZH combinations present; no unexpected duplicates of the same `(language, faq_id)` |
| Rollback collection `onlybook_faq` | Exists with dense dim **384** |

**Who checks what:**

| Check | Automatic (`/ready` after startup) | Manual preflight | Future automation (not implemented) |
|-------|------------------------------------|------------------|-------------------------------------|
| Config pair PREFER_BGE ↔ collection | Yes | — | — |
| Embedder type + dim (from startup snapshot) | Yes | — | — |
| Qdrant reachability | Yes (lightweight) | Yes | — |
| Collection exists + dense dim | Yes (metadata) | Yes | — |
| Sparse-vector configuration | **No** | **Yes** | Recommended |
| Payload fields / indexes | **No** | **Yes** (sample points) | Recommended |
| Full corpus inventory / duplicates | **No** | **Yes** | Recommended |
| Reranker loaded | **No** | Optional | Recommended |

---

## 2. Dependencies

Install the optional BGE dependency group on the staging host or image:

```bash
uv sync --group bge
```

Optional (if you also run pytest on that host):

```bash
uv sync --group bge --group dev
```

Confirm packages import:

```bash
uv run python -c "import sentence_transformers, torch; print(sentence_transformers.__version__, torch.__version__)"
```

---

## 3. Environment variables

### BGE cutover (staging active)

```bash
# --- Active BGE cutover ---
QDRANT_URL=https://<staging-qdrant-host>:6333
QDRANT_API_KEY=<staging-secret>
QDRANT_COLLECTION=onlybook_faq_bge_m3_v1
PREFER_BGE=true
RETRIEVAL_LANGUAGE_FILTER=true

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3

# Optional pruning (defaults are fine)
# PRUNE_ABSOLUTE_THRESHOLD=0.05
# PRUNE_RELATIVE_THRESHOLD=0.05

# LLM (staging)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<staging-secret>
MOCK_LLM=false

# App
APP_ENV=staging
LOG_LEVEL=INFO
```

If staging Qdrant does **not** require authentication, **omit** `QDRANT_API_KEY`
entirely rather than setting an empty value. Never commit real secrets.

### Consistency rule (mandatory)

| Mode | `QDRANT_COLLECTION` | `PREFER_BGE` | Dense path used for retrieval |
|------|---------------------|--------------|--------------------------------|
| BGE cutover | `onlybook_faq_bge_m3_v1` | `true` | BGE-M3 (dim **1024**) |
| Hash rollback | `onlybook_faq` | `false` | Offline hash embedder (dim **384**) |

Never pair `PREFER_BGE=true` with `onlybook_faq`, or `PREFER_BGE=false` with
`onlybook_faq_bge_m3_v1`. Startup and readiness guards reject that mismatch.

### Hash mode and `EMBEDDING_MODEL` (verified from code)

When `PREFER_BGE=false`, the application selects the **hash embedding path**
(`OfflineHashEmbedder`, dim 384) for retrieval and indexing helpers that honor
`prefer_bge`. It does **not** use `BAAI/bge-m3` for dense retrieval even if
`EMBEDDING_MODEL=BAAI/bge-m3` remains set in the environment.

**Correct hash rollback env changes (required):**

```bash
QDRANT_COLLECTION=onlybook_faq
PREFER_BGE=false
```

Then restart the API. Leaving `EMBEDDING_MODEL=BAAI/bge-m3` unchanged is OK for
rollback.

---

## 4. Startup verification

### Liveness vs readiness

| Endpoint | Role | Success |
|----------|------|---------|
| `GET /health` | Liveness — process is up | HTTP **200** |
| `GET /ready` | Readiness — safe to send RAG traffic | HTTP **200** and `"ready": true` |

**Kubernetes probes:**

```text
liveness:  GET /health
readiness: GET /ready
```

`/ready` is intended for periodic readiness probing. After the lightweight
implementation:

- does **not** reload the embedder
- does **not** clear embedder model caches (`cache_clear`)
- does **not** run embedding or reranking inference
- reuses the **startup embedder snapshot**
- may perform a **lightweight** Qdrant connectivity + collection metadata check

### Liveness

```bash
curl -sf "$STAGING_API/health"
```

Expect HTTP 200 and a body similar to:

```json
{"status": "ok", "live": true, "app": "Multilingual RAG Chatbot", "llm_provider": "openai"}
```

`/health` does **not** initialize retrieval or load BGE.

### Readiness — success check (fail on non-2xx)

```bash
curl -sf "$STAGING_API/ready" >/dev/null
echo "ready_http_ok=$?"
```

### Readiness — diagnostic body (always print JSON; do not use `-f`)

```bash
curl -sS "$STAGING_API/ready" | jq .
```

On failure you should still see JSON with `"ready": false` and an `errors` list
(HTTP **503**). Do **not** pipe a bare HTTP status string into `jq`.

### Repeated-readiness check (before relying on periodic probes)

```bash
for i in $(seq 1 20); do
  curl -sS -o /dev/null \
    -w 'status=%{http_code} total=%{time_total}\n' \
    "$STAGING_API/ready"
  sleep 1
done
```

Confirm:

- All responses are HTTP **200** in a healthy BGE deployment
- Latency stays stable (no multi-second spikes every probe)
- Logs do **not** show repeated BGE model loads
- Process/container memory does **not** grow continuously

The shell loop alone does **not** prove memory stability — pair it with
process/container metrics (RSS, cgroup memory) and application logs.

### Readiness response fields (current implementation)

| Field | Meaning |
|-------|---------|
| `ready` / `status` | All hard checks passed → `ready` / `not_ready` |
| `prefer_bge` / `collection` | Active cutover config |
| `expected_dense_dim` | **1024** (BGE) or **384** (hash) |
| `probe` | `lightweight` for `/ready` path |
| `checks.config_pair` | `PREFER_BGE` ↔ collection consistent |
| `checks.embedder` | From **startup snapshot** (type + dim); `source=startup_snapshot` |
| `checks.qdrant_reachable` | Lightweight Qdrant connectivity |
| `checks.collection_exists` | Configured collection present |
| `checks.collection_dim` | Dense size matches mode |
| `checks.collection_points` | Point count (informational) |
| `startup` | Snapshot from process startup |

### Startup logs (BGE mode)

After deploy/restart with BGE env, expect lines similar to:

```text
Loaded dense embedder BAAI/bge-m3 dim=1024
Retrieval startup OK: collection=onlybook_faq_bge_m3_v1 prefer_bge=true embedder_dim=1024 collection_dim=1024
Retrieval startup checks: ok_bge_mode
```

### Readiness implementation (verified)

```text
Application startup:
- validate PREFER_BGE and collection pairing
- initialize the selected embedder once (BGE when PREFER_BGE=true)
- validate embedder dimension (1024 BGE / 384 hash)
- validate Qdrant reachability when available
- validate collection existence and dense dimension
- store a readiness/startup snapshot on app.state

GET /ready (lightweight):
- read the cached startup embedder validation
- re-check config pair against current settings
- optionally re-check Qdrant connectivity and collection dense dim (metadata only)
- do not call get_dense_embedder.cache_clear()
- do not recreate or reload the BGE model
- do not run embedding or reranking inference
- return HTTP 503 when not ready
```

**Not checked automatically today (manual preflight / future):**

- Reranker loaded
- Sparse-vector configuration details
- Payload field indexes (`language`, `faq_id`, `title`)

---

## 5. Smoke tests

Run after BGE cutover (chat SSE). **Default tests omit `language`** so automatic
language detection is exercised. Send only `message` and `session_id`.

### Multilingual matrix (auto language detection)

| Lang | Message | Expect |
|------|---------|--------|
| EN | How do I get a refund? | EN refund FAQ; public `source` like `faq-en-Q18` |
| KO | 환불은 어떻게 받나요? | KO refund FAQ; language-aligned |
| JA | 無料体験が終わったら自動で課金？ | Trial billing FAQ |
| ZH | 如何申请退款？ | ZH refund FAQ |

### Default SSE example (no forced language)

```bash
curl -sN -X POST "$STAGING_API/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I get a refund?",
    "session_id": "staging-bge-en-001"
  }'
```

Repeat with KO/JA/ZH messages and distinct `session_id` values
(e.g. `staging-bge-ko-001`). These default smokes **validate automatic language
detection**.

### Optional override (forced language)

Only if you need to isolate router/filter behavior:

```bash
curl -sN -X POST "$STAGING_API/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I get a refund?",
    "session_id": "staging-bge-en-forced-001",
    "language": "en"
  }'
```

Do **not** treat forced-language as a substitute for the default auto-detect
matrix.

### Common smoke pass / fail criteria

A smoke run **passes** only if all of the following hold:

| Criterion | Pass |
|-----------|------|
| `meta.language` | Matches the input language (auto-detect or forced) |
| Fixed FAQ matrix (EN/KO/JA/ZH known queries) | **`meta.route == simple_retrieve`** (required, not optional) |
| Expected FAQ | Present as **Top-1** in sources (known gold FAQ) |
| `sources` size | Between **1 and 3** items (post-pruning) |
| `source` public id | Matches `^faq-(en\|ko\|ja\|zh)-Q\d+$` (e.g. `faq-ko-Q18`) or safe filename-stem fallback |
| Path leak | `source` must **not** contain `/`, `\`, `data/onlybook_faq`, or other internal paths |
| Titles | Readable, user-facing FAQ questions (not `Q15 Faq 15 Ja`) |
| Logical SSE sequence | `meta` → `sources` → `token`+ → `done` without `error` |
| SSE keepalives | Comments such as `: ping` may appear between events; ignore them in parsers |
| Tokens vs done | Concatenated token text matches `done.answer` when both present |
| Session | `done.session_id` matches the request `session_id` |
| Citations | Answer has **no** unmapped `[1]`, `[2]`, or similar bare numeric refs |

Expected **logical** SSE sequence (events need not be physically adjacent):

```text
meta → sources → token+ → done

SSE keepalive comments such as ": ping" may appear between events.
Clients and smoke-test parsers must ignore them and continue processing.
```

### Recommended edge-case tests

Route expectations for edge cases are evaluated **separately** from the fixed
FAQ matrix (do not require `simple_retrieve` for these unless observed):

1. **Compound / multi-aspect question** where a valid secondary document may
   survive pruning as Top-2 (score above absolute and relative thresholds).
   Confirm sources still ≤ 3 and Top-1 remains the primary FAQ.

2. **Unrelated / out-of-knowledge question**
   Confirm the system does not invent unsupported definitive policy; empty or
   low-confidence retrieval should not produce fabricated operational claims.

---

## 6. Logging and monitoring

| Log / metric | Why |
|--------------|-----|
| `retrieve_and_rerank collection=... prefer_bge=... language_filter=... retrieval_ms=... rerank_ms=...` | Mode correctness + latency |
| `prune_docs candidates=... kept=...` | Context size / noise control |
| `sources_selected title=... public_id=... internal_source=...` | Client-safe id vs full path (debug only) |
| Empty hits / 5xx rate | Retrieval or Qdrant issues |
| Embedder / reranker load logs | Detect unexpected reloads |
| Memory / OOM | Capacity under soak |
| First-request latency | Cold start (CPU can be high) |

---

## 7. Rollback

### Switch to hash

```bash
QDRANT_COLLECTION=onlybook_faq
PREFER_BGE=false
# restart API
```

`EMBEDDING_MODEL` may remain `BAAI/bge-m3`; with `PREFER_BGE=false` retrieval
uses the hash embedder, not BGE-M3 (see §3).

### Verify hash mode

```bash
curl -sf "$STAGING_API/health" >/dev/null
curl -sf "$STAGING_API/ready" >/dev/null
curl -sS "$STAGING_API/ready" | jq .
```

Expect:

```text
Retrieval startup: hash mode collection=onlybook_faq prefer_bge=false
ok_hash_mode
```

And readiness with dense dim **384**, `ready: true`.

Smoke at least one EN query (and preferably one CJK) with non-empty sources.

### Restore BGE after drill

```bash
QDRANT_COLLECTION=onlybook_faq_bge_m3_v1
PREFER_BGE=true
# restart API
```

```bash
curl -sf "$STAGING_API/ready" >/dev/null
curl -sS "$STAGING_API/ready" | jq .
```

Expect dim **1024**, log `ok_bge_mode`, then re-run a short BGE smoke.

---

## 8. Staging-specific risks and gotchas

| Risk | Mitigation |
|------|------------|
| Env mismatch (BGE collection + hash vectors) | Guards; `GET /ready`; deploy checklist |
| Missing `bge` group on image | Bake `uv sync --group bge` into image/CI |
| Slow first request on CPU | Embedder warmed at startup when `PREFER_BGE=true`; `/ready` no longer reloads BGE |
| Qdrant client/server version skew | Warnings OK if search works; align when possible |
| Empty / wrong staging collection | Pre-check points count and dim **1024** |
| Secrets in logs | Never log API keys; secrets only in env/secret store |
| BGE corpus not on staging Qdrant | Re-ingest or restore `onlybook_faq_bge_m3_v1` before cutover |

**Rollback retention:** Do not delete or recreate `onlybook_faq` while it is the
active rollback target. Keep it until production BGE is stable and the agreed
retention window has expired.

---

## 9. Client sources field

| Layer | `source` value |
|-------|----------------|
| SSE / client | Public id `faq-{lang}-Q<number>` (e.g. `faq-ja-Q15`) or filename-stem fallback |
| Server logs | Full internal path in `sources_selected ... internal_source=...` |

**Primary public ID pattern** (when `faq_id` + language are known):

```text
faq-{lang}-Q<number>
```

Expected regex:

```regex
^faq-(en|ko|ja|zh)-Q\d+$
```

Examples: `faq-en-Q18`, `faq-ko-Q18`, `faq-ja-Q15`, `faq-zh-Q1`.
No zero-padding requirement beyond what the stored `faq_id` already contains.

**Fallback** when `faq_id` is missing: filename stem only (e.g. `Q15_faq_15_ja`),
still without directory paths.

Titles prefer FAQ **question** text (display-title resolution).
Client payloads must not expose paths such as `data/onlybook_faq/...`.

---

## 10. Recommended staging cutover order

1. **Manual Qdrant preflight (§1)** for `onlybook_faq_bge_m3_v1` (dim **1024**,
   sparse config, sample payload fields) and rollback `onlybook_faq` (dim **384**).
2. Install deps: `uv sync --group bge` (or image build).
3. Set BGE env: `QDRANT_COLLECTION=onlybook_faq_bge_m3_v1`, `PREFER_BGE=true`,
   `RETRIEVAL_LANGUAGE_FILTER=true`.
4. Deploy / restart the API.
5. Verify `GET /health` 200 and `GET /ready` 200 (`ready: true`, dim **1024**)
   and startup log `ok_bge_mode`. Run the repeated-readiness loop in §4.
6. Run multilingual smoke (EN/KO/JA/ZH) with **message + session_id only**;
   apply the pass criteria in §5 (`route=simple_retrieve` for the fixed matrix).
7. **Rollback drill (required before production planning):**
   a. Switch to hash: `QDRANT_COLLECTION=onlybook_faq`, `PREFER_BGE=false`, restart.
   b. Verify `/ready` 200, dim **384**, log `ok_hash_mode`.
   c. Smoke ≥1 EN and preferably 1 CJK query; non-empty sources.
   d. Restore BGE: `QDRANT_COLLECTION=onlybook_faq_bge_m3_v1`, `PREFER_BGE=true`, restart.
   e. Verify `/ready` 200, dim **1024**, log `ok_bge_mode`.
   f. Re-run a short BGE smoke (e.g. EN + KO refund).
8. **Soak** under the criteria in §10.1 (leave staging on BGE after a successful drill).
9. Record the rollback card (§11) and that the round-trip drill passed.
10. **Only after a successful BGE → Hash → BGE round-trip and soak**, proceed to
    production cutover planning with the same checklist.

### 10.1 Soak acceptance criteria (record agreed values before approval)

Default gate (both conditions required unless you explicitly document an
exception with separate tests):

| Criterion | Default gate |
|-----------|----------------|
| Duration **and** volume | ≥ **30 minutes** **and** ≥ **100** representative requests |
| 5xx | **Zero** |
| Empty sources on known FAQ smokes | **Zero** for the fixed smoke set |
| Latency | P95 below the agreed staging threshold (record the number) |
| Memory | No continuously increasing memory / no OOM |
| Model lifecycle | No repeated unexpected embedder or reranker full reloads under steady traffic |
| Quality | No multilingual Top-1 accuracy regression vs the pre-soak smoke baseline |

If the environment cannot meet **both** time and volume in one window, run and
record **separate** required tests:

- **Time-based soak** — memory stability and unexpected model reloads
- **Volume-based test** — latency, errors, and retrieval quality

Exact thresholds may be adjusted for staging but must be **agreed and recorded**
before staging sign-off.

### 10.2 Staging validation record — 2026-07-20

**Decision: PASS.** Staging BGE cutover validation is complete. Production
cutover **planning may begin**. This is **not** production cutover approval or
production readiness.

| Check | Result |
|-------|--------|
| Multilingual fixed smoke (EN/KO/JA/ZH) | PASS |
| BGE dense lifecycle reuse | PASS |
| Reranker lifecycle reuse | PASS |
| BGE → Hash → BGE rollback round-trip drill | PASS |
| 30-minute / 100-request soak | PASS |

**Authoritative soak metrics (formal detailed F2E report):**

| Metric | Value |
|--------|--------|
| Date | 2026-07-20 |
| Decision | PASS |
| Requests | **100** (EN=25, KO=25, JA=25, ZH=25) |
| Retries | **0** |
| Observation duration | **1967.7 s** (includes **120 s** cooldown) |
| Overall P50 total latency | **8.32 s** |
| Overall P95 total latency | **10.90 s** |
| Overall maximum total latency | **12.51 s** |
| Agreed P95 gate | **≤ 20.0 s** |
| HTTP / non-200 failures | **0** |
| SSE errors | **0** |
| Empty-source failures | **0** |
| Fixed-FAQ Top-1 failures | **0** |
| API restarts | **0** |
| Qdrant restarts | **0** |
| API OOM | **false** |
| Qdrant OOM | **false** |
| Dense BGE loads | **1 → 1** |
| Reranker loads | **1 → 1** |
| Application startup completions | **1 → 1** |
| Qdrant collections | Unchanged |
| Docker volumes | Unchanged |

**Model lifecycle:** No unexpected dense BGE or reranker reload under soak;
startup completions remain 1.

**Runtime / Qdrant / volume integrity:** API and Qdrant stayed healthy with zero
restarts and no OOM; collection point counts, dimensions, and volume metadata
unchanged.

**Final staging mode (leave staging on BGE):**

```text
QDRANT_COLLECTION=onlybook_faq_bge_m3_v1
PREFER_BGE=true
```

**Evidence caveat:** Temporary raw soak JSON/JSONL artifacts were deleted during
planned cleanup, so exact raw-data recomputation is no longer possible. Two
textual summaries contained different aggregate latency values. The **formal
detailed F2E report** is the authoritative record. Do not state or imply that
those summaries came from separate runs; that cannot be verified.

§5 smoke criteria, §7 rollback procedure, and §10.1 soak gates remain the
reusable operational procedures for future drills.

---

## 11. Quick rollback card

```text
SET  QDRANT_COLLECTION=onlybook_faq
SET  PREFER_BGE=false
RESTART api
CHECK  GET /health  → 200
CHECK  GET /ready   → 200, ready:true, dense dim 384
CHECK  log ok_hash_mode
CHECK  one EN smoke with non-empty sources
```

Restore BGE (after drill or when re-enabling):

```text
SET  QDRANT_COLLECTION=onlybook_faq_bge_m3_v1
SET  PREFER_BGE=true
RESTART api
CHECK  GET /ready → 200, ready:true, dense dim 1024
CHECK  log ok_bge_mode
```

---

## 12. Brief section summary

| Topic | Key points |
|-------|------------|
| Environment variables | BGE: `onlybook_faq_bge_m3_v1` + `PREFER_BGE=true` + language filter. Hash: `onlybook_faq` + `PREFER_BGE=false`. Hash retrieval uses OfflineHashEmbedder even if `EMBEDDING_MODEL` still names BGE-M3. |
| Startup checks | `/health` = liveness. `/ready` = lightweight (startup embedder snapshot + Qdrant metadata); no BGE reload. Use `curl -sf` for success-only; `curl -sS \| jq` for body. |
| Smoke tests | Message + session_id only (auto language). Fixed matrix requires `route=simple_retrieve`, Top-1 FAQ, 1–3 sources, `faq-{lang}-Q<number>`, keepalives ignored, no path leaks, no `[1]`/`[2]`. |
| Rollback | Two env vars + restart; verify dim 384 and hash logs; drill full BGE → Hash → BGE before production planning. |
| Monitoring | Retrieval logs, prune, sources_selected, 5xx, latency, memory, model reloads; repeated `/ready` latency/memory check. |
| Risks | Env mismatch, missing bge deps, CPU cost, version skew, empty collection, secrets. Retain hash collection until retention policy expires. Manual preflight for sparse/payload. |
