# Hybrid Retrieval & Reranker Cost / Latency

## Pipeline

1. **Hybrid search (Qdrant)** — dense (BGE-M3 in production, hash embedder offline) + **hashed log-TF sparse vectors** (not full BM25), fused with RRF. Retrieve `top_n` (default **40**).
2. **Rerank** — BGE-reranker-v2-m3 (production) or lexical overlap (offline). Keep `top_k` (default **6**).

### Sparse vectors (implementation note)

Sparse encoding (`app.retrieval.embeddings.sparse_bm25_vector`) is a **deterministic hashed term-frequency** scheme:

- tokenize with a multilingual word/CJK regex
- hash each token into a fixed index space
- store **1 + log(tf)** weights (log-TF style), not IDF / classic BM25 scoring

Qdrant receives these sparse vectors for hybrid RRF fusion. Documentation and code names may still say “bm25” for historical reasons; do **not** treat the sparse path as a true BM25 implementation.

Facade: `app.retrieval.hybrid.retrieve_and_rerank` logs and returns:

| Metric | Meaning |
|--------|---------|
| `retrieval_ms` | Qdrant hybrid (or dense fallback) latency |
| `rerank_ms` | Cross-encoder / lexical rerank latency |
| `total_ms` | End-to-end retrieve+rerank |

## Cost & latency impact of BGE-reranker-v2-m3

| Factor | Impact |
|--------|--------|
| **Model load** | Cross-encoder weights (~1GB class) loaded once per process; cold start seconds–tens of seconds on CPU |
| **Per-request cost** | Scores `top_n` (query, passage) pairs — compute ≈ O(top_n). Larger `top_n` raises latency nearly linearly |
| **Hardware** | GPU: typically low tens of ms for 20–40 pairs; CPU: often 100–800+ ms depending on hardware |
| **API $** | Local weights → no per-token API fee; infra cost is memory/CPU/GPU |
| **Quality** | Usually large lift on noisy hybrid candidates, especially cross-lingual and short CJK queries |

### Tuning knobs

- Lower `RETRIEVAL_TOP_N` (e.g. 20) for latency-sensitive paths.
- Lower `RETRIEVAL_TOP_K` if the generator only needs 3–5 contexts.
- Disable rerank for health/smoke (use lexical) when `prefer_bge=False`.
- Cache nothing by default (queries are unique); consider embedding cache at ingest only.

### Measured baseline (record on target hardware)

| Environment | top_n | retrieval_ms | rerank_ms | Notes |
|-------------|-------|--------------|-----------|-------|
| Offline CI (hash + lexical) | 20 | typically &lt; 50 | typically &lt; 5 | No BGE weights |
| Production (BGE-M3 + bge-reranker-v2-m3) | 40 | *measure & fill* | *measure & fill* | Run once post-deploy |

```bash
uv run python -c "
from app.retrieval.hybrid import retrieve_and_rerank
r = retrieve_and_rerank('환불 정책', language='ko', top_n=20, top_k=5)
print(r.metrics)
"
```

## Metadata filters

Allowlisted exact-match payload filters (chat `metadata_filters` and hybrid kwargs):
`language`, `category`, `source`, `doc_type`, `faq_id`, `intent`.

Automatic language filtering (`RETRIEVAL_LANGUAGE_FILTER`) still applies when no
explicit `language` filter is supplied. Explicit request filters use Qdrant
`MatchValue` conditions only (no arbitrary query injection).
