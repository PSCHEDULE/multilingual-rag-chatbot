# CJK Chunking Strategy

## Decision

**Primary:** LlamaIndex `SemanticSplitterNodeParser` driven by **BGE-M3** embeddings  
(`BAAI/bge-m3`), so chunk boundaries follow meaning shifts rather than English-centric
whitespace / fixed token windows.

**Why not default `RecursiveCharacterTextSplitter`?**

| Approach | Issue for KO/JA/ZH |
|----------|-------------------|
| Whitespace / char recursive defaults | CJK often has little or no inter-word space; splits land mid-phrase |
| Fixed token windows only | Ignores topic shifts; cuts entities and honorific clauses |
| Semantic splitter + multilingual embedder | Aligns breaks with embedding distance spikes across all four languages |

**Offline / CI:** The same `SemanticSplitterNodeParser` path runs with a deterministic
bag-of-character-n-gram embedder when BGE-M3 weights are not loaded (`prefer_bge_m3=False`).
Production and quality reviews should set `--prefer-bge-m3` (or load HF embeddings)
when hardware allows.

**Fallback (if manual review fails):**

- Korean: `kiwipiepy` morpheme-aware pre-segmentation  
- Japanese: SudachiPy  
- Chinese: punctuation / sentence secondary splits  
- Always available: `cjk_sentence_pack` packer in `app/ingestion/chunking.py`

## Parameters

| Setting | Default | Role |
|---------|---------|------|
| `CHUNK_BREAKPOINT_PERCENTILE` | 95 | Semantic breakpoint sensitivity |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense model for production semantic split |
| buffer_size | 1 | LlamaIndex semantic buffer |

## Manual quality review (M1 gate)

Reviewed via `scripts/chunk_preview.py` on `data/sample_docs` (≥3 docs per language).  
Mode used for this recorded review: **semantic splitter + offline n-gram embedder**
(CI-reproducible). Spot-check with BGE-M3 recommended before production ingest.

### Korean (`ko`) — 3 documents

| Document | Chunks (approx) | Boundary quality | Notes | Pass |
|----------|-----------------|------------------|-------|------|
| `ko/refund_policy.md` | 1–3 | Good | Title + policy paragraphs stay coherent; no mid-Hangul-word cuts | PASS |
| `ko/shipping_info.md` | 1–3 | Good | Shipping tiers remain grouped | PASS |
| `ko/account_security.md` | 1–3 | Good | 2FA guidance not split mid-sentence | PASS |

### Japanese (`ja`) — 3 documents

| Document | Chunks (approx) | Boundary quality | Notes | Pass |
|----------|-----------------|------------------|-------|------|
| `ja/refund_policy.md` | 1–3 | Good | Mixed kanji/kana sentences intact | PASS |
| `ja/shipping_info.md` | 1–3 | Good | 配送 windows stay together | PASS |
| `ja/account_security.md` | 1–3 | Good | 2FA steps not fragmented | PASS |

### Chinese (`zh`) — 3 documents

| Document | Chunks (approx) | Boundary quality | Notes | Pass |
|----------|-----------------|------------------|-------|------|
| `zh/refund_policy.md` | 1–3 | Good | Full-width punctuation respected | PASS |
| `zh/shipping_info.md` | 1–3 | Good | 运费规则 coherent | PASS |
| `zh/account_security.md` | 1–3 | Good | 安全建议 not mid-clause split | PASS |

### English (`en`) — 3 documents (completeness)

| Document | Pass | Notes |
|----------|------|-------|
| `en/refund_policy.md` | PASS | Semantic / packer keeps FAQ sections usable |
| `en/shipping_info.md` | PASS | |
| `en/account_security.md` | PASS | |

## Checks (PASS criteria)

- Chunks rarely split mid-sentence / mid-entity without justification  
- No empty chunks; multi-topic docs can split at semantic/paragraph breaks  
- Metadata includes correct `language` + `source`  
- Failures → tune `CHUNK_BREAKPOINT_PERCENTILE` or enable language-specific fallback  

## Re-run review

```bash
uv run python scripts/chunk_preview.py --input data/sample_docs --out /tmp/chunk_preview.json
# Optional production embedder:
# uv run python scripts/chunk_preview.py --input data/sample_docs --out /tmp/chunk_preview_bge.json --prefer-bge-m3
```
