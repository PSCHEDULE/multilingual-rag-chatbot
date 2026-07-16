# SSE Chat API Contract (frozen for widget)

**Endpoint:** `POST /v1/chat/stream`  
**Content-Type (request):** `application/json`  
**Response:** `text/event-stream`

## Request body

```json
{
  "message": "환불 정책이 어떻게 되나요?",
  "session_id": "optional-uuid",
  "language": "ko",
  "metadata_filters": { "category": "refund_policy" }
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `message` | yes | User utterance |
| `session_id` | no | Continuity; server generates if omitted |
| `language` | no | Explicit `ko` / `en` / `ja` / `zh` |
| `metadata_filters` | no | Reserved for retrieval filters |

## Events

### `meta`

```text
event: meta
data: {"language":"ko","route":"simple_retrieve","session_id":"..."}
```

### `token`

```text
event: token
data: {"text":"환불"}
```

Tokens may be characters or larger chunks depending on the LLM provider.

### `sources`

```text
event: sources
data: {"items":[{"title":"What is the refund policy?","score":0.91,"source":"faq-en-Q18"}]}
```

| Field | Notes |
|-------|--------|
| `title` | User-facing label (prefer FAQ question text) |
| `score` | Rerank / retrieval score |
| `source` | **Language-unique public id** — preferred form `faq-{lang}-{faq_id}` (e.g. `faq-ja-Q15`, `faq-ko-Q18`). Falls back to filename stem if `faq_id` is missing. Full internal paths (e.g. `data/onlybook_faq/...`) are **not** exposed; they remain in server logs. |

May arrive before or interleaved with tokens; widget should accept either order.

### `error`

```text
event: error
data: {"message":"..."}
```

### `done`

```text
event: done
data: {"finish_reason":"stop","session_id":"...","answer":"full text optional"}
```

`finish_reason` is `stop` on success or `error` after an `error` event.

## Client notes

- Prefer `fetch` + streaming body reader (POST SSE). `EventSource` is GET-only.
- CORS must allow the widget origin.
- This contract is **frozen** as of M5; breaking changes require versioning.
