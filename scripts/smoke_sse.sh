#!/usr/bin/env bash
# Smoke-test POST /v1/chat/stream SSE endpoint.
# Requires meta + sources + done; fails on transport errors or error events.
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
export MOCK_LLM="${MOCK_LLM:-1}"

BODY='{"message":"What is the refund policy?","language":"en"}'
OUT="$(mktemp)"
trap 'rm -f "${OUT}"' EXIT

# Finite connect + total timeouts; fail on HTTP/transport errors (no || true).
HTTP_CODE="$(
  curl -sS -N \
    --connect-timeout 5 \
    --max-time 60 \
    -o "${OUT}" \
    -w '%{http_code}' \
    -X POST "${BASE}/v1/chat/stream" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d "${BODY}"
)"

echo "--- HTTP ${HTTP_CODE} SSE payload (truncated) ---"
head -c 2000 "${OUT}" || true
echo

if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "smoke_sse FAIL: HTTP ${HTTP_CODE}" >&2
  cat "${OUT}" >&2 || true
  exit 1
fi

# Shared contract validation (meta, sources, done, no error, order, public ids)
if command -v uv >/dev/null 2>&1; then
  SMOKE_SSE_BODY_PATH="${OUT}" uv run python -c "
import os
from pathlib import Path
from app.api.sse_util import validate_smoke_sse_body
body = Path(os.environ['SMOKE_SSE_BODY_PATH']).read_text(encoding='utf-8', errors='replace')
validate_smoke_sse_body(body)
print('smoke_sse PASS')
"
else
  # Fallback without uv: require meta, sources, done; reject error; no token-only PASS
  if grep -q 'event: error' "${OUT}"; then
    echo "smoke_sse FAIL: error event present" >&2
    exit 1
  fi
  if ! grep -q 'event: meta' "${OUT}"; then
    echo "smoke_sse FAIL: missing meta event" >&2
    exit 1
  fi
  if ! grep -q 'event: sources' "${OUT}"; then
    echo "smoke_sse FAIL: missing sources event" >&2
    exit 1
  fi
  if ! grep -q 'event: done' "${OUT}"; then
    echo "smoke_sse FAIL: missing done event" >&2
    exit 1
  fi
  echo "smoke_sse PASS"
fi
