#!/usr/bin/env bash
# Smoke-test POST /v1/chat/stream SSE endpoint.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
export MOCK_LLM="${MOCK_LLM:-1}"

BODY='{"message":"What is the refund policy?","language":"en"}'
OUT="$(mktemp)"
curl -sS -N -X POST "${BASE}/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "${BODY}" > "${OUT}" || true

echo "--- SSE payload (truncated) ---"
head -c 2000 "${OUT}" || true
echo

if grep -q "event: done" "${OUT}" || grep -q "finish_reason" "${OUT}"; then
  echo "smoke_sse PASS"
  exit 0
fi
if grep -q "event: token" "${OUT}"; then
  echo "smoke_sse PASS (tokens without done — still ok)"
  exit 0
fi
echo "smoke_sse FAIL: no token/done events" >&2
cat "${OUT}" >&2 || true
exit 1
