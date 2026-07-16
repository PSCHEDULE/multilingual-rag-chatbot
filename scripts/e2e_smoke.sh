#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== health =="
curl -sf "${BASE}/health" | grep -q '"status":"ok\|"status": "ok"' || curl -sf "${BASE}/health"

echo "== openapi chat path =="
curl -sf "${BASE}/openapi.json" | grep -q 'chat'

echo "== widget assets =="
test -f widget/chatbot-widget.js
test -f widget/demo.html

echo "== sse smoke =="
export MOCK_LLM="${MOCK_LLM:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/smoke_sse.sh" "${BASE}"

echo "e2e_smoke PASS"
