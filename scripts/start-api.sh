#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime-env.sh"
cd "$BEV_CODE"
exec "$BEV_ROOT/.venv/bin/python" -m uvicorn bev.app:app \
  --host 127.0.0.1 --port "$BEV_PORT" --no-access-log
