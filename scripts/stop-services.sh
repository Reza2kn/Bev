#!/usr/bin/env bash
# Stop only executables belonging to this configured Bev installation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime-env.sh"
for service in api llama; do
  pidfile="$BEV_ROOT/logs/$service.pid"
  [ -f "$pidfile" ] || continue
  pid=$(cat "$pidfile")
  [[ "$pid" =~ ^[0-9]+$ ]] || { printf 'Invalid PID file %s\n' "$pidfile" >&2; exit 1; }
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$pidfile"; continue; fi
  if ! bev_process_matches "$service" "$pid"; then
    printf 'Refusing to stop unexpected process in %s\n' "$pidfile" >&2
    exit 1
  fi
  kill "$pid"
  for attempt in $(seq 1 150); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    state=$(ps -o stat= -p "$pid" || true)
    [[ "$state" == Z* ]] && break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null && [[ "$(ps -o stat= -p "$pid" || true)" != Z* ]]; then
    printf 'Timed out waiting for Bev %s (PID %s) to stop.\n' "$service" "$pid" >&2
    exit 1
  fi
  rm -f "$pidfile"
done
