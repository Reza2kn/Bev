#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime-env.sh"
if [ "$BEV_LLAMA_URL" != "http://127.0.0.1:$BEV_LLAMA_PORT" ]; then
  printf 'start-services.sh manages a local backend. For an external BEV_LLAMA_URL, use start-api.sh.\n' >&2
  exit 1
fi
mkdir -p "$BEV_ROOT/logs"

start_service() {
  local service="$1" script="$2" url="$3" logfile="$4" pidfile pid='' attempt
  pidfile="$BEV_ROOT/logs/$service.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    [[ "$pid" =~ ^[0-9]+$ ]] || { printf 'Invalid PID file: %s\n' "$pidfile" >&2; return 1; }
    if kill -0 "$pid" 2>/dev/null; then
      bev_process_matches "$service" "$pid" || { printf 'Refusing unexpected live process in %s\n' "$pidfile" >&2; return 1; }
    else
      pid=''
    fi
  fi
  if [ -z "$pid" ]; then
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      printf 'A service already answers %s but is not owned by this BEV_ROOT.\n' "$url" >&2
      return 1
    fi
    nohup bash "$BEV_CODE/scripts/$script" >"$logfile" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$pidfile"
  fi
  for ((attempt=0; attempt<${BEV_START_TIMEOUT_SECONDS:-180}; attempt++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'Bev %s exited before readiness. Inspect %s\n' "$service" "$logfile" >&2
      return 1
    fi
    sleep 1
  done
  printf 'Bev %s did not become healthy. Inspect %s\n' "$service" "$logfile" >&2
  return 1
}

start_service llama start-llama.sh "$BEV_LLAMA_URL/health" "$BEV_ROOT/logs/llama-patched.log"
start_service api start-api.sh "http://127.0.0.1:$BEV_PORT/health" "$BEV_ROOT/logs/api.log"
curl -fsS "http://127.0.0.1:$BEV_PORT/health"
printf '\n'
