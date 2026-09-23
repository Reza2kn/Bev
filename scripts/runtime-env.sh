#!/usr/bin/env bash
# Shared paths and settings. Source this file from the runtime scripts.
if [ -n "${BEV_ROOT:-}" ] && [ -n "${BEV_HOME:-}" ] && [ "$BEV_ROOT" != "$BEV_HOME" ]; then
  printf 'BEV_ROOT and legacy BEV_HOME disagree; set only one or make them equal.\n' >&2
  return 1
fi
BEV_ROOT="${BEV_ROOT:-${BEV_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/bev}}"
BEV_HOME="$BEV_ROOT"
BEV_CODE="${BEV_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
BEV_LLAMA_PORT="${BEV_LLAMA_PORT:-18780}"
BEV_PORT="${BEV_PORT:-18781}"
BEV_MODEL="${BEV_MODEL:-bev-bonsai-27b}"
BEV_LLAMA_URL="${BEV_LLAMA_URL:-http://127.0.0.1:$BEV_LLAMA_PORT}"
export BEV_ROOT BEV_HOME BEV_CODE BEV_LLAMA_PORT BEV_PORT BEV_MODEL BEV_LLAMA_URL

bev_process_matches() {
  local service="$1" pid="$2" executable args
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  IFS= read -r -d '' executable <"/proc/$pid/cmdline" || return 1
  args=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  case "$service" in
    api) [ "$executable" = "$BEV_ROOT/.venv/bin/python" ] &&
      [[ "$args" == "$BEV_ROOT/.venv/bin/python -m uvicorn bev.app:app "* ]] ;;
    llama) [ "$executable" = "$BEV_ROOT/prism-build/bin/llama-server" ] ;;
    *) return 1 ;;
  esac
}
