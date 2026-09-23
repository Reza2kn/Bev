#!/usr/bin/env bash
# Compatibility name; installation now works on supported Linux NVIDIA hosts.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/install.sh" "$@"
