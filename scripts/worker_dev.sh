#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${QUANT_PLATFORM_REDIS_URL:=redis://localhost:6379/0}"
export QUANT_PLATFORM_REDIS_URL

exec python -m quant_platform.jobs.workers "$@"
