#!/usr/bin/env sh
set -eu

: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"
: "${VERBOSE_LOGS:=false}"

VERBOSE_LOGS="$(printf "%s" "${VERBOSE_LOGS}" | tr '[:upper:]' '[:lower:]')"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VERBOSE_LOGS

VERBOSE_LOGS_VALUE="${VERBOSE_LOGS}"
LOG_LEVEL="error"

case "${VERBOSE_LOGS_VALUE}" in
  true|1|yes|y|on)
    LOG_LEVEL="debug"
    ;;
esac

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level "${LOG_LEVEL}"
