#!/usr/bin/env sh
set -eu

VERBOSE_LOGS_VALUE="$(printf "%s" "${VERBOSE_LOGS:-false}" | tr '[:upper:]' '[:lower:]')"
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

