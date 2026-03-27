#!/usr/bin/env bash
set -euo pipefail

APP="${APP:-demo}"
CELERY_BIN="${CELERY_BIN:-celery}"
QUEUES=(${CELERY_QUEUES:-celery dl_gpu})

# Gracefully ask all workers to shutdown.
${CELERY_BIN} -A "${APP}" control shutdown || true

# Fallback: ensure any remaining workers are terminated.
if pgrep -f "celery -A ${APP} worker" >/dev/null 2>&1; then
  pkill -f "celery -A ${APP} worker" || true
fi

# Purge all queued tasks for the configured queues.
for q in "${QUEUES[@]}"; do
  ${CELERY_BIN} -A "${APP}" purge -f -Q "${q}" || true
done
