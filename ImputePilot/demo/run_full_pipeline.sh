#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18000/api}"
LOG_DIR="${LOG_DIR:-/home/yyy/ImputePilot_proj/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/full_pipeline_${TIMESTAMP}.log}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

post_json() {
  local endpoint="$1"
  local body="${2:-{}}"
  curl -sS -X POST "${BASE_URL}${endpoint}" \
    -H "Content-Type: application/json" \
    -d "${body}"
}

check_json_ok() {
  python - "$1" <<'PY'
import json, sys
payload = sys.argv[1]
try:
    data = json.loads(payload)
except Exception:
    print("ERROR: Non-JSON response:", payload[:200])
    sys.exit(1)
if isinstance(data, dict) and data.get("error"):
    print("ERROR:", data.get("error"))
    sys.exit(1)
print("OK")
PY
}

extract_best_algo() {
  python - "$1" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
algo = (data.get("ImputePilot") or {}).get("algo") or ""
print(algo)
PY
}

upload_if_present() {
  local endpoint="$1"
  local file_path="$2"
  if [[ -n "${file_path}" ]]; then
    if [[ ! -f "${file_path}" ]]; then
      log "ERROR: File not found: ${file_path}"
      exit 1
    fi
    log "Uploading ${file_path} to ${endpoint}"
    curl -sS -X POST "${BASE_URL}${endpoint}" -F "files=@${file_path}" >/dev/null
  fi
}

log "Starting full pipeline run"
log "BASE_URL=${BASE_URL}"
log "LOG_FILE=${LOG_FILE}"

# Optional uploads
upload_if_present "/pipeline/upload/" "${TRAIN_ZIP:-}"
upload_if_present "/recommend/upload/" "${INFER_ZIP:-}"

log "Step 1: Clustering"
resp="$(post_json "/pipeline/clustering/" "{}")"
check_json_ok "$resp"

log "Step 2: Labeling"
resp="$(post_json "/pipeline/labeling/" "{}")"
check_json_ok "$resp"

log "Step 3: Feature Extraction"
resp="$(post_json "/pipeline/features/" "{}")"
check_json_ok "$resp"

log "Step 4: ModelRace"
resp="$(post_json "/pipeline/modelrace/" '{"alpha":0.5,"beta":0.5,"gamma":0.5,"seedPipelines":100,"pValue":0.01}')"
check_json_ok "$resp"

log "Step 5: Recommendation"
resp="$(post_json "/recommend/recommend/" '{"datasetId":"auto"}')"
check_json_ok "$resp"
best_algo="$(extract_best_algo "$resp")"
if [[ -z "${best_algo}" ]]; then
  log "ERROR: No recommended algorithm found in response."
  exit 1
fi
log "Recommended algorithm: ${best_algo}"

log "Step 6: Imputation (${best_algo})"
resp="$(post_json "/recommend/impute/" "{\"algorithm\":\"${best_algo}\"}")"
check_json_ok "$resp"

log "Step 7: Downstream Evaluation (forecasting)"
resp="$(post_json "/recommend/downstream/" '{"task":"forecasting","algorithm":null,"eval_missing_rate":0.1,"eval_seed":42,"regenerate_mask":false}')"
check_json_ok "$resp"

log "Step 8: Downstream Evaluation (classification)"
resp="$(post_json "/recommend/downstream/" '{"task":"classification","algorithm":null,"eval_missing_rate":0.1,"eval_seed":42,"regenerate_mask":false}')"
check_json_ok "$resp"

log "Pipeline completed successfully"
