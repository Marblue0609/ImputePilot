#!/usr/bin/env bash
# Source this file before starting Django:
#   source ImputePilot/demo/baseline_runners/use_baseline_envs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
FLAML_ENV_NAME="${FLAML_ENV_NAME:-ImputePilot-flaml}"
TUNE_ENV_NAME="${TUNE_ENV_NAME:-ImputePilot-tune}"
RAHA_ENV_NAME="${RAHA_ENV_NAME:-ImputePilot-raha}"
AUTOFOLIO_ENV_NAME="${AUTOFOLIO_ENV_NAME:-ImputePilot-autofolio}"

export BASELINE_RUNNERS_DIR="${SCRIPT_DIR}"
export FLAML_VENV_PY="${FLAML_VENV_PY:-${CONDA_HOME}/envs/${FLAML_ENV_NAME}/bin/python}"
export TUNE_VENV_PY="${TUNE_VENV_PY:-${CONDA_HOME}/envs/${TUNE_ENV_NAME}/bin/python}"
export RAHA_VENV_PY="${RAHA_VENV_PY:-${CONDA_HOME}/envs/${RAHA_ENV_NAME}/bin/python}"
export AUTOFOLIO_VENV_PY="${AUTOFOLIO_VENV_PY:-${CONDA_HOME}/envs/${AUTOFOLIO_ENV_NAME}/bin/python}"

echo "BASELINE_RUNNERS_DIR=${BASELINE_RUNNERS_DIR}"
echo "FLAML_VENV_PY=${FLAML_VENV_PY}"
echo "TUNE_VENV_PY=${TUNE_VENV_PY}"
echo "RAHA_VENV_PY=${RAHA_VENV_PY}"
echo "AUTOFOLIO_VENV_PY=${AUTOFOLIO_VENV_PY}"
echo "Start Django from: ${DEMO_DIR}"
