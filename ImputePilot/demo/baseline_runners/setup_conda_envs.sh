#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash baseline_runners/setup_conda_envs.sh
#
# Optional env overrides:
#   FLAML_ENV_NAME, TUNE_ENV_NAME, RAHA_ENV_NAME, AUTOFOLIO_ENV_NAME, PYTHON_VERSION

FLAML_ENV_NAME="${FLAML_ENV_NAME:-ImputePilot-flaml}"
TUNE_ENV_NAME="${TUNE_ENV_NAME:-ImputePilot-tune}"
RAHA_ENV_NAME="${RAHA_ENV_NAME:-ImputePilot-raha}"
AUTOFOLIO_ENV_NAME="${AUTOFOLIO_ENV_NAME:-ImputePilot-autofolio}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"

echo "[setup] Creating conda envs with Python ${PYTHON_VERSION}..."
conda create -y -n "${FLAML_ENV_NAME}" "python=${PYTHON_VERSION}"
conda create -y -n "${TUNE_ENV_NAME}" "python=${PYTHON_VERSION}"
conda create -y -n "${RAHA_ENV_NAME}" "python=${PYTHON_VERSION}"
conda create -y -n "${AUTOFOLIO_ENV_NAME}" "python=${PYTHON_VERSION}"

echo "[setup] Installing FLAML runner dependencies..."
conda run -n "${FLAML_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${FLAML_ENV_NAME}" python -m pip install \
  numpy pandas scikit-learn joblib flaml lightgbm xgboost

echo "[setup] Installing Tune runner dependencies..."
conda run -n "${TUNE_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${TUNE_ENV_NAME}" python -m pip install \
  numpy pandas scikit-learn joblib "ray[tune]"

echo "[setup] Installing RAHA runner dependencies..."
conda run -n "${RAHA_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${RAHA_ENV_NAME}" python -m pip install \
  numpy scikit-learn joblib

echo "[setup] Installing AutoFolio runner dependencies..."
conda run -n "${AUTOFOLIO_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${AUTOFOLIO_ENV_NAME}" python -m pip install \
  numpy pandas scikit-learn joblib

echo "[setup] Done."
echo "[setup] Next: source baseline_runners/use_baseline_envs.sh"
