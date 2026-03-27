# ImputePilot

ImputePilot is a Django demo for stable imputation model selection in time series data repair.
It is built on top of ImputePilot/RecImpute-style components and provides an end-to-end workflow from training to recommendation and evaluation.

Reference paper:
- **ImputePilot: Stable Model Selection for Data Repair in Time Series**
- Local file: `ImputePilot_Stable_Model_Selection_for_Data_Repair.txt`

## What It Does

- Train recommendation model: upload -> clustering -> labeling -> features -> model race
- Recommend/impute on inference data
- Compare with baselines: FLAML / Tune / AutoFolio / RAHA
- Optional external DL labeling/imputation: BRITS, DeepMVI, MRNN, MPIN, IIM

## Project Entry

- Django root: `ImputePilot/demo`
- Run server: `cd ImputePilot/demo && python manage.py runserver`
- API base: `/api/`
- Main backend: `ImputePilot/demo/ImputePilot_api/views.py`

## Environment Setup (Important)

This project uses **multiple conda environments**.

### 1) Main app env (`ImputePilot`)

Used for Django app + core pipeline.

```bash
conda create -y -n ImputePilot python=3.10
conda activate ImputePilot
pip install -r requirements.txt
```

Also required for async tasks:
- Redis server (default: `redis://127.0.0.1:6379/0`)

### 2) Baseline runner envs (separate from `ImputePilot`)

Auto setup script (recommended):

```bash
bash ImputePilot/demo/baseline_runners/setup_conda_envs.sh
```

This creates and installs:
- `ImputePilot-flaml`: `numpy pandas scikit-learn joblib flaml lightgbm xgboost`
- `ImputePilot-tune`: `numpy pandas scikit-learn joblib ray[tune]`
- `ImputePilot-raha`: `numpy scikit-learn joblib`
- `ImputePilot-autofolio`: `numpy pandas scikit-learn joblib`

Before starting Django, export runner paths:

```bash
source ImputePilot/demo/baseline_runners/use_baseline_envs.sh
```

### 3) External DL env (`ImputePilot-labeldl311-mpin`)

Used by `ImputePilot/demo/labeling_runners/dl_benchmark_runner.py`.

Recommended base:

```bash
conda create -y -n ImputePilot-labeldl311-mpin python=3.11
conda activate ImputePilot-labeldl311-mpin
pip install numpy==1.26.4 h5py pygrinder toml
```

Install PyTorch (choose one):

CPU:
```bash
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1:
```bash
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Install DL packages:

```bash
pip install pypots==1.1 --no-deps
pip install imputegap==1.1.1 --no-deps
```

Extra requirements by algorithm:
- MRNN needs: `tensorflow==2.16.2`
- MPIN needs: `torch_geometric` stack (install in this env)

## Quick Start

```bash
conda activate ImputePilot
cd ImputePilot/demo
python manage.py migrate
python manage.py runserver
```

Open:
- `http://127.0.0.1:8000/api/`

Optional Celery worker:

```bash
cd ImputePilot/demo
celery -A demo worker -l info
```

## API Workflow (Typical)

1. `POST /api/pipeline/upload/`
2. `POST /api/pipeline/clustering/`
3. `POST /api/pipeline/labeling/`
4. `POST /api/pipeline/features/`
5. `POST /api/pipeline/modelrace/`
6. `POST /api/recommend/upload/`
7. `POST /api/recommend/features/`
8. `POST /api/recommend/recommend/`
9. Optional: `POST /api/recommend/impute/`, `POST /api/recommend/compare/`, `POST /api/recommend/full_evaluation/`

## Notes

- Baseline and DL runners are executed via external Python interpreters configured in `ImputePilot/demo/demo/settings.py`.
- If baseline/DL envs are missing, core `ImputePilot` pipeline can still run, but related endpoints will fail.

## Maintainer

- Author: `zhexinjin@zju.edu.cn`
- Affiliation: Zhejiang University

## Attribution

This demo builds on prior ImputePilot/RecImpute-related research and implementations.

