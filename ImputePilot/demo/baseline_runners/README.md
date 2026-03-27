# Baseline Runners (Multi-Env Isolation)

This folder contains external runner scripts used by Django `views.py`:

- `flaml_runner.py`
- `tune_runner.py`
- `raha_runner.py`
- `autofolio_runner.py`

The runners follow this contract:

- args: `--mode train|predict --input <npz> --meta <json> --output <json>`
- output JSON includes `status` and task-specific fields
- trained model artifacts are saved in `baseline_runners/.state/`

## 1) Create dedicated conda environments

From project root:

```bash
bash ImputePilot/demo/baseline_runners/setup_conda_envs.sh
```

Default env names:

- `ImputePilot-flaml`
- `ImputePilot-tune`
- `ImputePilot-raha`
- `ImputePilot-autofolio`

You can override names with env vars before running setup:

```bash
export FLAML_ENV_NAME=my-flaml
export TUNE_ENV_NAME=my-tune
export RAHA_ENV_NAME=my-raha
export AUTOFOLIO_ENV_NAME=my-autofolio
bash ImputePilot/demo/baseline_runners/setup_conda_envs.sh
```

## 2) Export runner env vars for Django process

Before starting Django (in your main app env, e.g. `ImputePilot`):

```bash
source ImputePilot/demo/baseline_runners/use_baseline_envs.sh
```

This exports:

- `BASELINE_RUNNERS_DIR`
- `FLAML_VENV_PY`
- `TUNE_VENV_PY`
- `RAHA_VENV_PY`
- `AUTOFOLIO_VENV_PY`

## 3) Start Django in main environment

```bash
conda activate ImputePilot
source ImputePilot/demo/baseline_runners/use_baseline_envs.sh
cd ImputePilot/demo
python manage.py runserver
```

Once started, baseline training endpoints will automatically use external runners when env vars and scripts are detected.
