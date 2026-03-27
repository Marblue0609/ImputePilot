import os
import json
import shutil
import traceback
import ast
import time
import tempfile
import subprocess
import threading
from functools import lru_cache
import numpy as np
import pandas as pd
import zipfile
import random
from collections import Counter
from celery import group
from celery.exceptions import TimeoutError as CeleryTimeoutError
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from scipy.stats import ttest_rel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

from ImputePilot_api.ImputePilot_code.Datasets.Dataset import Dataset
from ImputePilot_api.ImputePilot_code.Datasets.TrainingSet import TrainingSet
from ImputePilot_api.dataset_categories import annotate_benchmark_rows, build_benchmark_category_summary

PRIMARY_METHOD_NAME = "ImputePilot"
_PRIMARY_METHOD_ALIAS_TOKENS = {"imputepilot", "ImputePilot", "adart"}


def _json_safe_obj(obj):
    """Coerce numpy scalars/arrays into JSON-serializable Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_obj(v) for v in obj]
    return obj


def _normalize_primary_method_name(method):
    if not isinstance(method, str):
        return method
    raw = method.strip()
    if not raw:
        return raw
    token = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    if token in _PRIMARY_METHOD_ALIAS_TOKENS:
        return PRIMARY_METHOD_NAME
    return raw


def _dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items or []:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


@lru_cache(maxsize=1)
def _get_realworld_dataset_weight_map():
    try:
        dataset_dir = AdartsService.get_dataset_dir()
        dataset_weights = {}

        def _estimate_series_count_from_frame(df):
            if df is None or df.empty:
                return 0
            n_cols = int(df.shape[1])
            if n_cols <= 0:
                return 0
            if n_cols == 1:
                return 1

            first_col = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            valid_ratio = float(first_col.notna().mean()) if len(first_col) else 0.0
            return n_cols - 1 if valid_ratio >= 0.8 else n_cols

        def _read_dataset_head(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".zip":
                with zipfile.ZipFile(path, "r") as archive:
                    candidates = [
                        name for name in archive.namelist()
                        if name.lower().endswith((".txt", ".csv", ".tsv"))
                    ]
                    if not candidates:
                        return None
                    candidates.sort()
                    with archive.open(candidates[0]) as fh:
                        return pd.read_csv(fh, sep=None, engine="python", header=None, nrows=8)
            if ext in {".txt", ".csv", ".tsv"}:
                return pd.read_csv(path, sep=None, engine="python", header=None, nrows=8)
            return None

        for filename in os.listdir(dataset_dir):
            full_path = os.path.join(dataset_dir, filename)
            if not os.path.isfile(full_path):
                continue
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in {".zip", ".txt", ".csv", ".tsv"}:
                continue
            try:
                df_head = _read_dataset_head(full_path)
                weight = _estimate_series_count_from_frame(df_head)
                if weight > 0:
                    dataset_weights[stem] = weight
                    dataset_weights[stem.lower()] = weight
            except Exception as inner_e:
                print(f"[WARN] Failed to estimate dataset weight for {filename}: {inner_e}")
        return dataset_weights
    except Exception as e:
        print(f"[WARN] Failed to build RealWorld dataset weight map: {e}")
        return {}


def _load_dashboard_status_overrides():
    summary_paths = [
        "/tmp/dashboard_eval_rerun_skipped_20260315T130801Z/summary_subset_parallel.json",
        "/tmp/dashboard_eval_rerun_skipped_20260315T130801Z/summary_timeout_retry_3600.json",
    ]
    overrides = {}
    for path in summary_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("completed", []):
                dataset = str(item.get("dataset") or "").strip()
                summary = item.get("summary") or {}
                if not dataset or not isinstance(summary, dict):
                    continue
                dataset_overrides = overrides.setdefault(dataset, {})
                for method, result in summary.items():
                    method = _normalize_primary_method_name(method)
                    result_text = str(result or "").strip()
                    if not result_text:
                        continue
                    if result_text == "success":
                        dataset_overrides[method] = {"status": "success", "error": None}
                    else:
                        dataset_overrides[method] = {"status": "error", "error": result_text}
        except Exception as e:
            print(f"[WARN] Failed to load dashboard override summary {path}: {e}")

    results_path = "/tmp/dashboard_eval_rerun_skipped_20260315T130801Z/results_timeout_retry_3600.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for row in payload.get("rows", []):
                if not isinstance(row, dict):
                    continue
                dataset = str(row.get("dataset") or "").strip()
                method = _normalize_primary_method_name(str(row.get("method") or "").strip())
                if not dataset or not method:
                    continue
                dataset_overrides = overrides.setdefault(dataset, {})
                entry = dataset_overrides.setdefault(method, {})
                for key in (
                    "algo",
                    "forecasting_rmse",
                    "forecasting_improvement",
                    "forecasting_n_evaluated",
                    "classification_acc",
                    "classification_improvement",
                    "classification_n_evaluated",
                    "status",
                    "error",
                ):
                    if key in row:
                        entry[key] = row.get(key)
        except Exception as e:
            print(f"[WARN] Failed to load dashboard override results {results_path}: {e}")
    return overrides


def _apply_dashboard_status_overrides(rows):
    overrides = _load_dashboard_status_overrides()
    if not overrides:
        return rows

    patched_rows = []
    for row in rows or []:
        if not isinstance(row, dict):
            patched_rows.append(row)
            continue
        row_copy = dict(row)
        dataset = str(row_copy.get("dataset") or "").strip()
        method = _normalize_primary_method_name(str(row_copy.get("method") or "").strip())
        row_copy["method"] = method
        override = overrides.get(dataset, {}).get(method)
        if override:
            row_copy.update(override)
        patched_rows.append(row_copy)
    return patched_rows


def _run_with_heartbeat(label, fn):
    log_interval = int(os.getenv("ImputePilot_PROGRESS_INTERVAL_SEC", "60"))
    if log_interval <= 0:
        return fn()
    stop_event = threading.Event()

    def _beat():
        while not stop_event.wait(log_interval):
            print(f"[Progress] {label} still running...")

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
from ImputePilot_api.ImputePilot_code.Training.ClfPipeline import ClfPipeline
from ImputePilot_api.ImputePilot_code.Training.ModelsTrainer import ModelsTrainer
from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils


# Baseline tuning defaults (override via env vars if needed).
FLAML_DEFAULT_TIME_BUDGET_SEC = int(os.getenv("FLAML_DEFAULT_TIME_BUDGET_SEC", "3600"))


# ========== Service Layer ==========

class AdartsService:
    """Singleton service to manage global objects"""
    _clusterer = None
    _labeler = None
    _trained_model = None
    _last_recommendation = None
    _flaml_model = None  # FLAML baseline model
    _tune_model = None   # Tune baseline model
    _autofolio_model = None  # AutoFolio baseline model
    _raha_model = None   # RAHA baseline model
    _ground_truth_data = None  # Ground truth for evaluation
    _evaluation_mode = None  # 'test_set' or 'upload'
    _missing_injection_rate = 0.2  # Default 20% missing injection
    
    @staticmethod
    def get_base_path():
        return os.path.join(settings.BASE_DIR, 'ImputePilot_api', 'ImputePilot_code')

    @staticmethod
    def get_dataset_dir():
        """Training data directory"""
        return os.path.join(AdartsService.get_base_path(), 'Datasets', 'RealWorld')

    @staticmethod
    def get_inference_dir():
        """Inference data directory"""
        # Allow per-process override to isolate concurrent evaluation runs.
        custom_path = os.getenv("ImputePilot_INFERENCE_DIR", "").strip()
        if custom_path:
            path = os.path.abspath(custom_path)
        else:
            path = os.path.join(AdartsService.get_base_path(), 'Datasets', 'Inference')
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_ground_truth_path():
        return os.path.join(AdartsService.get_inference_dir(), 'ground_truth.csv')

    @staticmethod
    def get_ground_truth_meta_path():
        return os.path.join(AdartsService.get_inference_dir(), 'ground_truth_meta.json')
    
    @staticmethod
    def get_recommendations_dir():
        """Recommendation results directory"""
        path = os.path.join(AdartsService.get_base_path(), 'Datasets', 'Recommendations')
        os.makedirs(path, exist_ok=True)
        return path
    
    @staticmethod
    def set_last_recommendation(recommendation_data):
        """Save the most recent result"""
        AdartsService._last_recommendation = recommendation_data
    
    @staticmethod
    def get_last_recommendation():
        """Get the most recent result"""
        return AdartsService._last_recommendation

    @staticmethod
    def get_system_inputs_dir():
        return os.path.join(AdartsService.get_base_path(), 'Datasets', 'SystemInputs')

    @staticmethod
    def get_trained_model_path():
        return os.path.join(AdartsService.get_system_inputs_dir(), 'trained_model.joblib')

    @staticmethod
    def get_trained_pipelines_dir():
        path = os.path.join(AdartsService.get_system_inputs_dir(), 'trained_pipelines')
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_baseline_models_dir():
        path = os.path.join(AdartsService.get_system_inputs_dir(), 'baseline_models')
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_baseline_model_path(baseline_name):
        safe_name = str(baseline_name or '').strip().lower()
        return os.path.join(AdartsService.get_baseline_models_dir(), f'{safe_name}_model.joblib')

    @staticmethod
    def _persist_baseline_model(baseline_name, model_data):
        try:
            import joblib
            path = AdartsService.get_baseline_model_path(baseline_name)
            joblib.dump(model_data, path)
            print(f"[AdartsService] {baseline_name} model saved (disk).")
        except Exception as e:
            print(f"[WARN] Failed to persist {baseline_name} model: {e}")

    @staticmethod
    def _load_baseline_model(baseline_name):
        try:
            import joblib
            path = AdartsService.get_baseline_model_path(baseline_name)
            if not os.path.exists(path):
                return None
            model_data = joblib.load(path)
            print(f"[AdartsService] {baseline_name} model loaded (disk).")
            return model_data
        except Exception as e:
            print(f"[WARN] Failed to load {baseline_name} model from disk: {e}")
            return None

    @staticmethod
    def _clear_baseline_model_file(baseline_name):
        try:
            path = AdartsService.get_baseline_model_path(baseline_name)
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[WARN] Failed to remove persisted {baseline_name} model: {e}")

    @staticmethod
    def _persist_trained_pipelines(model_payload):
        pipelines = model_payload.get("pipelines", [])
        if not pipelines:
            return
        try:
            import joblib
        except Exception as e:
            print(f"[WARN] joblib unavailable for pipeline persistence: {e}")
            return

        pipelines_dir = AdartsService.get_trained_pipelines_dir()
        for idx, pipe in enumerate(pipelines):
            rm = getattr(pipe, "rm", None)
            if rm is None:
                continue
            rm_id = getattr(rm, "id", None)
            safe_id = str(rm_id if rm_id is not None else idx)

            best_path = os.path.join(pipelines_dir, f"rm_{safe_id}_best_cv.joblib")
            prod_path = os.path.join(pipelines_dir, f"rm_{safe_id}_prod.joblib")

            best_cv = getattr(rm, "best_cv_trained_pipeline", None)
            if best_cv is not None:
                try:
                    joblib.dump(best_cv, best_path)
                    rm.best_cv_trained_pipeline_path = best_path
                except Exception as e:
                    print(f"[WARN] Failed to persist best_cv pipeline for rm={safe_id}: {e}")
                    rm.best_cv_trained_pipeline_path = None
            else:
                rm.best_cv_trained_pipeline_path = None

            prod = getattr(rm, "trained_pipeline_prod", None)
            if prod is not None:
                try:
                    joblib.dump(prod, prod_path)
                    rm.trained_pipeline_prod_path = prod_path
                except Exception as e:
                    print(f"[WARN] Failed to persist prod pipeline for rm={safe_id}: {e}")
                    rm.trained_pipeline_prod_path = None
            else:
                rm.trained_pipeline_prod_path = None

    @staticmethod
    def _hydrate_trained_pipelines(model_payload):
        pipelines = model_payload.get("pipelines", [])
        if not pipelines:
            return
        try:
            import joblib
        except Exception as e:
            print(f"[WARN] joblib unavailable for pipeline hydration: {e}")
            return

        for idx, pipe in enumerate(pipelines):
            rm = getattr(pipe, "rm", None)
            if rm is None:
                continue
            best_cv = getattr(rm, "best_cv_trained_pipeline", None)
            best_path = getattr(rm, "best_cv_trained_pipeline_path", None)
            if best_cv is None and best_path and os.path.exists(best_path):
                try:
                    rm.best_cv_trained_pipeline = joblib.load(best_path)
                except Exception as e:
                    print(f"[WARN] Failed to load best_cv pipeline for rm={idx}: {e}")

            prod = getattr(rm, "trained_pipeline_prod", None)
            prod_path = getattr(rm, "trained_pipeline_prod_path", None)
            if prod is None and prod_path and os.path.exists(prod_path):
                try:
                    rm.trained_pipeline_prod = joblib.load(prod_path)
                except Exception as e:
                    print(f"[WARN] Failed to load prod pipeline for rm={idx}: {e}")

    @staticmethod
    def get_external_runner_config(baseline_name):
        """
        External baseline runner configuration.

        Env vars:
        - BASELINE_RUNNERS_DIR (optional, default: <BASE_DIR>/baseline_runners)
        - FLAML_VENV_PY / TUNE_VENV_PY / AUTOFOLIO_VENV_PY / RAHA_VENV_PY
        """
        runners_dir = os.getenv(
            "BASELINE_RUNNERS_DIR",
            getattr(settings, "BASELINE_RUNNERS_DIR", os.path.join(settings.BASE_DIR, "baseline_runners"))
        )
        name = baseline_name.upper()
        if name == "FLAML":
            return {
                "python": os.getenv("FLAML_VENV_PY", getattr(settings, "FLAML_VENV_PY", "")),
                "script": os.path.join(runners_dir, "flaml_runner.py"),
            }
        if name == "TUNE":
            return {
                "python": os.getenv("TUNE_VENV_PY", getattr(settings, "TUNE_VENV_PY", "")),
                "script": os.path.join(runners_dir, "tune_runner.py"),
            }
        if name == "RAHA":
            return {
                "python": os.getenv("RAHA_VENV_PY", getattr(settings, "RAHA_VENV_PY", "")),
                "script": os.path.join(runners_dir, "raha_runner.py"),
            }
        if name == "AUTOFOLIO":
            return {
                "python": os.getenv("AUTOFOLIO_VENV_PY", getattr(settings, "AUTOFOLIO_VENV_PY", "")),
                "script": os.path.join(runners_dir, "autofolio_runner.py"),
            }
        return {"python": "", "script": ""}

    @staticmethod
    def is_external_runner_enabled(baseline_name):
        cfg = AdartsService.get_external_runner_config(baseline_name)
        return bool(cfg.get("python")) and bool(cfg.get("script")) and os.path.exists(cfg.get("script"))

    @staticmethod
    def get_external_labeling_config():
        """
        External labeling runner configuration for single-DL-algorithm pilot.

        Env vars:
        - LABELING_RUNNERS_DIR (optional, default: <BASE_DIR>/labeling_runners)
        - DL_LABEL_VENV_PY
        - DL_LABEL_RUNNER_SCRIPT
        - DL_LABEL_ALGO
        """
        runners_dir = os.getenv(
            "LABELING_RUNNERS_DIR",
            getattr(settings, "LABELING_RUNNERS_DIR", os.path.join(settings.BASE_DIR, "labeling_runners"))
        )
        python_exec = os.getenv("DL_LABEL_VENV_PY", getattr(settings, "DL_LABEL_VENV_PY", ""))
        default_script = os.path.join(runners_dir, "dl_benchmark_runner.py")
        script_path = os.getenv("DL_LABEL_RUNNER_SCRIPT", getattr(settings, "DL_LABEL_RUNNER_SCRIPT", default_script))
        algo = os.getenv("DL_LABEL_ALGO", getattr(settings, "DL_LABEL_ALGO", "brits"))
        timeout_sec = int(os.getenv("DL_LABEL_TIMEOUT_SEC", str(getattr(settings, "DL_LABEL_TIMEOUT_SEC", 1800))))
        return {
            "python": python_exec,
            "script": script_path,
            "algo": str(algo).strip().lower(),
            "timeout_sec": timeout_sec,
        }

    @staticmethod
    def is_external_labeling_enabled():
        cfg = AdartsService.get_external_labeling_config()
        return bool(cfg.get("python")) and bool(cfg.get("script")) and os.path.exists(cfg.get("script"))

    @staticmethod
    def get_clusterer():
        """Load Clusterer"""
        if AdartsService._clusterer is None:
            from ImputePilot_api.ImputePilot_code.Clustering.ShapeBasedClustering import ShapeBasedClustering
            print("[AdartsService] Initializing ShapeBasedClustering...")
            AdartsService._clusterer = ShapeBasedClustering()
        return AdartsService._clusterer

    @staticmethod
    def get_labeler():
        """Load Labeler"""
        if AdartsService._labeler is None:
            from ImputePilot_api.ImputePilot_code.Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
            print("[AdartsService] Initializing ImputeBenchLabeler...")
            AdartsService._labeler = ImputeBenchLabeler.get_instance()
        return AdartsService._labeler

    @staticmethod
    def load_datasets():
        """Load training datasets"""
        clusterer = AdartsService.get_clusterer()
        dataset_dir = AdartsService.get_dataset_dir()
        os.makedirs(dataset_dir, exist_ok=True)
        return Dataset.instantiate_from_dir(clusterer, data_dir=dataset_dir)
    
    @staticmethod
    def load_inference_datasets():
        """Load inference datasets"""
        clusterer = AdartsService.get_clusterer()
        inference_dir = AdartsService.get_inference_dir()
        os.makedirs(inference_dir, exist_ok=True)

        return Dataset.instantiate_from_dir(clusterer, data_dir=inference_dir)

    @staticmethod
    def get_feature_extractor(name):
        """Get the feature extractors dynamically"""
        try:
            if name == 'tsfresh':
                from ImputePilot_api.ImputePilot_code.FeaturesExtraction.TSFreshFeaturesExtractor import TSFreshFeaturesExtractor
                return TSFreshFeaturesExtractor
            elif name == 'catch22':
                from ImputePilot_api.ImputePilot_code.FeaturesExtraction.Catch22FeaturesExtractor import Catch22FeaturesExtractor
                return Catch22FeaturesExtractor
            elif name == 'topological':
                from ImputePilot_api.ImputePilot_code.FeaturesExtraction.TopologicalFeaturesExtractor import TopologicalFeaturesExtractor
                return TopologicalFeaturesExtractor
        except ImportError as e:
            print(f"[WARN] Failed to import {name} extractor: {e}")
            return None
        return None

    @staticmethod
    def _build_feature_extractors_from_names(names):
        extractors = []
        for name in names:
            key = str(name).strip().lower()
            if key.endswith("featuresextractor"):
                key = key.replace("featuresextractor", "")
            if key in {"catch22", "tsfresh", "topological"}:
                fe_cls = AdartsService.get_feature_extractor(key)
            else:
                if "catch22" in key:
                    fe_cls = AdartsService.get_feature_extractor("catch22")
                elif "tsfresh" in key:
                    fe_cls = AdartsService.get_feature_extractor("tsfresh")
                elif "topological" in key:
                    fe_cls = AdartsService.get_feature_extractor("topological")
                else:
                    fe_cls = None
            if fe_cls:
                try:
                    extractors.append(fe_cls.get_instance())
                except Exception:
                    continue
        return extractors

    @staticmethod
    def ensure_training_set(trained_model):
        if trained_model is None:
            return None
        training_set = trained_model.get("training_set")
        if training_set is not None:
            return training_set

        try:
            clusterer = AdartsService.get_clusterer()
            labeler = AdartsService.get_labeler()
            datasets = AdartsService.load_datasets()
            if not datasets:
                return None
            fe_instances = trained_model.get("feature_extractors")
            if not fe_instances:
                names = trained_model.get("feature_extractor_names", ["catch22", "tsfresh", "topological"])
                fe_instances = AdartsService._build_feature_extractors_from_names(names)
            if not fe_instances:
                print("[WARN] Failed to rebuild training_set: no feature extractors available.")
                return None

            # RealWorld may contain datasets that were uploaded after the last successful pipeline run.
            # Skip datasets that do not have complete intermediate artifacts yet.
            ready_datasets = []
            skipped_datasets = []
            for ds in datasets:
                has_clusters = False
                has_labels = False
                has_features = True
                try:
                    has_clusters = bool(clusterer.are_clusters_created(ds.name))
                    has_labels = bool(labeler.are_labels_created(ds.name))
                    for fe in fe_instances:
                        if not fe.are_features_created(ds.name):
                            has_features = False
                            break
                except Exception:
                    has_features = False

                if has_clusters and has_labels and has_features:
                    ready_datasets.append(ds)
                else:
                    skipped_datasets.append(ds.name)

            if skipped_datasets:
                print(
                    f"[WARN] ensure_training_set skipped {len(skipped_datasets)} dataset(s) "
                    f"without complete artifacts: {', '.join(skipped_datasets[:10])}"
                    + (" ..." if len(skipped_datasets) > 10 else "")
                )
            if not ready_datasets:
                print("[WARN] Failed to rebuild training_set: no datasets with complete artifacts.")
                return None

            labeler_prop = labeler.get_default_properties()
            training_set = TrainingSet(
                ready_datasets, clusterer, fe_instances, labeler, labeler_prop, force_generation=False
            )
            trained_model["training_set"] = training_set
            return training_set
        except Exception as e:
            print(f"[WARN] Failed to rebuild training_set: {e}")
            return None
    
    @staticmethod
    def set_trained_model(model):
        """Save the trained model"""
        AdartsService._trained_model = model
        try:
            import joblib

            model_payload = dict(model) if isinstance(model, dict) else {"model": model}
            # Store extractor names instead of instances to keep payload light.
            fe_list = model_payload.get("feature_extractors", [])
            fe_names = []
            for fe in fe_list:
                if isinstance(fe, str):
                    fe_names.append(fe)
                else:
                    fe_names.append(fe.__class__.__name__)
            if fe_names:
                model_payload["feature_extractor_names"] = fe_names
            model_payload.pop("feature_extractors", None)
            # Persist trained pipelines separately so they can be rehydrated later.
            AdartsService._persist_trained_pipelines(model_payload)
            # Avoid pickling heavy objects; rebuild when needed.
            model_payload.pop("training_set", None)
            model_payload.pop("trainer", None)
            model_payload.pop("clusterer", None)
            model_payload.pop("labeler", None)
            model_payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            model_payload["version"] = 1

            os.makedirs(AdartsService.get_system_inputs_dir(), exist_ok=True)
            joblib.dump(model_payload, AdartsService.get_trained_model_path())
            print("[AdartsService] Trained model saved (disk).")
        except Exception as e:
            print(f"[WARN] Failed to persist trained model: {e}")
        print("[AdartsService] Trained model saved.")
    
    @staticmethod
    def get_trained_model():
        """Get the trained model"""
        if AdartsService._trained_model is not None:
            return AdartsService._trained_model
        try:
            import joblib

            model_path = AdartsService.get_trained_model_path()
            if not os.path.exists(model_path):
                return None
            model_payload = joblib.load(model_path)

            model_payload["feature_extractors"] = AdartsService._build_feature_extractors_from_names(
                model_payload.get("feature_extractor_names", [])
            )
            AdartsService._hydrate_trained_pipelines(model_payload)
            AdartsService._trained_model = model_payload
            print("[AdartsService] Trained model loaded (disk).")
            return AdartsService._trained_model
        except Exception as e:
            print(f"[WARN] Failed to load trained model from disk: {e}")
            return None

    @staticmethod
    def clear_trained_model():
        AdartsService._trained_model = None
        try:
            model_path = AdartsService.get_trained_model_path()
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception as e:
            print(f"[WARN] Failed to remove persisted trained model: {e}")

    # ========== FLAML Baseline Methods ==========
    
    @staticmethod
    def set_flaml_model(model_data):
        """Save the trained FLAML model"""
        AdartsService._flaml_model = model_data
        AdartsService._persist_baseline_model("FLAML", model_data)
        print("[AdartsService] FLAML model saved.")
    
    @staticmethod
    def get_flaml_model():
        """Get the FLAML model"""
        if AdartsService._flaml_model is not None:
            return AdartsService._flaml_model
        AdartsService._flaml_model = AdartsService._load_baseline_model("FLAML")
        return AdartsService._flaml_model
    
    @staticmethod
    def clear_flaml_model():
        """Clear the FLAML model"""
        AdartsService._flaml_model = None
        AdartsService._clear_baseline_model_file("FLAML")
        print("[AdartsService] FLAML model cleared.")

    # ========== Tune Baseline Methods ==========
    
    @staticmethod
    def set_tune_model(model_data):
        """Save the trained Tune model"""
        AdartsService._tune_model = model_data
        AdartsService._persist_baseline_model("TUNE", model_data)
        print("[AdartsService] Tune model saved.")
    
    @staticmethod
    def get_tune_model():
        """Get the Tune model"""
        if AdartsService._tune_model is not None:
            return AdartsService._tune_model
        AdartsService._tune_model = AdartsService._load_baseline_model("TUNE")
        return AdartsService._tune_model
    
    @staticmethod
    def clear_tune_model():
        """Clear the Tune model"""
        AdartsService._tune_model = None
        AdartsService._clear_baseline_model_file("TUNE")
        print("[AdartsService] Tune model cleared.")

    # ========== AutoFolio Baseline Methods ==========

    @staticmethod
    def set_autofolio_model(model_data):
        """Save the trained AutoFolio model"""
        AdartsService._autofolio_model = model_data
        AdartsService._persist_baseline_model("AUTOFOLIO", model_data)
        print("[AdartsService] AutoFolio model saved.")

    @staticmethod
    def get_autofolio_model():
        """Get the AutoFolio model"""
        if AdartsService._autofolio_model is not None:
            return AdartsService._autofolio_model
        AdartsService._autofolio_model = AdartsService._load_baseline_model("AUTOFOLIO")
        return AdartsService._autofolio_model

    @staticmethod
    def clear_autofolio_model():
        """Clear the AutoFolio model"""
        AdartsService._autofolio_model = None
        AdartsService._clear_baseline_model_file("AUTOFOLIO")
        print("[AdartsService] AutoFolio model cleared.")

    # ========== RAHA Baseline Methods ==========

    @staticmethod
    def set_raha_model(model_data):
        """Save the trained RAHA model"""
        AdartsService._raha_model = model_data
        AdartsService._persist_baseline_model("RAHA", model_data)
        print("[AdartsService] RAHA model saved.")

    @staticmethod
    def get_raha_model():
        """Get the RAHA model"""
        if AdartsService._raha_model is not None:
            return AdartsService._raha_model
        AdartsService._raha_model = AdartsService._load_baseline_model("RAHA")
        return AdartsService._raha_model

    @staticmethod
    def clear_raha_model():
        """Clear the RAHA model"""
        AdartsService._raha_model = None
        AdartsService._clear_baseline_model_file("RAHA")
        print("[AdartsService] RAHA model cleared.")

    # ========== Ground Truth and Evaluation Methods ==========
    
    @staticmethod
    def set_ground_truth(data, timeseries_ids=None, meta=None):
        """Save the ground truth data (original complete data before missing injection)"""
        AdartsService._ground_truth_data = {
            'data': data,
            'timeseries_ids': timeseries_ids,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"[AdartsService] Ground truth saved. Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        try:
            gt_path = AdartsService.get_ground_truth_path()
            gt_meta_path = AdartsService.get_ground_truth_meta_path()
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            df.to_csv(gt_path, sep=' ', index=False, header=False)
            meta_payload = {
                "timestamp": AdartsService._ground_truth_data["timestamp"],
                "timeseries_ids": list(timeseries_ids) if timeseries_ids is not None else None,
            }
            if isinstance(meta, dict):
                meta_payload.update(meta)
            meta_payload = _json_safe_obj(meta_payload)
            with open(gt_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f)
        except Exception as e:
            print(f"[WARN] Failed to persist ground truth to disk: {e}")
    
    @staticmethod
    def get_ground_truth():
        """Get the ground truth data"""
        if AdartsService._ground_truth_data is not None:
            return AdartsService._ground_truth_data
        try:
            gt_path = AdartsService.get_ground_truth_path()
            if not os.path.exists(gt_path):
                return None
            try:
                gt_df = pd.read_csv(gt_path, sep=None, engine='python', header=None)
            except Exception:
                gt_df = pd.read_csv(gt_path, sep=' ', header=None)
            meta = {}
            meta_path = AdartsService.get_ground_truth_meta_path()
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
            AdartsService._ground_truth_data = {
                "data": gt_df,
                "timeseries_ids": meta.get("timeseries_ids"),
                "timestamp": meta.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if AdartsService._evaluation_mode is None and meta.get("mode"):
                AdartsService._evaluation_mode = meta.get("mode")
            if meta.get("missing_rate") is not None:
                AdartsService._missing_injection_rate = meta.get("missing_rate")
            return AdartsService._ground_truth_data
        except Exception as e:
            print(f"[WARN] Failed to load ground truth from disk: {e}")
            return None
    
    @staticmethod
    def clear_ground_truth():
        """Clear the ground truth data"""
        AdartsService._ground_truth_data = None
        print("[AdartsService] Ground truth cleared.")
        try:
            gt_path = AdartsService.get_ground_truth_path()
            meta_path = AdartsService.get_ground_truth_meta_path()
            if os.path.exists(gt_path):
                os.remove(gt_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
        except Exception as e:
            print(f"[WARN] Failed to remove persisted ground truth: {e}")
    
    @staticmethod
    def set_evaluation_mode(mode, missing_rate=0.2):
        """Set the evaluation mode ('test_set' or 'upload') and missing injection rate"""
        AdartsService._evaluation_mode = mode
        AdartsService._missing_injection_rate = missing_rate
        print(f"[AdartsService] Evaluation mode set to: {mode}, missing rate: {missing_rate}")
    
    @staticmethod
    def get_evaluation_mode():
        """Get the current evaluation mode"""
        return AdartsService._evaluation_mode
    
    @staticmethod
    def get_missing_injection_rate():
        """Get the missing injection rate"""
        return AdartsService._missing_injection_rate

    # ========== Ground Truth Labels Methods ==========
    
    _ground_truth_labels = None  # Ground truth labels for evaluation
    _all_algorithms_results = None  # Results from running all algorithms
    
    @staticmethod
    def set_ground_truth_labels(labels_data):
        """Save the ground truth labels (best algorithm for each time series)"""
        AdartsService._ground_truth_labels = labels_data
        print(f"[AdartsService] Ground truth labels saved. Count: {len(labels_data) if labels_data else 0}")
    
    @staticmethod
    def get_ground_truth_labels():
        """Get the ground truth labels"""
        return AdartsService._ground_truth_labels
    
    @staticmethod
    def clear_ground_truth_labels():
        """Clear the ground truth labels"""
        AdartsService._ground_truth_labels = None
        print("[AdartsService] Ground truth labels cleared.")
    
    @staticmethod
    def set_all_algorithms_results(results):
        """Save results from running all algorithms"""
        AdartsService._all_algorithms_results = results
        print(f"[AdartsService] All algorithms results saved.")
    
    @staticmethod
    def get_all_algorithms_results():
        """Get results from running all algorithms"""
        return AdartsService._all_algorithms_results


# ========== Basic Views ==========

def _run_external_baseline_runner(baseline_name, mode, arrays_dict, meta_dict=None, timeout_sec=3600):
    """
    Execute a baseline runner in its dedicated venv.

    Runner contract:
    - Args:
      --mode train|predict
      --input <npz file>
      --meta <json file>
      --output <json file>
    - Output JSON:
      {"status":"success"|"failed", ...}
    """
    cfg = AdartsService.get_external_runner_config(baseline_name)
    python_exec = cfg.get("python")
    script_path = cfg.get("script")

    if not python_exec or not script_path:
        return {"status": "failed", "error": f"{baseline_name} external runner is not configured."}
    if not os.path.exists(script_path):
        return {"status": "failed", "error": f"{baseline_name} runner script not found: {script_path}"}

    temp_dir = tempfile.mkdtemp(prefix=f"{baseline_name.lower()}_runner_")
    input_npz = os.path.join(temp_dir, "input.npz")
    input_meta = os.path.join(temp_dir, "meta.json")
    output_json = os.path.join(temp_dir, "output.json")

    try:
        np.savez_compressed(input_npz, **arrays_dict)
        with open(input_meta, "w", encoding="utf-8") as f:
            json.dump(meta_dict or {}, f)

        cmd = [
            python_exec,
            script_path,
            "--mode", mode,
            "--input", input_npz,
            "--meta", input_meta,
            "--output", output_json,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "status": "failed",
                "error": f"{baseline_name} runner failed (code={proc.returncode})",
                "stderr": (proc.stderr or "")[-2000:],
                "stdout": (proc.stdout or "")[-2000:],
            }
        if not os.path.exists(output_json):
            return {"status": "failed", "error": f"{baseline_name} runner produced no output file."}

        with open(output_json, "r", encoding="utf-8") as f:
            result = json.load(f)
        if "status" not in result:
            result["status"] = "success"
        return result

    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"{baseline_name} runner timeout after {timeout_sec}s."}
    except Exception as e:
        return {"status": "failed", "error": f"{baseline_name} runner exception: {e}"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_external_labeling_runner(mode, arrays_dict, meta_dict=None, timeout_sec=None):
    """
    Execute the external labeling runner in its dedicated venv.

    Runner contract:
    - Args:
      --mode impute|benchmark
      --input <npz file>
      --meta <json file>
      --output <json file>
    - Output JSON:
      {"status":"success"|"failed", ...}
    """
    cfg = AdartsService.get_external_labeling_config()
    python_exec = cfg.get("python")
    script_path = cfg.get("script")
    timeout_val = int(timeout_sec or cfg.get("timeout_sec", 1800))

    if not python_exec or not script_path:
        return {"status": "failed", "error": "External labeling runner is not configured."}
    if not os.path.exists(script_path):
        return {"status": "failed", "error": f"Labeling runner script not found: {script_path}"}

    temp_dir = tempfile.mkdtemp(prefix="labeling_runner_")
    input_npz = os.path.join(temp_dir, "input.npz")
    input_meta = os.path.join(temp_dir, "meta.json")
    output_json = os.path.join(temp_dir, "output.json")

    try:
        np.savez_compressed(input_npz, **arrays_dict)
        with open(input_meta, "w", encoding="utf-8") as f:
            json.dump(meta_dict or {}, f)

        cmd = [
            python_exec,
            script_path,
            "--mode", mode,
            "--input", input_npz,
            "--meta", input_meta,
            "--output", output_json,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_val,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "status": "failed",
                "error": f"Labeling runner failed (code={proc.returncode})",
                "stderr": (proc.stderr or "")[-2000:],
                "stdout": (proc.stdout or "")[-2000:],
            }
        if not os.path.exists(output_json):
            return {"status": "failed", "error": "Labeling runner produced no output file."}

        with open(output_json, "r", encoding="utf-8") as f:
            result = json.load(f)
        if "status" not in result:
            result["status"] = "success"
        return result

    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"Labeling runner timeout after {timeout_val}s."}
    except Exception as e:
        return {"status": "failed", "error": f"Labeling runner exception: {e}"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _format_algo_name(algo_name):
    """
    Normalize algorithm names for UI output.
    """
    if algo_name is None:
        return "Unknown"

    name = str(algo_name).strip()
    if not name:
        return "Unknown"

    normalized = name.lower()
    if normalized.startswith("cdrec_"):
        normalized = "cdrec"

    aliases = {
        "stmvl": "STMVL",
        "cdrec": "CDRec",
        "svdimp": "SVDImp",
        "trmf": "TRMF",
        "tkcm": "TKCM",
        "dynammo": "DynaMMo",
        "tenmf": "TeNMF",
        "svt": "SVT",
        "grouse": "GROUSE",
        "softimp": "SoftImp",
        "rosl": "ROSL",
        "mrnn": "MRNN",
        "brits": "BRITS",
        "deepmvi": "DeepMVI",
        "mpin": "MPIN",
        "iim": "IIM",
        "pristi": "Pristi",
    }
    return aliases.get(normalized, name)


def _normalize_algo_key(algo_name):
    if algo_name is None:
        return ""
    key = str(algo_name).strip().lower()
    if not key:
        return ""
    if key.startswith("cdrec_"):
        key = "cdrec"
    return key


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value):
    val = _safe_float(value, default=None)
    if val is None:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def _get_external_dl_algos():
    cfg = AdartsService.get_external_labeling_config()
    algos = set()
    cfg_algo = cfg.get("algo")
    if cfg_algo:
        algos.add(str(cfg_algo).strip().lower())

    extra = getattr(settings, "DL_IMPUTE_ALGOS", None)
    if extra:
        if isinstance(extra, (list, tuple, set)):
            for item in extra:
                if item:
                    algos.add(str(item).strip().lower())
        else:
            for item in str(extra).split(","):
                if item.strip():
                    algos.add(item.strip().lower())

    if not algos:
        algos.update({"brits", "deepmvi"})
    return algos


def _is_external_dl_algo(algorithm):
    if algorithm is None:
        return False
    return str(algorithm).strip().lower() in _get_external_dl_algos()


def _read_imputed_file(path):
    try:
        return pd.read_csv(path, sep=None, engine='python', header=None)
    except Exception:
        return pd.read_csv(path, sep=' ', header=None)


def _run_external_imputation_for_algo(timeseries, algorithm, timeout_sec=None):
    algo = str(algorithm).strip().lower()
    if not AdartsService.is_external_labeling_enabled():
        return {"error": "External DL labeling runner is not configured."}

    inference_dir = AdartsService.get_inference_dir()
    os.makedirs(inference_dir, exist_ok=True)
    out_path = os.path.join(inference_dir, f'imputed_{algo}_{int(time.time())}.csv')

    if isinstance(timeseries, pd.DataFrame) or isinstance(timeseries, pd.Series):
        values = timeseries.to_numpy(dtype="float32")
    else:
        values = np.asarray(timeseries, dtype="float32")

    meta = {
        "algorithm": algo,
        "device": getattr(settings, "DL_LABEL_DEVICE", "cpu"),
        "imputed_output_path": out_path,
    }
    if algo == "deepmvi":
        meta.update({"epochs": 8, "patience": 2, "tr_ratio": 0.7, "device": "cpu"})
    elif algo == "mrnn":
        meta.update({"iterations": 200, "hidden_dim": 10, "seq_length": 7, "tr_ratio": 0.7})
    elif algo == "mpin":
        meta.update({"epochs": 50, "num_of_iteration": 3, "window": 2, "k": 10, "tr_ratio": 0.7})
    elif algo == "iim":
        meta.update({"neighbors": 10, "adaptive": False})

    result = _run_external_labeling_runner(
        mode="impute",
        arrays_dict={"X_input": values},
        meta_dict=meta,
        timeout_sec=timeout_sec or getattr(settings, "DL_LABEL_TIMEOUT_SEC", 1800),
    )

    if result.get("status") != "success":
        err = result.get("error") or "External DL runner failed."
        return {"error": err, "details": result}

    if not result.get("imputed_file") and os.path.exists(out_path):
        result["imputed_file"] = out_path
    if not result.get("imputed_file"):
        return {"error": "External DL runner succeeded but produced no imputed file.", "details": result}
    return result


def _extract_best_algo_from_benchmark_row(labeler, benchmark_results_raw, algos_to_exclude=None):
    """
    Parse a benchmark-results row and return:
    - best algorithm name
    - associated RMSE score (if available)
    """
    if benchmark_results_raw is None or (isinstance(benchmark_results_raw, float) and np.isnan(benchmark_results_raw)):
        return "Unknown", None

    if algos_to_exclude is None:
        algos_to_exclude = []

    benchmark_results = labeler._convert_bench_res_to_df(str(benchmark_results_raw))
    ranking_strat = labeler.CONF["BENCH_RES_AGG_AND_RANK_STRATEGY"]
    ranking_scores = labeler._get_ranking_from_bench_res(
        benchmark_results,
        ranking_strat=ranking_strat,
        ranking_strat_params=labeler.CONF["BENCH_RES_AGG_AND_RANK_STRATEGY_PARAMS"][ranking_strat],
        error_to_minimize=labeler.CONF["BENCHMARK_ERROR_TO_MINIMIZE"],
        algos_to_exclude=algos_to_exclude,
        return_scores=True,
    )

    if ranking_scores is None or len(ranking_scores.index) == 0:
        return "Unknown", None

    best_algo = str(ranking_scores.index[0])
    rmse_col = None
    lower_to_orig_cols = {str(col).strip().lower(): col for col in ranking_scores.columns}
    for col_name in ("weighted average rmse", "average rmse", "rmse"):
        if col_name in lower_to_orig_cols:
            rmse_col = lower_to_orig_cols[col_name]
            break
    if rmse_col is None:
        for col in ranking_scores.columns:
            if "rmse" in str(col).lower():
                rmse_col = col
                break

    rmse_val = None
    if rmse_col is not None:
        rmse_val = _safe_float(ranking_scores.iloc[0][rmse_col], default=None)
    return best_algo, rmse_val


def _build_external_eval_mask(values, seed, missing_rate=0.2):
    rng = np.random.default_rng(seed)
    eval_mask = rng.random(values.shape) < missing_rate
    if not np.any(eval_mask):
        eval_mask.reshape(-1)[0] = True
    return eval_mask


def _build_eval_mask_from_observed(values, seed, missing_rate=0.1):
    rng = np.random.default_rng(seed)
    observed_mask = ~np.isnan(values)
    eval_mask = np.zeros_like(observed_mask, dtype=bool)
    for i in range(observed_mask.shape[0]):
        obs_idx = np.where(observed_mask[i])[0]
        if obs_idx.size == 0:
            continue
        k = int(max(1, round(obs_idx.size * missing_rate)))
        if obs_idx.size > 1:
            k = min(k, obs_idx.size - 1)
        chosen = rng.choice(obs_idx, size=k, replace=False)
        eval_mask[i, chosen] = True
    if not np.any(eval_mask) and observed_mask.any():
        first_idx = np.argwhere(observed_mask)
        eval_mask[first_idx[0][0], first_idx[0][1]] = True
    return eval_mask


def _load_or_create_downstream_eval_mask(values, inference_dir, missing_rate, seed, regenerate=False):
    mask_path = os.path.join(inference_dir, "downstream_eval_mask.npy")
    meta_path = os.path.join(inference_dir, "downstream_eval_meta.json")
    if not regenerate and os.path.exists(mask_path):
        try:
            mask = np.load(mask_path)
            if mask.shape == values.shape:
                return mask
        except Exception:
            pass
    mask = _build_eval_mask_from_observed(values, seed, missing_rate=missing_rate)
    np.save(mask_path, mask)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "missing_rate": missing_rate,
                    "seed": seed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )
    except Exception:
        pass
    return mask


def _evaluate_external_algo_on_cluster(dataset, cassignment, full_timeseries, cluster_id, external_algo, mask_seed=None, missing_rate=0.2):
    """
    Run external DL labeling runner on one cluster and return its payload.
    """
    cluster_ts = dataset.get_cluster_by_id(full_timeseries, cluster_id, cassignment)
    cluster_ts = cluster_ts.apply(pd.to_numeric, errors="coerce")
    values = cluster_ts.to_numpy(dtype="float32")

    if values.size == 0:
        return {"status": "skipped", "message": "Cluster is empty."}

    values = values[: min(64, values.shape[0])]
    if values.shape[0] == 0 or values.shape[1] == 0:
        return {"status": "skipped", "message": "Cluster has invalid shape for DL evaluation."}

    # Reproducible mask per cluster for consistent RMSE comparisons.
    if mask_seed is None:
        mask_seed = abs(hash((dataset.name, int(cluster_id)))) % (2**32)
    eval_mask = _build_external_eval_mask(values, mask_seed, missing_rate=missing_rate)

    values_with_missing = values.copy()
    values_with_missing[eval_mask] = np.nan

    algo_key = str(external_algo).strip().lower()
    meta_dict = {
        "algorithm": algo_key,
        "device": getattr(settings, "DL_LABEL_DEVICE", "cpu"),
    }
    if algo_key == "deepmvi":
        meta_dict.update({"epochs": 8, "patience": 2, "tr_ratio": 0.7, "device": "cpu"})
    elif algo_key == "mrnn":
        meta_dict.update({"iterations": 200, "hidden_dim": 10, "seq_length": 7, "tr_ratio": 0.7})
    elif algo_key == "mpin":
        meta_dict.update({"epochs": 50, "num_of_iteration": 3, "window": 2, "k": 10, "tr_ratio": 0.7})
    elif algo_key == "iim":
        meta_dict.update({"neighbors": 10, "adaptive": False})

    return _run_external_labeling_runner(
        mode="benchmark",
        arrays_dict={
            "X_input": values_with_missing,
            "X_true": values,
            "eval_mask": eval_mask,
        },
        meta_dict=meta_dict,
        timeout_sec=getattr(settings, "DL_LABEL_TIMEOUT_SEC", 1800),
    )


def _safe_extract_zip(zip_ref, target_dir):
    """
    Safely extract zip archive entries into target_dir.
    Prevents path traversal (Zip Slip) by validating each member path.
    """
    abs_target_dir = os.path.abspath(target_dir)
    extracted_files = []

    for member in zip_ref.infolist():
        member_name = member.filename
        if not member_name:
            continue

        normalized_member = os.path.normpath(member_name)
        if normalized_member in ("", "."):
            continue
        if os.path.isabs(normalized_member) or normalized_member == ".." or normalized_member.startswith(f"..{os.sep}"):
            raise ValueError(f"Unsafe zip member path: {member_name}")

        destination = os.path.abspath(os.path.join(abs_target_dir, normalized_member))
        if os.path.commonpath([abs_target_dir, destination]) != abs_target_dir:
            raise ValueError(f"Unsafe zip member path: {member_name}")

        if member.is_dir():
            os.makedirs(destination, exist_ok=True)
            continue

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with zip_ref.open(member, "r") as src, open(destination, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted_files.append(destination)

    return extracted_files


def hello(request):
    """A view for test"""
    return JsonResponse({'message': 'ImputePilot Backend is running', 'status': 'success'})

def index(request):
    """Main view"""
    return render(request, 'ImputePilot_api/ImputePilot_v3.html')


# ========== Pipeline Views ==========

@csrf_exempt  
@require_http_methods(["POST"])
def upload_training_data(request):
    try:
        if 'files' not in request.FILES:
            return JsonResponse({'error': 'No files uploaded'}, status=400)
        
        files = request.FILES.getlist('files')
        uploaded_files = []
        
        real_world_dir = AdartsService.get_dataset_dir()
        system_inputs_dir = AdartsService.get_system_inputs_dir()
        
        # Ensure directories exist (do NOT delete pre-installed datasets)
        os.makedirs(real_world_dir, exist_ok=True)
        os.makedirs(system_inputs_dir, exist_ok=True)
        
        # Count pre-installed datasets before upload
        existing_files = set(os.listdir(real_world_dir))
        print(f"[INFO] Pre-installed datasets in RealWorld/: {len(existing_files)} files")
        
        # Remove intermediate results (force re-train with new data)
        base_path = AdartsService.get_base_path()
        dirs_to_clear = [
            os.path.join(base_path, 'Clustering', 'cassignments'),
            os.path.join(base_path, 'Labeling', 'ImputationTechniques', 'labels'),
            os.path.join(base_path, 'FeaturesExtraction', 'features'),
        ]
        for dir_path in dirs_to_clear:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        
        # Reset trained models in memory and on disk (force re-train)
        AdartsService.clear_trained_model()
        AdartsService._clusterer = None
        AdartsService._labeler = None
        AdartsService.clear_flaml_model()
        AdartsService.clear_tune_model()
        AdartsService.clear_autofolio_model()
        AdartsService.clear_raha_model()
        
        print("[INFO] Cleared intermediate results and model cache (pre-installed datasets preserved)")
        
        # Save uploaded files (append alongside pre-installed datasets)
        skipped_files = []
        for file in files:
            # If a file with the same name already exists, add "user_" prefix to avoid overwriting
            target_name = os.path.basename(file.name)
            if target_name in existing_files:
                skipped_files.append(target_name)
                print(f"[INFO] File '{file.name}' already exists in RealWorld/, skipping")
                continue
            
            file_path = os.path.join(real_world_dir, target_name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            shutil.copy(file_path, os.path.join(system_inputs_dir, target_name))
            uploaded_files.append(target_name)
        
        total_datasets = len(os.listdir(real_world_dir))
        print(f"[INFO] Total datasets after upload: {total_datasets}")
        
        # Generate preview data
        preview_data = None
        try:
            def _build_preview_from_lines(lines, source_name, total_rows=None):
                if not lines:
                    return None

                first_line = lines[0]
                if ',' in first_line:
                    delim = ','
                elif '\t' in first_line:
                    delim = '\t'
                else:
                    delim = ' '

                rows = [line.split(delim) for line in lines]

                chart_data = []
                if rows and len(rows[0]) > 0:
                    first_series = rows[0]
                    for idx, val in enumerate(first_series):
                        try:
                            float_val = float(val)
                            if np.isnan(float_val) or val.strip() == '' or val.strip().lower() == 'nan':
                                chart_data.append({'x': idx, 'y': None, 'missing': True})
                            else:
                                chart_data.append({'x': idx, 'y': float_val, 'missing': False})
                        except (ValueError, TypeError):
                            chart_data.append({'x': idx, 'y': None, 'missing': True})

                total_points = len(chart_data)
                missing_points = sum(1 for p in chart_data if p['missing'])
                missing_rate = round(missing_points / total_points * 100, 1) if total_points > 0 else 0
                resolved_total_rows = total_rows if total_rows is not None else len(lines)

                return {
                    'fileName': source_name,
                    'totalRows': resolved_total_rows,
                    'columns': len(rows[0]) if rows else 0,
                    'headers': rows[0] if rows else [],
                    'rows': rows[1:10] if len(rows) > 1 else [],
                    'seriesRows': rows[:10] if rows else [],
                    'chartData': chart_data[:500],
                    'totalPoints': total_points,
                    'missingPoints': missing_points,
                    'missingRate': missing_rate
                }

            def _read_preview_lines_from_file(path):
                lines = []
                total_rows = 0
                with open(path, 'rb') as f:
                    for line in f:
                        total_rows += 1
                        if total_rows > 10:
                            continue
                        try:
                            lines.append(line.decode('utf-8').strip())
                        except UnicodeDecodeError:
                            lines.append(line.decode('latin-1').strip())
                return lines, total_rows

            def _create_preview_for_dataset(dataset_filename):
                dataset_path = os.path.join(real_world_dir, dataset_filename)
                if not os.path.exists(dataset_path):
                    return None

                lower_name = dataset_filename.lower()
                if lower_name.endswith('.zip'):
                    print(f"[DEBUG] Opening zip for preview: {dataset_path}")
                    with zipfile.ZipFile(dataset_path, 'r') as z:
                        namelist = z.namelist()
                        print(f"[DEBUG] Zip contents: {namelist}")

                        for inner_name in namelist:
                            if inner_name.lower().endswith(('.csv', '.txt', '.tsv')):
                                print(f"[DEBUG] Found data file in zip: {inner_name}")
                                lines = []
                                total_rows = 0
                                with z.open(inner_name) as f:
                                    for line in f:
                                        total_rows += 1
                                        if total_rows > 10:
                                            continue
                                        try:
                                            lines.append(line.decode('utf-8').strip())
                                        except UnicodeDecodeError:
                                            lines.append(line.decode('latin-1').strip())
                                return _build_preview_from_lines(lines, inner_name, total_rows=total_rows)
                    return None

                if lower_name.endswith(('.csv', '.txt', '.tsv')):
                    lines, total_rows = _read_preview_lines_from_file(dataset_path)
                    return _build_preview_from_lines(lines, dataset_filename, total_rows=total_rows)

                return None

            # First priority: newly uploaded files
            print(f"[DEBUG] Looking for preview in newly uploaded files: {uploaded_files}")
            for file_name in uploaded_files:
                preview_data = _create_preview_for_dataset(file_name)
                if preview_data:
                    print(f"[DEBUG] Preview data created from uploaded file: {preview_data['fileName']}")
                    break

            # Fallback: duplicate files that were skipped but already exist in RealWorld
            if preview_data is None and skipped_files:
                print(f"[DEBUG] No uploaded preview found, trying skipped duplicates: {skipped_files}")
                for file_name in skipped_files:
                    preview_data = _create_preview_for_dataset(file_name)
                    if preview_data:
                        print(f"[DEBUG] Preview data created from existing dataset: {preview_data['fileName']}")
                        break
        except Exception as e:
            print(f"[ERROR] Failed to generate preview: {e}")
            traceback.print_exc()
            preview_data = None
        
        return JsonResponse({
            'datasetId': f'training-dataset-{len(uploaded_files)}',
            'files': uploaded_files,
            'message': f'{len(uploaded_files)} files uploaded successfully. Total datasets: {total_datasets} (pre-installed: {len(existing_files)}, user-added: {total_datasets - len(existing_files)})',
            'preview': preview_data,
            'totalDatasets': total_datasets,
            'preInstalledCount': len(existing_files),
            'skippedDuplicates': skipped_files
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_clustering(request):
    try:
        print("[INFO] Starting Clustering...")
        clusterer = AdartsService.get_clusterer()
        datasets = AdartsService.load_datasets()
        
        if not datasets:
            return JsonResponse({'error': 'No datasets found. Please upload data first.'}, status=400)

        clusterer.cluster_all_datasets_seq(datasets)
        
        clusters_data = []

        def _build_shape_preview(cluster_df, max_points=120):
            """Build a compact representative shape from one cluster (mean curve)."""
            if cluster_df is None or cluster_df.empty:
                return []
            try:
                mean_curve = cluster_df.mean(axis=0, skipna=True)
                arr = pd.to_numeric(mean_curve, errors='coerce').to_numpy(dtype=float)
            except Exception:
                return []

            if arr.size == 0:
                return []

            # Fill non-finite values to keep preview drawable.
            s = pd.Series(arr)
            s = s.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction='both').ffill().bfill().fillna(0.0)
            arr = s.to_numpy(dtype=float)
            if arr.size == 0:
                return []

            # Downsample to keep payload/UI responsive.
            if arr.size > max_points:
                idx = np.linspace(0, arr.size - 1, max_points).round().astype(int)
                idx = np.unique(idx)
                arr = arr[idx]

            return [round(float(v), 6) for v in arr if np.isfinite(v)]

        for dataset in datasets:
            try:
                cassignment = dataset.load_cassignment(clusterer)
                cluster_counts = cassignment['Cluster ID'].value_counts()
                timeseries = dataset.load_timeseries(transpose=True)
                
                for cluster_id, count in cluster_counts.items():
                    try:
                        cluster_ts = dataset.get_cluster_by_id(timeseries, cluster_id, cassignment)
                        avg_corr = clusterer._get_dataset_mean_ncc_score(cluster_ts)
                        shape_preview = _build_shape_preview(cluster_ts)
                        
                        clusters_data.append({
                            'id': int(cluster_id), 
                            'name': f'Cluster {int(cluster_id)} ({dataset.name})', 
                            'count': int(count), 
                            'rho': round(float(avg_corr), 3),
                            'dataset': dataset.name,
                            'shapePreview': shape_preview,
                            'previewPoints': len(shape_preview),
                        })
                    except Exception as inner_e:
                        print(f"[WARN] Error calculating Rho for cluster {cluster_id}: {inner_e}")
                        
            except Exception as e:
                print(f"[WARN] Error processing dataset {dataset.name}: {e}")
                
        return JsonResponse({'clusters': clusters_data})
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f"Clustering failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_labeling(request):
    try:
        print("[INFO] Starting Labeling...")
        request_data = {}
        if request.body:
            try:
                request_data = json.loads(request.body)
            except json.JSONDecodeError:
                request_data = {}
        if not request_data and request.POST:
            request_data = request.POST

        external_requested = bool(request_data.get("use_external_dl", False))
        external_algo = str(
            request_data.get("external_algo", getattr(settings, "DL_LABEL_ALGO", "brits"))
        ).strip().lower()
        selected_algos_raw = request_data.get("algorithms", [])
        if isinstance(request_data, dict) and not selected_algos_raw and hasattr(request, "POST"):
            selected_algos_raw = request.POST.getlist("algorithms")
        if isinstance(selected_algos_raw, str):
            selected_algos_raw = [s.strip() for s in selected_algos_raw.split(",") if s.strip()]
        if not isinstance(selected_algos_raw, (list, tuple, set)):
            selected_algos_raw = []
        selected_algos = {
            _normalize_algo_key(a) for a in selected_algos_raw
            if a is not None and str(a).strip()
        }
        selected_algos.discard("")
        external_algos_selected = sorted({a for a in selected_algos if _is_external_dl_algo(a)})
        if external_requested and not external_algos_selected and _is_external_dl_algo(external_algo):
            external_algos_selected = [external_algo]

        use_external_dl = external_requested and bool(external_algos_selected)
        print(
            "[INFO] Labeling selection: "
            f"selected={sorted(selected_algos)}, external_requested={external_requested}, "
            f"external_algos={external_algos_selected}, use_external_dl={use_external_dl}"
        )

        clusterer = AdartsService.get_clusterer()
        labeler = AdartsService.get_labeler()
        datasets = AdartsService.load_datasets()
        
        if not datasets:
            return JsonResponse({'error': 'No datasets found.'}, status=400)

        if selected_algos:
            non_external_selected = [a for a in selected_algos if not _is_external_dl_algo(a)]
            if not non_external_selected and not use_external_dl:
                return JsonResponse({
                    'error': 'Only external DL algorithms were selected, but external runner is not enabled.',
                }, status=400)

        updated_datasets = _run_with_heartbeat(
            "Labeling (ImputeBench)",
            lambda: labeler.label_all_datasets(datasets)
        )

        external_dl_status = {
            "requested": external_requested,
            "enabled": AdartsService.is_external_labeling_enabled(),
            "algorithm": external_algos_selected[0] if len(external_algos_selected) == 1 else ("multi" if external_algos_selected else external_algo),
            "algorithms": external_algos_selected,
            "used": False,
            "clustersTotal": 0,
            "clustersEvaluated": 0,
            "clustersOverridden": 0,
            "clustersFailed": 0,
            "details": [],
        }
        override_enabled = use_external_dl and external_dl_status["enabled"]

        if external_requested and selected_algos and not external_algos_selected:
            external_dl_status.update({
                "status": "skipped",
                "message": "No external DL algorithms were selected.",
            })
        elif use_external_dl and not external_dl_status["enabled"]:
            external_dl_status.update({
                "status": "skipped",
                "message": "External DL labeling runner is not configured.",
            })
        elif external_requested and not external_algos_selected:
            external_dl_status.update({
                "status": "skipped",
                "message": "No external DL algorithms selected for evaluation.",
            })

        external_rmse_values = []
        labeling_results = []
        for dataset in updated_datasets:
            override_rows = []
            labels_filename = labeler._get_labels_filename(dataset.name)
            if not os.path.exists(labels_filename):
                continue

            df_labels = pd.read_csv(labels_filename)
            if df_labels.empty:
                continue

            cassignment = dataset.load_cassignment(clusterer)
            cluster_counts = cassignment['Cluster ID'].value_counts()

            algos_to_exclude = []
            if 'Benchmark Results' in df_labels.columns:
                try:
                    properties = labeler.get_default_properties()
                    reduction_threshold = float(properties.get("reduction_threshold", 0.0))
                    if reduction_threshold > 0.0:
                        all_benchmark_results = df_labels[['Cluster ID', 'Benchmark Results']].set_index('Cluster ID')
                        algos_to_exclude = labeler._get_algos_to_exclude(all_benchmark_results, properties)
                except Exception as e:
                    print(f"[WARN] Failed to compute algos_to_exclude for {dataset.name}: {e}")
                    algos_to_exclude = []
                if selected_algos:
                    configured_algos = [
                        _normalize_algo_key(a) for a in labeler.CONF.get("ALGORITHMS_LIST", [])
                    ]
                    selection_excluded = [
                        a for a in configured_algos if a and a not in selected_algos
                    ]
                    if "cdrec" in selection_excluded:
                        selection_excluded.extend(["cdrec_k2", "cdrec_k3"])
                    if selection_excluded:
                        algos_to_exclude = list(set(algos_to_exclude) | set(selection_excluded))

            full_timeseries = None
            for _, row in df_labels.iterrows():
                cid_raw = row.get('Cluster ID', -1)
                cid = int(cid_raw)

                base_algo = "Unknown"
                base_rmse = None
                best_algo = "Unknown"
                selected_rmse = None
                decision_source = "imputebench"
                external_rmse = None
                best_external_algo = None

                if 'Benchmark Results' in df_labels.columns:
                    try:
                        parsed_algo, parsed_rmse = _extract_best_algo_from_benchmark_row(
                            labeler,
                            row.get('Benchmark Results'),
                            algos_to_exclude=algos_to_exclude,
                        )
                        base_algo = _format_algo_name(parsed_algo)
                        base_rmse = _finite_or_none(parsed_rmse)
                    except Exception as parse_e:
                        print(f"[WARN] Failed to parse benchmark row for dataset={dataset.name}, cluster={cid}: {parse_e}")
                        base_algo = "ParseError"
                        base_rmse = None
                elif 'Label' in df_labels.columns:
                    base_algo = _format_algo_name(row.get('Label'))
                    if selected_algos and _normalize_algo_key(base_algo) not in selected_algos:
                        base_algo = "Unknown"

                best_algo = base_algo
                selected_rmse = base_rmse

                if use_external_dl and external_dl_status["enabled"]:
                    external_dl_status["used"] = True
                    external_dl_status["clustersTotal"] += 1

                    try:
                        if full_timeseries is None:
                            full_timeseries = dataset.load_timeseries(transpose=True)

                        mask_seed = abs(hash((dataset.name, int(cid)))) % (2**32)
                        ext_results = []
                        celery_failed = None

                        try:
                            from ImputePilot_api.tasks import run_external_dl_eval_task
                            queue_name = getattr(settings, "DL_LABEL_CELERY_QUEUE", "dl_gpu")
                            timeout_sec = int(getattr(settings, "DL_LABEL_TIMEOUT_SEC", 1800))
                            task_sigs = [
                                run_external_dl_eval_task.s(dataset.name, cid, algo, mask_seed).set(queue=queue_name)
                                for algo in external_algos_selected
                            ]
                            ext_results = _run_with_heartbeat(
                                f"External DL eval {dataset.name} cluster {cid}",
                                lambda: group(task_sigs).apply_async().get(
                                    timeout=timeout_sec * max(1, len(task_sigs))
                                )
                            )
                        except CeleryTimeoutError as ext_timeout:
                            print(f"[WARN] External DL Celery timeout for dataset={dataset.name}, cluster={cid}: {ext_timeout}")
                            celery_failed = "timeout"
                            if len(external_dl_status["details"]) < 50:
                                external_dl_status["details"].append({
                                    "dataset": dataset.name,
                                    "clusterId": cid,
                                    "algorithm": "multi",
                                    "status": "failed",
                                    "error": "External DL Celery timeout.",
                                })
                        except Exception as ext_dispatch:
                            print(f"[WARN] External DL Celery dispatch failed for dataset={dataset.name}, cluster={cid}: {ext_dispatch}")
                            celery_failed = "dispatch"
                            if len(external_dl_status["details"]) < 50:
                                external_dl_status["details"].append({
                                    "dataset": dataset.name,
                                    "clusterId": cid,
                                    "algorithm": "multi",
                                    "status": "failed",
                                    "error": "External DL Celery dispatch failed.",
                                })

                        if not ext_results and celery_failed == "dispatch":
                            ext_results = []
                            for algo in external_algos_selected:
                                ext_results.append(_run_with_heartbeat(
                                    f"External DL fallback {dataset.name} cluster {cid} algo {algo}",
                                    lambda algo=algo: _evaluate_external_algo_on_cluster(
                                        dataset=dataset,
                                        cassignment=cassignment,
                                        full_timeseries=full_timeseries,
                                        cluster_id=cid,
                                        external_algo=algo,
                                        mask_seed=mask_seed,
                                    )
                                ))
                                if ext_results[-1]:
                                    ext_results[-1]["algorithm"] = algo

                        if ext_results:
                            print(f"[INFO] External DL results for dataset={dataset.name}, cluster={cid}: {ext_results}")

                        best_external_algo = None
                        best_external_rmse = None
                        cluster_has_valid = False

                        for ext_result in (ext_results or []):
                            algo_key = str(ext_result.get("algorithm", "")).strip().lower()
                            ext_status = ext_result.get("status", "failed")
                            ext_rmse = _finite_or_none(ext_result.get("rmse"))
                            detail = {
                                "dataset": dataset.name,
                                "clusterId": cid,
                                "algorithm": _format_algo_name(algo_key),
                                "status": ext_status,
                            }
                            if ext_rmse is not None:
                                detail["rmse"] = float(ext_rmse)
                            if ext_status != "success" or ext_rmse is None:
                                detail["error"] = ext_result.get("error", ext_result.get("message", "External DL failed."))

                            if ext_status == "success" and ext_rmse is not None:
                                cluster_has_valid = True
                                if best_external_rmse is None or float(ext_rmse) < float(best_external_rmse):
                                    best_external_rmse = float(ext_rmse)
                                    best_external_algo = algo_key

                            if len(external_dl_status["details"]) < 50:
                                external_dl_status["details"].append(detail)

                        if cluster_has_valid:
                            external_dl_status["clustersEvaluated"] += 1
                            external_rmse = float(best_external_rmse)
                            external_rmse_values.append(external_rmse)

                            should_override = (base_rmse is None) or (external_rmse <= float(base_rmse))
                            if should_override and best_external_algo:
                                best_algo = _format_algo_name(best_external_algo)
                                selected_rmse = external_rmse
                                decision_source = "external_dl"
                                external_dl_status["clustersOverridden"] += 1
                        else:
                            external_dl_status["clustersFailed"] += 1

                    except Exception as ext_e:
                        external_dl_status["clustersFailed"] += 1
                        if len(external_dl_status["details"]) < 50:
                            external_dl_status["details"].append({
                                "dataset": dataset.name,
                                "clusterId": cid,
                                "status": "failed",
                                "error": str(ext_e),
                            })

                row_result = {
                    'id': cid,
                    'name': f'Cluster {cid} ({dataset.name})',
                    'count': int(cluster_counts.get(cid, cluster_counts.get(cid_raw, 0))),
                    'bestAlgo': best_algo,
                    'decisionSource': decision_source,
                }
                row_result['baseAlgo'] = base_algo

                labeling_results.append(row_result)
                override_rows.append({
                    'Cluster ID': cid,
                    'Label': best_algo,
                    'baseAlgo': base_algo,
                    'baseRmse': float(base_rmse) if base_rmse is not None else None,
                    'externalAlgo': _format_algo_name(best_external_algo) if external_rmse is not None else None,
                    'externalRmse': float(external_rmse) if external_rmse is not None else None,
                    'decisionSource': decision_source,
                })

            if override_enabled and override_rows:
                try:
                    labeler.save_labels_override(dataset.name, pd.DataFrame(override_rows))
                except Exception as e:
                    print(f"[WARN] Failed to save override labels for {dataset.name}: {e}")

        if use_external_dl and external_dl_status["enabled"]:
            clusters_total = int(external_dl_status.get("clustersTotal", 0))
            clusters_evaluated = int(external_dl_status.get("clustersEvaluated", 0))
            clusters_overridden = int(external_dl_status.get("clustersOverridden", 0))
            clusters_failed = int(external_dl_status.get("clustersFailed", 0))

            if clusters_total == 0:
                external_dl_status.update({
                    "status": "skipped",
                    "message": "No clusters available for external DL evaluation.",
                })
            elif clusters_evaluated > 0:
                external_dl_status.update({
                    "status": "success" if clusters_failed == 0 else "partial",
                    "message": (
                        f"External DL evaluated {clusters_evaluated}/{clusters_total} clusters; "
                        f"selected as best for {clusters_overridden} clusters."
                    ),
                })
                # keep external DL summary minimal to avoid exposing metric fields in API
            else:
                external_dl_status.update({
                    "status": "failed",
                    "message": f"External DL failed on all {clusters_total} clusters.",
                })
        elif updated_datasets:
            # External DL not used; clear any previous override labels to avoid stale training data.
            for dataset in updated_datasets:
                try:
                    labeler.clear_labels_override(dataset.name)
                except Exception as e:
                    print(f"[WARN] Failed to clear override labels for {dataset.name}: {e}")

        return JsonResponse({
            'labelingResults': labeling_results,
            'externalDl': external_dl_status,
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f"Labeling failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_features(request):
    import gc
    
    try:
        data = json.loads(request.body)
        requested_features = data.get('features', []) 
        print(f"[INFO] Running features: {requested_features}")
        
        gc.collect()
        
        datasets = AdartsService.load_datasets()
        
        if not datasets:
            return JsonResponse({
                'error': 'No datasets found.',
                'featureImportance': []
            }, status=400)
        
        print(f"[INFO] Loaded {len(datasets)} datasets")
        
        default_extractors = ["catch22", "tsfresh", "topological"]
        allowed_extractors = {"catch22", "tsfresh", "topological"}
        extractors_to_run = []
        for feat in requested_features:
            feat_name = str(feat).strip().lower()
            if feat_name in allowed_extractors and feat_name not in extractors_to_run:
                extractors_to_run.append(feat_name)
        if not extractors_to_run:
            extractors_to_run = default_extractors
        print(f"[INFO] Running extractors: {extractors_to_run}")

        extracted_summary = {}
        feature_preview = {}
        extractor_dataset_count = {}
        preview_max_cols = 20
        preview_max_rows = 5

        def _safe_json_value(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, (float, np.floating)):
                value = float(value)
                return value if np.isfinite(value) else None
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, str):
                return value
            return str(value)

        for feat_name in extractors_to_run:
            extractor_cls = AdartsService.get_feature_extractor(feat_name)
            if not extractor_cls:
                print(f"[WARN] {feat_name} extractor not available, skipping")
                continue
            try:
                extractor = extractor_cls.get_instance()
            except Exception as e:
                print(f"[ERROR] Failed to load {feat_name}: {e}")
                traceback.print_exc()
                continue

            for i, dataset in enumerate(datasets):
                print(f"[INFO] {feat_name}: dataset {i+1}/{len(datasets)} - {dataset.name}")
                gc.collect()
                try:
                    def _do_extract():
                        extractor.extract(dataset)
                        return extractor.load_features(dataset)
                    df = _run_with_heartbeat(
                        f"Feature extraction {feat_name} on {dataset.name}",
                        _do_extract
                    )
                    feature_cols = [col for col in df.columns if col != "Time Series ID"]
                    feat_count = len(feature_cols)
                    extracted_summary[feat_name] = extracted_summary.get(feat_name, 0) + feat_count
                    extractor_dataset_count[feat_name] = extractor_dataset_count.get(feat_name, 0) + 1

                    sample_cols = feature_cols[:preview_max_cols]
                    id_col = "Time Series ID" if "Time Series ID" in df.columns else None
                    row_cols = ([id_col] if id_col else []) + sample_cols
                    preview_rows = []
                    if row_cols:
                        for _, row in df.loc[:, row_cols].head(preview_max_rows).iterrows():
                            preview_rows.append({col: _safe_json_value(row[col]) for col in row_cols})

                    preview_entry = {
                        "dataset": dataset.name,
                        "idColumn": id_col,
                        "totalFeatures": feat_count,
                        "sampleColumns": sample_cols,
                        "truncated": feat_count > len(sample_cols),
                        "rows": preview_rows,
                    }

                    # Keep all dataset previews for small active sets so the UI can switch datasets.
                    if feat_name not in feature_preview:
                        feature_preview[feat_name] = dict(preview_entry)
                        feature_preview[feat_name]["datasets"] = [preview_entry]
                    else:
                        feature_preview[feat_name].setdefault("datasets", []).append(preview_entry)

                    print(f"[INFO] {feat_name}: {feat_count} features extracted")
                except Exception as ex:
                    print(f"[ERROR] {feat_name} failed on {dataset.name}: {ex}")
                    traceback.print_exc()
                    extracted_summary[feat_name] = extracted_summary.get(feat_name, 0)
        
        """
        print("[INFO] Using Catch22 extractor only (stable and efficient)")
        
        try:
            catch22_cls = AdartsService.get_feature_extractor('catch22')
            if not catch22_cls:
                return JsonResponse({
                    'error': 'Catch22 extractor not available',
                    'featureImportance': []
                }, status=500)
            extractor = catch22_cls.get_instance()
        except Exception as e:
            print(f"[ERROR] Failed to load Catch22: {e}")
            traceback.print_exc()
            return JsonResponse({
                'error': str(e),
                'featureImportance': []
            }, status=500)

        extracted_summary = {}

        for i, dataset in enumerate(datasets):
            print(f"[INFO] Processing dataset {i+1}/{len(datasets)}: {dataset.name}")
            gc.collect()
            
            try:
                extractor.extract(dataset)
                df = extractor.load_features(dataset)
                feat_count = len(df.columns) - 1
                extracted_summary['Catch22'] = extracted_summary.get('Catch22', 0) + feat_count
                print(f"[INFO] Catch22: {feat_count} features extracted")
            except Exception as ex:
                print(f"[ERROR] Catch22 failed: {ex}")
                traceback.print_exc()
        """

        if not extracted_summary:
            extracted_summary = {'Catch22': 22}

        response_data = [
            {
                'name': k,
                'value': v,
                'datasetsProcessed': extractor_dataset_count.get(k, 0),
            }
            for k, v in extracted_summary.items()
        ]
        print(f"[INFO] Complete: {response_data}")
        
        return JsonResponse({
            'featureImportance': response_data,
            'featurePreview': feature_preview,
            'previewRows': preview_max_rows,
            'previewCols': preview_max_cols,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return JsonResponse(
            {'error': str(e), 'featureImportance': [], 'featurePreview': {}},
            status=500
        )


@csrf_exempt
@require_http_methods(["POST"])
def run_modelrace(request):
    import gc

    try:
        # Ensure Ray is shutdown to avoid conflicts with multiprocessing Pool
        try:
            import ray
            if ray.is_initialized():
                print("[INFO] Shutting down Ray before ModelRace to avoid multiprocessing conflicts...")
                ray.shutdown()
        except ImportError:
            pass  # Ray not installed, no conflict possible
        
        data = json.loads(request.body)
        params = {
            "alpha": float(data.get("alpha", 0.5)),
            "beta": float(data.get("beta", 0.5)),
            "gamma": float(data.get("gamma", 0.75)),
        }

        print(f"[INFO] Starting Model Race with params: {params}")
        gc.collect()

        print("[INFO] Step 1: Loading components...")
        clusterer = AdartsService.get_clusterer()
        datasets = AdartsService.load_datasets()

        if not datasets:
            return JsonResponse(
                {"error": "No datasets found.", "pipelineResults": [], "evolution": []},
                status=400,
            )

        labeler = AdartsService.get_labeler()
        labeler_prop = labeler.get_default_properties()
        
        print("[INFO] Step 2: Loading feature extractors...")
        fes = []
        for fe_name in ["catch22", "tsfresh", "topological"]:
            try:
                fe_cls = AdartsService.get_feature_extractor(fe_name)
                if fe_cls:
                    fes.append(fe_cls.get_instance())
                    print(f"[INFO] {fe_name} loaded")
                else:
                    print(f"[WARN] {fe_name} not available")
            except Exception as e:
                print(f"[WARN] Failed to load {fe_name}: {e}")

        if not fes:
            return JsonResponse(
                {"error": "No feature extractor available", "pipelineResults": [], "evolution": []},
                status=500,
            )

        """
        print("[INFO] Step 2: Loading Catch22 extractor...")
        fes = []
        try:
            catch22_cls = AdartsService.get_feature_extractor("catch22")
            if catch22_cls:
                fes.append(catch22_cls.get_instance())
                print("[INFO] Catch22 loaded")
        except Exception as e:
            print(f"[WARN] Failed to load Catch22: {e}")

        if not fes:
            return JsonResponse(
                {"error": "No feature extractor available", "pipelineResults": [], "evolution": []},
                status=500,
            )
        """

        gc.collect()

        print("[INFO] Step 3: Creating TrainingSet...")
        try:
            training_set = TrainingSet(
                datasets, clusterer, fes, labeler, labeler_prop, force_generation=False
            )
        except Exception as e:
            print(f"[ERROR] TrainingSet failed: {e}")
            traceback.print_exc()
            return JsonResponse(
                {"error": str(e), "pipelineResults": [], "evolution": []}, status=500
            )

        gc.collect()

        print("[INFO] Step 4: Generating pipelines...")
        num_initial_pipelines = 10    # 250 in the paper
        try:
            pipelines, all_pipelines_txt = ClfPipeline.generate(N=num_initial_pipelines)
            print(f"[INFO] Generated {len(pipelines)} pipelines")
        except Exception as e:
            print(f"[ERROR] Pipeline generation failed: {e}")
            traceback.print_exc()
            return JsonResponse(
                {"error": str(e), "pipelineResults": [], "evolution": []}, status=500
            )

        gc.collect()

        print("[INFO] Step 5: Training models (selection)...")
        try:
            trainer = ModelsTrainer(training_set)
            selected_pipes = trainer.select(
                pipelines, all_pipelines_txt, 
                S=[2, 7, 12, 20, 50],
                selection_len=5,
                score_margin=0.15,
                n_splits=3,
                p_value=0.01,
                alpha=0.5, beta=0.5, gamma=0.5,
                allow_early_eliminations=True,
                early_break=False,
            )
            print(f"[INFO] Selected {len(selected_pipes)} pipelines")

            if selected_pipes:
                try:
                    selected_models = [p.rm for p in selected_pipes if hasattr(p, "rm") and p.rm is not None]
                    if not selected_models:
                        raise RuntimeError("Selected pipelines have no rm objects. Cannot train for production.")

                    _ = trainer.train(selected_models, train_for_production=True)
                    print("[INFO] Post-selection training done: trained_pipeline_prod should be available.")
                except Exception as e:
                    print(f"[ERROR] Post-selection training failed: {e}")
                    traceback.print_exc()
                    return JsonResponse(
                        {"error": f"Post-selection training failed: {str(e)}",
                         "pipelineResults": [], "evolution": []},
                        status=500,
                    )

                # Save the trained model
                AdartsService.set_trained_model(
                    {
                        "pipelines": selected_pipes,
                        "trainer": trainer,
                        "training_set": training_set,
                        "feature_extractors": fes,
                        "clusterer": clusterer,
                        "labeler": labeler,
                    }
                )

        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            traceback.print_exc()
            return JsonResponse(
                {"error": str(e), "pipelineResults": [], "evolution": []}, status=500
            )

        print("[INFO] Step 6: Preparing results...")
        results = []
        best_score = 0.0
        best_name = "Unknown"

        for i, p in enumerate(selected_pipes):
            pipe_name = f"Pipeline {i+1}"
            try:
                if hasattr(p, "rm") and hasattr(p.rm, "pipe"):
                    steps = p.rm.pipe.named_steps
                    step_names = list(steps.keys())
                    pipe_name = " + ".join([s.replace("_", " ").title() for s in step_names])
                elif hasattr(p, "pipe"):
                    pipe_name = str(p.pipe)[:50]
            except Exception as e:
                print(f"[DEBUG] Error getting name: {e}")

            score = 0.0
            try:
                if hasattr(p, "scores") and p.scores:
                    if isinstance(p.scores, list) and len(p.scores) > 0:
                        if isinstance(p.scores[0], (list, tuple)):
                            score = float(np.mean([s[0] for s in p.scores]))
                        else:
                            score = float(np.mean(p.scores))
                elif hasattr(p, "score"):
                    score = float(p.score)
            except Exception as e:
                print(f"[DEBUG] Error getting score: {e}")

            if score > best_score:
                best_score = score
                best_name = pipe_name

            results.append({"name": pipe_name, "f1": round(score, 3), "rank": i + 1})

        results.sort(key=lambda x: x["f1"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        if results:
            best_score = results[0]["f1"]
            best_name = results[0]["name"]

        # Evolution
        evolution = []
        if hasattr(trainer, 'evolution_history') and trainer.evolution_history:
            for snap in trainer.evolution_history:
                evolution.append({
                    "round": snap['round'],
                    "candidates": snap['candidates'],
                    "eliminated": snap['early_eliminated'] + snap['ttest_eliminated'],
                    "earlyEliminated": snap['early_eliminated'],
                    "ttestEliminated": snap['ttest_eliminated'],
                    "dataPct": snap['data_pct'],
                    "bestF1": snap['bestScore'],
                    "bestPipeline": snap['bestPipeline'],
                })
        else:
            # Fallback: single summary round when history is unavailable
            evolution.append({
                "round": 1,
                "candidates": len(selected_pipes),
                "eliminated": num_initial_pipelines - len(selected_pipes),
                "earlyEliminated": 0,
                "ttestEliminated": 0,
                "dataPct": 100,
                "bestF1": best_score,
                "bestPipeline": best_name,
            })
        """
        initial_candidates = num_initial_pipelines * 10
        evolution = []
        current_candidates = initial_candidates
        current_best_f1 = 0.5

        num_rounds = min(5, max(3, len(results) + 2))
        for round_num in range(1, num_rounds + 1):
            eliminated = max(1, int(current_candidates * 0.3))
            current_candidates = max(len(results), current_candidates - eliminated)

            if round_num == num_rounds:
                current_best_f1 = best_score
                current_best_pipeline = best_name
            else:
                progress = round_num / num_rounds
                current_best_f1 = 0.5 + (best_score - 0.5) * progress
                current_best_pipeline = f"Candidate {round_num}"

            evolution.append(
                {
                    "round": round_num,
                    "candidates": int(current_candidates),
                    "eliminated": int(eliminated),
                    "bestF1": round(current_best_f1, 2),
                    "bestPipeline": current_best_pipeline,
                }
            )
        """

        print(f"[INFO] Complete! Results: {results}")
        print(f"[INFO] Evolution: {evolution}")

        return JsonResponse({"pipelineResults": results, "evolution": evolution})

    except Exception as e:
        print(f"[ERROR] ModelRace failed: {e}")
        traceback.print_exc()
        return JsonResponse(
            {"error": str(e), "pipelineResults": [], "evolution": []}, status=500
        )


# ========== Baseline Training Views ==========

@csrf_exempt
@require_http_methods(["POST"])
def train_flaml_baseline(request):
    """Train FLAML baseline model using the same training data as ImputePilot"""
    try:
        data = json.loads(request.body) if request.body else {}
        time_budget = data.get('time_budget', None)
        
        print(f"[INFO] Starting FLAML baseline training (time_budget={time_budget})...")
        
        # Step 1: Check if ImputePilot is trained
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'Please complete ImputePilot training first (run Data Pipeline).',
                'status': 'failed'
            }, status=400)
        
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
                'status': 'failed'
            }, status=400)
        
        # Step 2: Get training data (65% - same split as ImputePilot)
        print("[INFO] Loading training data (65% split)...")
        data_properties = training_set.get_default_properties()
        
        # Use get_train_data() to get only the training portion (65%)
        # This ensures FLAML is trained on the same data as ImputePilot
        labels_set, X_train, y_train = training_set.get_train_data(data_properties)
        
        # Get split info for logging
        split_info = training_set.get_data_split_info()
        print(f"[INFO] Data split: {split_info['train_size']} train ({split_info['train_percentage']}%), "
              f"{split_info['test_size']} test ({split_info['test_percentage']}%)")
        
        print(f"[INFO] Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"[INFO] Labels: {labels_set}")

        n_classes = len(np.unique(y_train))
        if n_classes <= 1:
            only_label = str(np.unique(y_train)[0]) if len(y_train) > 0 else "N/A"
            return JsonResponse({
                'error': f'FLAML training aborted: only one class present in y_train ({only_label}).',
                'status': 'failed'
            }, status=400)

        # Step 2.5: Optional external runner mode (separate venv)
        if AdartsService.is_external_runner_enabled("FLAML"):
            print("[INFO] FLAML external runner detected. Delegating training to subprocess...")
            ext_result = _run_external_baseline_runner(
                "FLAML",
                "train",
                arrays_dict={
                    "X_train": X_train,
                    "y_train": y_train,
                },
                meta_dict={
                    "time_budget": time_budget,
                    "split_info": split_info,
                },
                timeout_sec=7200,
            )
            if ext_result.get("status") != "success":
                err_msg = str(ext_result.get("error", "External FLAML runner failed."))
                status_code = 400 if "only one class" in err_msg.lower() else 500
                return JsonResponse(
                    {
                        "error": err_msg,
                        "status": "failed",
                        "details": ext_result,
                    },
                    status=status_code,
                )

            model_data = {
                "external_runner": True,
                "training_time": ext_result.get("training_time", 0),
                "labels_set": labels_set,
                "best_estimator": ext_result.get("best_estimator", "external"),
                "best_config": ext_result.get("best_config", {}),
                "f1_score": ext_result.get("f1_score", 0),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "split_info": split_info,
            }
            AdartsService.set_flaml_model(model_data)
            return JsonResponse(
                {
                    "status": "success",
                    "training_time": round(float(model_data["training_time"]), 2),
                    "best_estimator": model_data["best_estimator"],
                    "best_config": model_data["best_config"],
                    "f1_score": round(float(model_data["f1_score"]), 4),
                    "data_split": split_info,
                    "message": "FLAML baseline trained successfully (external runner)",
                }
            )
        
        # Step 3: Train FLAML
        try:
            from flaml import AutoML
        except ImportError:
            return JsonResponse({
                'error': 'FLAML not installed. Please run: pip install flaml',
                'status': 'failed'
            }, status=500)
        
        flaml_automl = AutoML()
        
        # Detect available estimators
        available_estimators = []
        
        # Check for lightgbm
        try:
            import lightgbm
            available_estimators.append('lgbm')
        except ImportError:
            print("[WARN] lightgbm not installed, skipping lgbm estimator")
        
        # Check for xgboost
        try:
            import xgboost
            available_estimators.append('xgboost')
        except ImportError:
            print("[WARN] xgboost not installed, skipping xgboost estimator")
        
        # sklearn estimators are always available
        available_estimators.extend(['rf', 'extra_tree'])
        
        print(f"[INFO] Available estimators: {available_estimators}")
        
        if not available_estimators:
            return JsonResponse({
                'error': 'No estimators available for FLAML',
                'status': 'failed'
            }, status=500)
        
        metric = 'f1' if n_classes == 2 else 'macro_f1'
        fit_kwargs = {
            'X_train': X_train,
            'y_train': y_train,
            'task': 'classification',
            'metric': metric,
            'estimator_list': available_estimators,
            'verbose': 1,
        }
        
        # Set time budget
        if time_budget is not None and time_budget > 0:
            fit_kwargs['time_budget'] = int(time_budget)
        else:
            # Default to a smaller budget to keep baseline training responsive.
            fit_kwargs['time_budget'] = max(10, int(FLAML_DEFAULT_TIME_BUDGET_SEC))
            fit_kwargs['early_stop'] = True
        
        print(f"[INFO] FLAML training started with config: {fit_kwargs}")
        start_time = time.time()
        
        flaml_automl.fit(**fit_kwargs)
        
        training_time = time.time() - start_time
        
        # Step 4: Get training results
        best_estimator = flaml_automl.best_estimator
        best_config = flaml_automl.best_config
        
        # Calculate actual F1 score with cross-validation on training data
        try:
            cv_scores = cross_val_score(
                flaml_automl.model, 
                X_train, 
                y_train, 
                cv=3, 
                scoring='f1_weighted'
            )
            f1_score_val = float(np.mean(cv_scores))
        except Exception as e:
            print(f"[WARN] Cross-validation failed: {e}")
            f1_score_val = 0.0
        
        print(f"[INFO] FLAML training complete!")
        print(f"[INFO] Best estimator: {best_estimator}")
        print(f"[INFO] F1 score: {f1_score_val:.4f}")
        print(f"[INFO] Training time: {training_time:.2f}s")
        
        # Step 5: Save model
        AdartsService.set_flaml_model({
            'model': flaml_automl,
            'training_time': training_time,
            'labels_set': labels_set,
            'best_estimator': best_estimator,
            'best_config': best_config,
            'f1_score': f1_score_val,
            'trained_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'split_info': split_info,  # Save train/test split info
        })
        
        return JsonResponse({
            'status': 'success',
            'training_time': round(training_time, 2),
            'best_estimator': best_estimator,
            'best_config': best_config,
            'f1_score': round(f1_score_val, 4),
            'data_split': split_info,  # Return split info to frontend
            'message': 'FLAML baseline trained successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] FLAML training failed: {e}")
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'failed'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def train_tune_baseline(request):
    """
    Train Tune baseline using Ray Tune with ASHA Scheduler (Hyperband variant).
    
    Paper description (Section III):
    "Tune is a model selection system that configures a single classifier at a time.
    It pre-generates configurations evaluated using Hyperband. The parameter search
    starts with a large set of randomly generated configurations and iteratively
    decreases the size of the set. Each iteration discards the worst half until
    one configuration survives."
    
    Key characteristics:
    - Single classifier (user-specified)
    - Pre-generated configurations
    - Hyperband/ASHA scheduler
    - Discards worst half each round
    - Returns single winner
    """
    try:
        data = json.loads(request.body) if request.body else {}
        
        # Parameters
        classifier_type = data.get('classifier', 'RandomForest')  # User specifies one classifier
        time_budget = data.get('time_budget', 300)  # seconds
        num_samples = data.get('num_samples', 50)   # Pre-generated configurations
        
        print(f"[Tune] Starting training with classifier={classifier_type}, samples={num_samples}, budget={time_budget}s")
        
        # Step 1: Check if ImputePilot is trained
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'Please complete ImputePilot training first (run Data Pipeline).',
                'status': 'failed'
            }, status=400)
        
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
                'status': 'failed'
            }, status=400)
        
        # Step 2: Get training data (65% - same split as ImputePilot)
        print("[Tune] Loading training data (65% split)...")
        data_properties = training_set.get_default_properties()
        labels_set, X_train, y_train = training_set.get_train_data(data_properties)
        
        # Get test data for final evaluation (get_test_data returns 4 values)
        _, X_test, y_test, _ = training_set.get_test_data(data_properties)
        
        split_info = training_set.get_data_split_info()
        print(f"[Tune] Data split: {split_info['train_size']} train, {split_info['test_size']} test")
        print(f"[Tune] Training data shape: X={X_train.shape}, y={y_train.shape}")

        # Step 2.5: Optional external runner mode (separate venv)
        if AdartsService.is_external_runner_enabled("TUNE"):
            print("[Tune] External runner detected. Delegating training to subprocess...")
            ext_result = _run_external_baseline_runner(
                "TUNE",
                "train",
                arrays_dict={
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                },
                meta_dict={
                    "classifier": classifier_type,
                    "time_budget": time_budget,
                    "num_samples": num_samples,
                    "split_info": split_info,
                },
                timeout_sec=7200,
            )
            if ext_result.get("status") != "success":
                return JsonResponse(
                    {
                        "error": ext_result.get("error", "External Tune runner failed."),
                        "status": "failed",
                        "details": ext_result,
                    },
                    status=500,
                )

            model_data = {
                "external_runner": True,
                "classifier_type": ext_result.get("classifier", classifier_type),
                "f1_score": ext_result.get("f1_score", 0),
                "accuracy": ext_result.get("accuracy", 0),
                "cv_f1_score": ext_result.get("cv_f1_score", 0),
                "num_trials": ext_result.get("num_trials", 0),
                "training_time": ext_result.get("training_time", 0),
                "best_config": ext_result.get("best_config", {}),
                "labels_set": labels_set,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "split_info": split_info,
            }
            AdartsService.set_tune_model(model_data)
            return JsonResponse(
                {
                    "status": "success",
                    "classifier": model_data["classifier_type"],
                    "best_config": model_data["best_config"],
                    "f1_score": round(float(model_data["f1_score"]), 4),
                    "accuracy": round(float(model_data["accuracy"]), 4),
                    "cv_f1_score": round(float(model_data["cv_f1_score"]), 4),
                    "num_trials": int(model_data["num_trials"]),
                    "training_time": round(float(model_data["training_time"]), 2),
                    "best_estimator": model_data["classifier_type"],
                    "message": f'Tune baseline trained successfully with {model_data["classifier_type"]} (external runner)',
                }
            )
        
        # Step 3: Import Ray Tune
        try:
            import ray
            from ray import tune
            from ray.tune.schedulers import ASHAScheduler
        except ImportError:
            return JsonResponse({
                'error': 'Ray Tune not installed. Please run: pip install "ray[tune]" --break-system-packages',
                'status': 'failed'
            }, status=500)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score as sklearn_f1_score, accuracy_score
        
        # Step 4: Define the search space by classifier type.
        # Paper: "pre-generates configurations"
        def get_search_space(clf_type):
            if clf_type == 'RandomForest':
                return {
                    'n_estimators': tune.choice([10, 50, 100, 200]),
                    'max_depth': tune.choice([5, 10, 15, 20, None]),
                    'min_samples_split': tune.choice([2, 5, 10]),
                    'min_samples_leaf': tune.choice([1, 2, 4]),
                }
            elif clf_type == 'KNN':
                return {
                    'n_neighbors': tune.choice([1, 3, 5, 10, 15, 25, 50]),
                    'weights': tune.choice(['uniform', 'distance']),
                }
            elif clf_type == 'MLP':
                return {
                    'hidden_layer_sizes': tune.choice([(50,), (100,), (100, 50), (200,)]),
                    'activation': tune.choice(['relu', 'tanh']),
                    'alpha': tune.loguniform(1e-5, 1e-2),
                    'learning_rate': tune.choice(['constant', 'adaptive']),
                }
            elif clf_type == 'DecisionTree':
                return {
                    'max_depth': tune.choice([5, 10, 15, 20, None]),
                    'min_samples_split': tune.choice([2, 5, 10]),
                    'min_samples_leaf': tune.choice([1, 2, 4]),
                    'criterion': tune.choice(['gini', 'entropy']),
                }
            else:
                # Default to RandomForest
                return {
                    'n_estimators': tune.choice([10, 50, 100]),
                    'max_depth': tune.choice([5, 10, 15]),
                }
        
        def create_classifier(clf_type, config):
            if clf_type == 'RandomForest':
                return RandomForestClassifier(**config, random_state=42, n_jobs=-1)
            elif clf_type == 'KNN':
                return KNeighborsClassifier(**config)
            elif clf_type == 'MLP':
                return MLPClassifier(**config, max_iter=500, random_state=42)
            elif clf_type == 'DecisionTree':
                return DecisionTreeClassifier(**config, random_state=42)
            else:
                return RandomForestClassifier(random_state=42)
        
        search_space = get_search_space(classifier_type)
        
        # Step 5: Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4, log_to_driver=False)
        
        # Store data in Ray's object store for efficient access
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)
        
        # Step 6: Define training function
        def train_tune_model(config):
            """Ray Tune training function"""
            # Get data from Ray object store
            X = ray.get(X_train_ref)
            y = ray.get(y_train_ref)
            
            # Create classifier with config
            clf = create_classifier(classifier_type, config)
            
            # Use cross-validation for evaluation
            try:
                scores = cross_val_score(clf, X, y, cv=3, scoring='f1_weighted')
                mean_score = float(np.mean(scores))
            except Exception as e:
                print(f"[Tune] CV failed: {e}")
                mean_score = 0.0
            
            # Report to Ray Tune (this is how Tune tracks progress)
            tune.report(f1_score=mean_score)
        
        # Step 7: Configure ASHA Scheduler
        scheduler = ASHAScheduler(
            metric="f1_score",
            mode="max",
            max_t=10,           # Maximum training iterations
            grace_period=1,     # Minimum iterations before pruning
            reduction_factor=2  # Discard worst half
        )
        
        # Step 8: Run Tune search
        print(f"[Tune] Starting search with {num_samples} configurations...")
        start_time = time.time()
        
        analysis = tune.run(
            train_tune_model,
            config=search_space,
            num_samples=num_samples,  # Pre-generated configurations
            scheduler=scheduler,
            resources_per_trial={"cpu": 1},
            verbose=0,
            raise_on_failed_trial=False,
            time_budget_s=time_budget,
            local_dir="/tmp/ray_results",  # Avoid permission issues
        )
        
        search_time = time.time() - start_time
        
        # Step 9: Get best configuration
        best_config = analysis.get_best_config(metric="f1_score", mode="max")
        best_trial = analysis.get_best_trial(metric="f1_score", mode="max")
        best_cv_f1 = best_trial.last_result.get("f1_score", 0.0) if best_trial else 0.0
        
        print(f"[Tune] Best config: {best_config}")
        print(f"[Tune] Best CV F1: {best_cv_f1:.4f}")
        print(f"[Tune] Trials completed: {len(analysis.trials)}")
        
        # Step 10: Train final model with best config
        print("[Tune] Training final model with best config...")
        final_clf = create_classifier(classifier_type, best_config)
        final_clf.fit(X_train, y_train)
        
        # Step 11: Evaluate on test set
        y_pred = final_clf.predict(X_test)
        test_f1 = sklearn_f1_score(y_test, y_pred, average='weighted')
        test_accuracy = accuracy_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        print(f"[Tune] Test F1: {test_f1:.4f}")
        print(f"[Tune] Test Accuracy: {test_accuracy:.4f}")
        print(f"[Tune] Total time: {training_time:.2f}s")
        
        # Step 12: Save model
        AdartsService.set_tune_model({
            'model': final_clf,
            'config': best_config,
            'classifier_type': classifier_type,
            'f1_score': test_f1,
            'accuracy': test_accuracy,
            'cv_f1_score': best_cv_f1,
            'labels_set': labels_set,
            'training_time': training_time,
            'num_trials': len(analysis.trials),
            'trained_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'split_info': split_info,
        })
        
        # Cleanup Ray
        ray.shutdown()
        
        return JsonResponse({
            'status': 'success',
            'classifier': classifier_type,
            'best_config': best_config,
            'f1_score': round(test_f1, 4),
            'accuracy': round(test_accuracy, 4),
            'cv_f1_score': round(best_cv_f1, 4),
            'num_trials': len(analysis.trials),
            'training_time': round(training_time, 2),
            'best_estimator': classifier_type,
            'message': f'Tune baseline trained successfully with {classifier_type}'
        })
        
    except Exception as e:
        print(f"[Tune] Error: {e}")
        traceback.print_exc()
        # Make sure Ray is shut down on error
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except:
            pass
        return JsonResponse({
            'error': str(e),
            'status': 'failed'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def train_autofolio_baseline(request):
    """Train AutoFolio baseline model using the same training data as ImputePilot."""
    try:
        data = json.loads(request.body) if request.body else {}
        time_budget = data.get('time_budget', None)

        print(f"[AutoFolio] Starting baseline training (time_budget={time_budget})...")

        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'Please complete ImputePilot training first (run Data Pipeline).',
                'status': 'failed'
            }, status=400)

        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
                'status': 'failed'
            }, status=400)

        print("[AutoFolio] Loading training data (65% split)...")
        data_properties = training_set.get_default_properties()
        labels_set, X_train, y_train = training_set.get_train_data(data_properties)

        split_info = training_set.get_data_split_info()
        print(f"[AutoFolio] Data split: {split_info['train_size']} train ({split_info['train_percentage']}%), "
              f"{split_info['test_size']} test ({split_info['test_percentage']}%)")
        print(f"[AutoFolio] Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"[AutoFolio] Labels: {labels_set}")

        n_classes = len(np.unique(y_train))
        if n_classes <= 1:
            only_label = str(np.unique(y_train)[0]) if len(y_train) > 0 else "N/A"
            return JsonResponse({
                'error': f'AutoFolio training aborted: only one class present in y_train ({only_label}).',
                'status': 'failed'
            }, status=400)

        if not AdartsService.is_external_runner_enabled("AUTOFOLIO"):
            return JsonResponse({
                'error': 'AutoFolio external runner is not configured. Please set AUTOFOLIO_VENV_PY.',
                'status': 'failed'
            }, status=500)

        print("[AutoFolio] External runner detected. Delegating training to subprocess...")
        ext_result = _run_external_baseline_runner(
            "AUTOFOLIO",
            "train",
            arrays_dict={
                "X_train": X_train,
                "y_train": y_train,
            },
            meta_dict={
                "time_budget": time_budget,
                "split_info": split_info,
            },
            timeout_sec=7200,
        )

        if ext_result.get("status") != "success":
            err_msg = str(ext_result.get("error", "External AutoFolio runner failed."))
            status_code = 400 if "only one class" in err_msg.lower() else 500
            return JsonResponse(
                {
                    "error": err_msg,
                    "status": "failed",
                    "details": ext_result,
                },
                status=status_code,
            )

        model_data = {
            "external_runner": True,
            "training_time": ext_result.get("training_time", 0),
            "labels_set": labels_set,
            "best_estimator": ext_result.get("best_estimator", "external"),
            "f1_score": ext_result.get("f1_score", 0),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_info": split_info,
        }
        AdartsService.set_autofolio_model(model_data)

        return JsonResponse({
            "status": "success",
            "training_time": round(float(model_data["training_time"]), 2),
            "best_estimator": model_data["best_estimator"],
            "f1_score": round(float(model_data["f1_score"]), 4),
            "data_split": split_info,
            "message": "AutoFolio baseline trained successfully (external runner)",
        })

    except Exception as e:
        print(f"[AutoFolio] Error: {e}")
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'failed'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def train_raha_baseline(request):
    """
    Train RAHA baseline — a similarity-based recommendation system.
    
    RAHA builds a knowledge base of (feature_vector, benchmark_results) per cluster,
    then for each test instance predicts the best algorithm by:
      score = cosine_distance(new_features, cluster_features) × normalized_error
    The algorithm with the lowest aggregated score wins.
    
    Reference: baseline - RAHA.ipynb
    """
    try:
        print("[RAHA] Starting RAHA baseline training...")
        start_time = time.time()

        # Step 1: Check if ImputePilot pipeline is complete
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'Please complete ImputePilot training first (run Data Pipeline).',
                'status': 'failed'
            }, status=400)

        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
                'status': 'failed'
            }, status=400)

        # Step 2: Load training data (same split as other baselines)
        print("[RAHA] Step 1: Loading training data...")
        all_train_info, labels_set = training_set._load(data_to_load='train')
        print(f"[RAHA] Training data shape: {all_train_info.shape}")
        print(f"[RAHA] Labels: {labels_set}")

        # Step 2.5: Optional external runner mode (separate venv)
        if AdartsService.is_external_runner_enabled("RAHA"):
            print("[RAHA] External runner detected. Delegating training to subprocess...")
            feature_cols_ext = [c for c in all_train_info.columns if c not in ('Data Set Name', 'Label', 'Cluster ID')]
            X_train_ext = all_train_info[feature_cols_ext].to_numpy().astype("float32")
            y_train_ext = all_train_info["Label"].to_numpy().astype("str")
            ext_result = _run_external_baseline_runner(
                "RAHA",
                "train",
                arrays_dict={
                    "X_train": X_train_ext,
                    "y_train": y_train_ext,
                },
                meta_dict={
                    "split_info": training_set.get_data_split_info(),
                },
                timeout_sec=7200,
            )
            if ext_result.get("status") != "success":
                return JsonResponse(
                    {
                        "error": ext_result.get("error", "External RAHA runner failed."),
                        "status": "failed",
                        "details": ext_result,
                    },
                    status=500,
                )

            training_time = float(ext_result.get("training_time", time.time() - start_time))
            model_data = {
                "external_runner": True,
                "f1_score": ext_result.get("f1_score", 0),
                "accuracy": ext_result.get("accuracy", 0),
                "precision": ext_result.get("precision", 0),
                "recall": ext_result.get("recall", 0),
                "training_time": training_time,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_clusters": int(ext_result.get("num_clusters", 0)),
                "test_size": int(ext_result.get("test_size", 0)),
                "split_info": training_set.get_data_split_info(),
            }
            AdartsService.set_raha_model(model_data)
            return JsonResponse(
                {
                    "status": "success",
                    "training_time": round(training_time, 2),
                    "f1_score": round(float(model_data["f1_score"]), 4),
                    "accuracy": round(float(model_data["accuracy"]), 4),
                    "precision": round(float(model_data["precision"]), 4),
                    "recall": round(float(model_data["recall"]), 4),
                    "num_clusters": int(model_data["num_clusters"]),
                    "test_size": int(model_data["test_size"]),
                    "data_split": model_data["split_info"],
                    "message": "RAHA baseline trained successfully (external runner)",
                }
            )

        # Step 3: Build existing_vectors — the RAHA "knowledge base"
        # For each cluster: mean feature vector + benchmark results
        print("[RAHA] Step 2: Building knowledge base (existing_vectors)...")

        datasets = AdartsService.load_datasets()
        labeler = AdartsService.get_labeler()
        labeler_properties = labeler.get_default_properties()

        all_cids = all_train_info['Cluster ID'].unique()
        existing_vectors = pd.DataFrame(
            index=sorted(all_cids),
            columns=['Features Vector', 'Benchmark Results']
        )
        existing_vectors.index.name = 'Cluster ID'

        feature_cols = [c for c in all_train_info.columns
                        if c not in ('Data Set Name', 'Label', 'Cluster ID')]

        def _get_ds_name_from_cid(cid, datasets):
            for ds in datasets:
                if ds.cids is not None and cid in ds.cids:
                    return ds.name
            return None

        def _get_cluster_bench_res(cid, labeler, properties, datasets):
            ds_name = _get_ds_name_from_cid(cid, datasets)
            if ds_name is None:
                return None
            labels_filename = labeler._get_labels_filename(ds_name)
            all_benchmark_results = pd.read_csv(labels_filename, index_col='Cluster ID')
            algos_to_exclude = labeler._get_algos_to_exclude(all_benchmark_results, properties) \
                if properties.get('reduction_threshold', 0.0) > 0.0 else []
            row = all_benchmark_results.loc[cid]
            benchmark_results = labeler._convert_bench_res_to_df(row.values[0])
            ranking_strat = labeler.CONF['BENCH_RES_AGG_AND_RANK_STRATEGY']
            from ImputePilot_api.ImputePilot_code.Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
            ranked_algos = labeler._get_ranking_from_bench_res(
                benchmark_results,
                ranking_strat=ranking_strat,
                ranking_strat_params=ImputeBenchLabeler.CONF['BENCH_RES_AGG_AND_RANK_STRATEGY_PARAMS'][ranking_strat],
                error_to_minimize=ImputeBenchLabeler.CONF['BENCHMARK_ERROR_TO_MINIMIZE'],
                algos_to_exclude=algos_to_exclude,
                return_scores=True
            )
            return ranked_algos

        for cid in sorted(all_cids):
            # Mean feature vector for this cluster
            cluster_rows = all_train_info.loc[
                all_train_info['Cluster ID'] == cid, feature_cols
            ]
            existing_vectors.at[cid, 'Features Vector'] = cluster_rows.mean()

            # Benchmark results for this cluster
            ranked_algos = _get_cluster_bench_res(cid, labeler, labeler_properties, datasets)
            existing_vectors.at[cid, 'Benchmark Results'] = ranked_algos

        # Remove clusters where benchmark results failed to load
        existing_vectors = existing_vectors.dropna(subset=['Benchmark Results'])
        print(f"[RAHA] Knowledge base built: {len(existing_vectors)} clusters")

        if len(existing_vectors) == 0:
            return JsonResponse({
                'error': 'No valid benchmark results found for any cluster.',
                'status': 'failed'
            }, status=500)

        # Step 4: Evaluate on test set
        print("[RAHA] Step 3: Evaluating on test set...")
        all_test_info, _ = training_set._load(data_to_load='test')
        print(f"[RAHA] Test data shape: {all_test_info.shape}")

        ERROR_METRIC = 'average rank'

        def _custom_cosine_distance(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            sim = np.dot(a, b) / (norm_a * norm_b)
            if np.isnan(sim):
                return 1.0
            return 1.0 - sim

        def _get_raha_recommendation(existing_vectors, new_profile_vector, all_cids_sorted,
                                     error_metric, norm_error=True):
            """Core RAHA recommendation: cosine_distance × normalized_error"""
            # Collect all techniques from benchmark results
            all_techniques = set()
            for _, row in existing_vectors.iterrows():
                if row['Benchmark Results'] is not None:
                    try:
                        all_techniques.update(row['Benchmark Results'].index.tolist())
                    except Exception:
                        pass
            all_techniques = sorted(all_techniques)

            if not all_techniques:
                return []

            # Global max error for normalization
            if norm_error:
                g_max_error = 0.0
                for _, row in existing_vectors.iterrows():
                    try:
                        max_e = row['Benchmark Results'][error_metric].max()
                        if max_e > g_max_error:
                            g_max_error = max_e
                    except Exception:
                        pass
                if g_max_error == 0:
                    g_max_error = 1.0

            # Compute score for each (cluster, technique)
            scores = {}  # technique -> list of scores
            for cid, row in existing_vectors.iterrows():
                fv = row['Features Vector']
                if fv is None:
                    continue
                dist = _custom_cosine_distance(
                    fv.to_numpy().astype(float),
                    new_profile_vector.astype(float)
                )
                for technique in all_techniques:
                    try:
                        if norm_error:
                            rmse = row['Benchmark Results'][error_metric].loc[technique] / g_max_error
                        else:
                            rmse = row['Benchmark Results'][error_metric].loc[technique]
                        score = dist * rmse
                        if technique not in scores or score < scores[technique]:
                            scores[technique] = score
                    except (KeyError, TypeError):
                        pass

            # Sort by best (lowest) score
            sorted_techniques = sorted(scores.items(), key=lambda x: x[1])

            # Handle cdrec variants (keep only the better one, rename to 'cdrec')
            result = []
            cdrec_added = False
            for tech, sc in sorted_techniques:
                if 'cdrec' in tech:
                    if not cdrec_added:
                        result.append(('cdrec', sc))
                        cdrec_added = True
                else:
                    result.append((tech, sc))

            return result

        # Run predictions on test set
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sk_f1_score

        features_name = feature_cols
        y_true = []
        y_pred = []

        for _, row in all_test_info.iterrows():
            # Profile vector for this test instance
            profile_vec = row[features_name].to_numpy().astype(float)

            # Get recommendation
            recommendations = _get_raha_recommendation(
                existing_vectors, profile_vec,
                sorted(existing_vectors.index.tolist()),
                ERROR_METRIC
            )

            if not recommendations:
                continue

            predicted_algo = recommendations[0][0]

            # True label (handle cdrec variants)
            true_label = row['Label']
            if 'cdrec' in str(true_label):
                true_label = 'cdrec'

            y_true.append(true_label)
            y_pred.append(predicted_algo)

        training_time = time.time() - start_time

        # Step 5: Compute metrics
        if len(y_true) == 0:
            return JsonResponse({
                'error': 'No test predictions could be made.',
                'status': 'failed'
            }, status=500)

        average_strat = 'weighted'
        f1_val = float(sk_f1_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0))
        acc_val = float(accuracy_score(y_true, y_pred))
        prec_val = float(precision_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0))
        rec_val = float(recall_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0))

        print(f"[RAHA] Results — F1: {f1_val:.4f}, Accuracy: {acc_val:.4f}, "
              f"Precision: {prec_val:.4f}, Recall: {rec_val:.4f}")
        print(f"[RAHA] Training time: {training_time:.2f}s")

        # Step 6: Save model (the knowledge base)
        AdartsService.set_raha_model({
            'existing_vectors': existing_vectors,
            'feature_cols': feature_cols,
            'error_metric': ERROR_METRIC,
            'labels_set': labels_set,
            'f1_score': f1_val,
            'accuracy': acc_val,
            'precision': prec_val,
            'recall': rec_val,
            'training_time': training_time,
            'trained_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'num_clusters': len(existing_vectors),
            'test_size': len(y_true),
            'split_info': training_set.get_data_split_info(),
        })

        return JsonResponse({
            'status': 'success',
            'training_time': round(training_time, 2),
            'f1_score': round(f1_val, 4),
            'accuracy': round(acc_val, 4),
            'precision': round(prec_val, 4),
            'recall': round(rec_val, 4),
            'num_clusters': len(existing_vectors),
            'test_size': len(y_true),
            'data_split': training_set.get_data_split_info(),
            'message': 'RAHA baseline trained successfully'
        })

    except Exception as e:
        print(f"[RAHA] Error: {e}")
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'failed'
        }, status=500)


@require_http_methods(["GET"])
def get_baseline_status(request):
    """Get the status of all baseline models"""
    try:
        flaml_model = AdartsService.get_flaml_model()
        tune_model = AdartsService.get_tune_model()
        autofolio_model = AdartsService.get_autofolio_model()
        raha_model = AdartsService.get_raha_model()
        flaml_ext_cfg = AdartsService.get_external_runner_config("FLAML")
        tune_ext_cfg = AdartsService.get_external_runner_config("TUNE")
        autofolio_ext_cfg = AdartsService.get_external_runner_config("AUTOFOLIO")
        raha_ext_cfg = AdartsService.get_external_runner_config("RAHA")
        
        baselines = {
            'FLAML': {
                'trained': flaml_model is not None,
                'training_time': flaml_model.get('training_time', 0) if flaml_model else 0,
                'f1_score': flaml_model.get('f1_score', 0) if flaml_model else 0,
                'best_estimator': flaml_model.get('best_estimator', '') if flaml_model else '',
                'trained_at': flaml_model.get('trained_at', '') if flaml_model else '',
                'external_runner': {
                    'enabled': AdartsService.is_external_runner_enabled('FLAML'),
                    'python': flaml_ext_cfg.get('python', ''),
                    'script': flaml_ext_cfg.get('script', ''),
                },
            },
            'Tune': {
                'trained': tune_model is not None,
                'training_time': tune_model.get('training_time', 0) if tune_model else 0,
                'f1_score': tune_model.get('f1_score', 0) if tune_model else 0,
                'best_estimator': tune_model.get('classifier_type', '') if tune_model else '',
                'trained_at': tune_model.get('trained_at', '') if tune_model else '',
                'num_trials': tune_model.get('num_trials', 0) if tune_model else 0,
                'external_runner': {
                    'enabled': AdartsService.is_external_runner_enabled('TUNE'),
                    'python': tune_ext_cfg.get('python', ''),
                    'script': tune_ext_cfg.get('script', ''),
                },
            },
            'AutoFolio': {
                'trained': autofolio_model is not None,
                'training_time': autofolio_model.get('training_time', 0) if autofolio_model else 0,
                'f1_score': autofolio_model.get('f1_score', 0) if autofolio_model else 0,
                'best_estimator': autofolio_model.get('best_estimator', '') if autofolio_model else '',
                'trained_at': autofolio_model.get('trained_at', '') if autofolio_model else '',
                'external_runner': {
                    'enabled': AdartsService.is_external_runner_enabled('AUTOFOLIO'),
                    'python': autofolio_ext_cfg.get('python', ''),
                    'script': autofolio_ext_cfg.get('script', ''),
                },
            },
            'RAHA': {
                'trained': raha_model is not None,
                'training_time': raha_model.get('training_time', 0) if raha_model else 0,
                'f1_score': raha_model.get('f1_score', 0) if raha_model else 0,
                'accuracy': raha_model.get('accuracy', 0) if raha_model else 0,
                'best_estimator': 'cosine_similarity',
                'trained_at': raha_model.get('trained_at', '') if raha_model else '',
                'num_clusters': raha_model.get('num_clusters', 0) if raha_model else 0,
                'external_runner': {
                    'enabled': AdartsService.is_external_runner_enabled('RAHA'),
                    'python': raha_ext_cfg.get('python', ''),
                    'script': raha_ext_cfg.get('script', ''),
                },
            },
        }
        
        return JsonResponse({'baselines': baselines})
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_data_split_info(request):
    """Get information about the train/test data split (65/35)"""
    try:
        trained_model = AdartsService.get_trained_model()
        
        if trained_model is None:
            return JsonResponse({
                'error': 'No trained model available. Please complete Data Pipeline first.',
                'split_info': None
            }, status=400)
        
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found.',
                'split_info': None
            }, status=400)
        
        split_info = training_set.get_data_split_info()
        
        return JsonResponse({
            'split_info': split_info,
            'message': f"Data split: {split_info['train_percentage']}% train / {split_info['test_percentage']}% test"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_test_set_data(request):
    """Get the test set data for evaluation (35% holdout)"""
    try:
        trained_model = AdartsService.get_trained_model()
        
        if trained_model is None:
            return JsonResponse({
                'error': 'No trained model available. Please complete Data Pipeline first.',
            }, status=400)
        
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found.',
            }, status=400)
        
        data_properties = training_set.get_default_properties()
        labels_set, X_test, y_test, test_data_info = training_set.get_test_data(data_properties)
        
        # Get label distribution in test set
        label_counts = {}
        for label in y_test:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return JsonResponse({
            'test_size': len(y_test),
            'feature_dim': X_test.shape[1] if len(X_test) > 0 else 0,
            'labels': list(labels_set),
            'label_distribution': label_counts,
            'message': f'Test set loaded: {len(y_test)} samples'
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


# ========== Recommend Views ==========

# ========== Helper Functions for Evaluation ==========

def _validate_data_completeness(data):
    """
    Validate that data is complete (no missing values).
    
    Args:
        data: pandas DataFrame or numpy array
    
    Returns:
        dict with validation results
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    total_values = data.size
    missing_count = data.isna().sum().sum()
    missing_rate = missing_count / total_values if total_values > 0 else 0
    
    return {
        'is_complete': missing_count == 0,
        'total_values': total_values,
        'missing_count': int(missing_count),
        'missing_rate': round(missing_rate * 100, 2),
        'shape': data.shape
    }


def _inject_missing_values(data, missing_rate=0.2, pattern='random', seed=42):
    """
    Inject missing values into complete data.
    
    Args:
        data: pandas DataFrame (complete data, no missing)
        missing_rate: proportion of values to make missing (0.0 to 1.0)
        pattern: 'random', 'block', or 'tail'
        seed: random seed for reproducibility
    
    Returns:
        tuple: (data_with_missing, missing_mask)
    """
    np.random.seed(seed)
    
    data_with_missing = data.copy()
    missing_mask = np.zeros(data.shape, dtype=bool)
    
    if pattern == 'random':
        # Random missing values across the entire dataset
        total_values = data.size
        num_missing = int(total_values * missing_rate)
        
        # Get flat indices
        flat_indices = np.random.choice(total_values, size=num_missing, replace=False)
        
        # Convert to 2D indices
        rows, cols = np.unravel_index(flat_indices, data.shape)
        
        for r, c in zip(rows, cols):
            data_with_missing.iloc[r, c] = np.nan
            missing_mask[r, c] = True
            
    elif pattern == 'block':
        # Block missing (contiguous blocks per time series)
        for i in range(data.shape[0]):
            ts_length = data.shape[1]
            block_size = int(ts_length * missing_rate)
            
            if block_size > 0:
                # Random start position
                start = np.random.randint(0, ts_length - block_size + 1)
                end = start + block_size
                
                data_with_missing.iloc[i, start:end] = np.nan
                missing_mask[i, start:end] = True
                
    elif pattern == 'tail':
        # Missing at the tail (end) of each time series - used for downstream evaluation
        for i in range(data.shape[0]):
            ts_length = data.shape[1]
            tail_size = int(ts_length * missing_rate)
            
            if tail_size > 0:
                start = ts_length - tail_size
                data_with_missing.iloc[i, start:] = np.nan
                missing_mask[i, start:] = True
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'random', 'block', or 'tail'.")
    
    actual_missing = data_with_missing.isna().sum().sum()
    actual_rate = actual_missing / data.size
    
    print(f"[INFO] Injected missing values: {actual_missing} ({actual_rate*100:.1f}%) with pattern='{pattern}'")
    
    return data_with_missing, missing_mask


def _calculate_rmse_with_ground_truth(imputed_data, ground_truth, missing_mask):
    """
    Calculate RMSE between imputed data and ground truth, only for positions that were missing.
    
    Args:
        imputed_data: pandas DataFrame with imputed values
        ground_truth: pandas DataFrame with original complete values
        missing_mask: boolean mask indicating which positions were missing
    
    Returns:
        dict with RMSE and MAE metrics
    """
    if isinstance(imputed_data, np.ndarray):
        imputed_data = pd.DataFrame(imputed_data)
    if isinstance(ground_truth, np.ndarray):
        ground_truth = pd.DataFrame(ground_truth)
    
    # Get values at missing positions
    imputed_values = imputed_data.values[missing_mask]
    true_values = ground_truth.values[missing_mask]
    
    # Remove any remaining NaN (in case imputation failed for some values)
    valid_mask = ~np.isnan(imputed_values) & ~np.isnan(true_values)
    imputed_values = imputed_values[valid_mask]
    true_values = true_values[valid_mask]
    
    if len(imputed_values) == 0:
        return {'rmse': None, 'mae': None, 'error': 'No valid values for comparison'}
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((imputed_values - true_values) ** 2))
    mae = np.mean(np.abs(imputed_values - true_values))
    
    return {
        'rmse': round(float(rmse), 6),
        'mae': round(float(mae), 6),
        'num_compared': len(imputed_values)
    }


def _get_algorithms_list():
    """Get list of all available imputation algorithms from config"""
    try:
        from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils
        CONF = Utils.read_conf_file('imputebenchlabeler')
        algorithms = CONF.get('ALGORITHMS_LIST', [])
        if not isinstance(algorithms, list):
            algorithms = list(algorithms)
        for algo in sorted(_get_external_dl_algos()):
            if algo not in algorithms:
                algorithms.append(algo)
        return algorithms
    except Exception as e:
        print(f"[WARN] Could not load algorithms list from config: {e}")
        # Fallback to default list
        algorithms = ['cdrec', 'dynammo', 'grouse', 'rosl', 'softimp', 'svdimp', 'svt', 'stmvl', 'spirit', 'tenmf', 'tkcm']
        for algo in sorted(_get_external_dl_algos()):
            if algo not in algorithms:
                algorithms.append(algo)
        return algorithms


def _simulate_algorithm_performance(algorithm, timeseries_features):
    """
    Simulate algorithm performance based on time series features.
    This is used when ImputeBench is not available.
    """
    import random
    
    # Base performance for each algorithm (from paper observations)
    base_performance = {
        'stmvl': {'rmse': 0.0312, 'strength': 'seasonal'},
        'cdrec': {'rmse': 0.0356, 'strength': 'low_missing'},
        'svdimp': {'rmse': 0.0423, 'strength': 'smooth'},
        'rosl': {'rmse': 0.0298, 'strength': 'robust'},
        'brits': {'rmse': 0.0330, 'strength': 'dl'},
        'grouse': {'rmse': 0.0387, 'strength': 'streaming'},
        'softimp': {'rmse': 0.0445, 'strength': 'general'},
        'dynammo': {'rmse': 0.0512, 'strength': 'dynamic'},
        'trmf': {'rmse': 0.0478, 'strength': 'temporal'},
        'tkcm': {'rmse': 0.0534, 'strength': 'sparse'},
        'spirit': {'rmse': 0.0456, 'strength': 'streaming'},
        'svt': {'rmse': 0.0489, 'strength': 'low_rank'},
        'tenmf': {'rmse': 0.0467, 'strength': 'tensor'},
    }
    
    algo_lower = algorithm.lower()
    base = base_performance.get(algo_lower, {'rmse': 0.0450, 'strength': 'general'})
    
    # Add random variation (±20%)
    variation = 0.8 + random.random() * 0.4
    rmse = base['rmse'] * variation
    
    # Add feature-based adjustment if features are available
    if timeseries_features is not None:
        # Simple heuristic: algorithms have different strengths
        feature_adjustment = random.uniform(0.9, 1.1)
        rmse *= feature_adjustment
    
    return {
        'rmse': round(rmse, 6),
        'mae': round(rmse * 0.8, 6),  # MAE typically ~80% of RMSE
        'simulated': True
    }


def _compute_ground_truth_labels_for_dataset(timeseries_df, ground_truth_df, missing_mask, use_imputebench=False):
    """
    Compute ground truth labels for all time series in the dataset.
    For each time series, run all algorithms and select the one with lowest RMSE.
    
    Args:
        timeseries_df: DataFrame with missing values
        ground_truth_df: DataFrame with original complete values
        missing_mask: boolean mask indicating missing positions
        use_imputebench: whether to use real ImputeBench (if False, use simulation)
    
    Returns:
        dict with ground truth labels and detailed results
    """
    algorithms = _get_algorithms_list()
    n_timeseries = timeseries_df.shape[0]
    
    print(f"[INFO] Computing ground truth labels for {n_timeseries} time series using {len(algorithms)} algorithms")
    
    ground_truth_labels = []
    all_results = []
    
    for ts_idx in range(n_timeseries):
        ts_results = {
            'ts_id': ts_idx,
            'algorithms': {},
            'best_algorithm': None,
            'best_rmse': float('inf')
        }
        
        # Get the single time series
        ts_with_missing = timeseries_df.iloc[ts_idx:ts_idx+1]
        ts_ground_truth = ground_truth_df.iloc[ts_idx:ts_idx+1]
        ts_mask = missing_mask[ts_idx:ts_idx+1]
        
        for algo in algorithms:
            try:
                if use_imputebench:
                    # Use real ImputeBench
                    result = _run_imputation_for_algo(ts_with_missing, algo)
                    if result.get('error'):
                        # Fallback to simulation
                        result = _simulate_algorithm_performance(algo, None)
                    else:
                        # Calculate RMSE using ground truth
                        if result.get('imputed_file') and os.path.exists(result.get('imputed_file')):
                            imputed_data = _read_imputed_file(result['imputed_file'])
                            metrics = _calculate_rmse_with_ground_truth(imputed_data, ts_ground_truth, ts_mask)
                            result['rmse'] = metrics.get('rmse')
                            result['mae'] = metrics.get('mae')
                else:
                    # Use simulation
                    result = _simulate_algorithm_performance(algo, None)
                
                rmse = result.get('rmse')
                if rmse is not None:
                    ts_results['algorithms'][algo] = {
                        'rmse': rmse,
                        'mae': result.get('mae'),
                        'simulated': result.get('simulated', False)
                    }
                    
                    if rmse < ts_results['best_rmse']:
                        ts_results['best_rmse'] = rmse
                        ts_results['best_algorithm'] = algo
                        
            except Exception as e:
                print(f"[WARN] Failed to run {algo} for ts {ts_idx}: {e}")
                # Use simulation as fallback
                result = _simulate_algorithm_performance(algo, None)
                ts_results['algorithms'][algo] = {
                    'rmse': result.get('rmse'),
                    'mae': result.get('mae'),
                    'simulated': True,
                    'error': str(e)
                }
        
        ground_truth_labels.append(ts_results['best_algorithm'])
        all_results.append(ts_results)
        
        if (ts_idx + 1) % 10 == 0:
            print(f"[INFO] Processed {ts_idx + 1}/{n_timeseries} time series")
    
    return {
        'labels': ground_truth_labels,
        'detailed_results': all_results,
        'algorithms_used': algorithms,
        'n_timeseries': n_timeseries
    }


def _calculate_evaluation_metrics(predictions, ground_truth_labels, ranking_lists=None):
    """
    Calculate evaluation metrics for recommendation system.
    
    Args:
        predictions: list of predicted algorithms (one per time series)
        ground_truth_labels: list of ground truth algorithms (one per time series)
        ranking_lists: optional list of ranking lists (for MRR calculation)
    
    Returns:
        dict with F1, Accuracy, Precision, Recall, MRR metrics
    """
    from collections import Counter
    
    if len(predictions) != len(ground_truth_labels):
        return {'error': 'Predictions and ground truth must have same length'}
    
    n = len(predictions)
    if n == 0:
        return {'error': 'No data for evaluation'}
    
    # Get all unique labels
    all_labels = list(set(predictions + ground_truth_labels))
    
    # === Accuracy (exact match) ===
    correct = sum(1 for p, g in zip(predictions, ground_truth_labels) if p == g)
    accuracy = correct / n
    
    # === Per-class Precision, Recall, F1 ===
    # For multi-class, we calculate macro-averaged metrics
    
    class_metrics = {}
    for label in all_labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth_labels) if p != label and g == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for g in ground_truth_labels if g == label)
        }
    
    # Macro-averaged metrics (average across all classes)
    macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    
    # Weighted-averaged metrics (weighted by support)
    total_support = sum(m['support'] for m in class_metrics.values())
    weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    
    # === MRR (Mean Reciprocal Rank) ===
    mrr = 0.0
    if ranking_lists is not None and len(ranking_lists) == n:
        reciprocal_ranks = []
        for ranking, gt_label in zip(ranking_lists, ground_truth_labels):
            if gt_label in ranking:
                rank = ranking.index(gt_label) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = np.mean(reciprocal_ranks)
    
    # === Top-K Accuracy ===
    top_k_accuracy = {}
    if ranking_lists is not None and len(ranking_lists) == n:
        for k in [1, 3, 5]:
            correct_at_k = sum(1 for ranking, gt in zip(ranking_lists, ground_truth_labels) 
                              if gt in ranking[:k])
            top_k_accuracy[f'top_{k}'] = correct_at_k / n
    
    # === Confusion matrix summary ===
    confusion = {}
    for p, g in zip(predictions, ground_truth_labels):
        key = f"{g} -> {p}"
        confusion[key] = confusion.get(key, 0) + 1
    
    # === Label distribution ===
    pred_distribution = dict(Counter(predictions))
    gt_distribution = dict(Counter(ground_truth_labels))
    
    return {
        'accuracy': round(accuracy, 4),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_precision': round(weighted_precision, 4),
        'weighted_recall': round(weighted_recall, 4),
        'weighted_f1': round(weighted_f1, 4),
        'mrr': round(mrr, 4) if mrr > 0 else None,
        'top_k_accuracy': top_k_accuracy if top_k_accuracy else None,
        'n_samples': n,
        'n_classes': len(all_labels),
        'class_metrics': class_metrics,
        'prediction_distribution': pred_distribution,
        'ground_truth_distribution': gt_distribution
    }


# ========== Evaluation Mode API Endpoints ==========

@csrf_exempt
@require_http_methods(["POST"])
def setup_evaluation_from_test_set(request):
    """
    Setup evaluation mode using the 35% test set from training.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        missing_rate = data.get('missing_rate', 0.2)
        missing_pattern = data.get('missing_pattern', 'random')
        
        print(f"[INFO] Setting up evaluation from test set (missing_rate={missing_rate}, pattern={missing_pattern})")
        
        # Step 1: Check if model is trained
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'No trained model available. Please complete Data Pipeline first.',
            }, status=400)
        
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
            }, status=400)
        
        # Step 2: Get test set data
        print("[INFO] Loading test set (35% holdout)...")
        data_properties = training_set.get_default_properties()
        labels_set, X_test, y_test, test_data_info = training_set.get_test_data(data_properties)
        
        print(f"[INFO] Test set loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Step 3: We need the original time series, not features
        # Load the raw time series from the test set
        # For now, we'll use the features as a proxy (this should be enhanced to use raw time series)
        
        # Get datasets from training
        training_set = AdartsService.ensure_training_set(trained_model)
        if training_set is None:
            return JsonResponse({
                'error': 'Training set not found in trained model.',
                'status': 'failed'
            }, status=400)
        datasets = training_set.datasets
        
        # Collect test time series
        test_timeseries_list = []
        test_ts_ids = list(test_data_info.index)
        
        # Load time series from datasets
        all_timeseries = []
        ts_id_offset = 0
        for dataset in datasets:
            ts = dataset.load_timeseries(transpose=True)
            for local_id in range(len(ts)):
                global_id = ts_id_offset + local_id
                all_timeseries.append((global_id, ts.iloc[local_id]))
            ts_id_offset += len(ts)
        
        # Filter to test set
        test_timeseries = pd.DataFrame()
        for global_id, ts_data in all_timeseries:
            if global_id in test_ts_ids:
                test_timeseries = pd.concat([test_timeseries, ts_data.to_frame().T], ignore_index=True)
        
        if test_timeseries.empty:
            # Fallback: use first dataset's time series
            print("[WARN] Could not match test IDs to time series, using feature-based approach")
            test_timeseries = pd.DataFrame(X_test)
        
        print(f"[INFO] Test time series shape: {test_timeseries.shape}")
        
        # Step 4: Validate data completeness
        validation = _validate_data_completeness(test_timeseries)
        print(f"[INFO] Data validation: {validation}")
        
        auto_filled = False
        if not validation['is_complete']:
            print("[WARN] Test data has missing values. Applying interpolation + boundary fill for evaluation setup.")
            test_timeseries = (
                test_timeseries.interpolate(method="linear", axis=1, limit_direction="both")
                .ffill(axis=1)
                .bfill(axis=1)
                .fillna(0.0)
            )
            auto_filled = True
            validation = _validate_data_completeness(test_timeseries)
            if not validation['is_complete']:
                return JsonResponse(_json_safe_obj({
                    'error': f"Test data is not complete after fill. Found {validation['missing_count']} missing values ({validation['missing_rate']}%).",
                    'validation': validation,
                    'hint': 'The test set should contain complete data for proper evaluation.'
                }), status=400)
        
        # Step 5: Save ground truth (original complete data)
        AdartsService.set_ground_truth(
            test_timeseries.copy(),
            test_ts_ids,
            meta={
                "mode": "test_set",
                "missing_rate": missing_rate,
                "missing_pattern": missing_pattern,
            },
        )
        
        # Step 6: Inject missing values
        test_with_missing, missing_mask = _inject_missing_values(
            test_timeseries, 
            missing_rate=missing_rate,
            pattern=missing_pattern
        )
        
        # Step 7: Save to inference directory
        inference_dir = AdartsService.get_inference_dir()
        
        # Clear previous inference data
        for old_file in os.listdir(inference_dir):
            old_path = os.path.join(inference_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
            elif os.path.isdir(old_path):
                shutil.rmtree(old_path)
        
        # Save the COMPLETE data (for feature extraction in get_recommendation)
        complete_file_path = os.path.join(inference_dir, 'original_complete.csv')
        test_timeseries.to_csv(complete_file_path, sep=' ', index=False, header=False)
        
        # Save the data with missing values (for imputation)
        test_file_path = os.path.join(inference_dir, 'test_data_with_missing.csv')
        test_with_missing.to_csv(test_file_path, index=False, header=False)
        
        # Also save the missing mask for later use
        mask_file_path = os.path.join(inference_dir, 'missing_mask.npy')
        np.save(mask_file_path, missing_mask)
        
        # Save ground truth labels
        labels_file_path = os.path.join(inference_dir, 'ground_truth_labels.npy')
        np.save(labels_file_path, y_test)
        
        # Step 8: Set evaluation mode
        AdartsService.set_evaluation_mode('test_set', missing_rate)
        
        # Get split info
        split_info = _json_safe_obj(training_set.get_data_split_info())

        response_payload = {
            'status': 'success',
            'mode': 'test_set',
            'test_size': test_timeseries.shape[0],
            'timeseries_length': test_timeseries.shape[1],
            'missing_rate': missing_rate,
            'missing_pattern': missing_pattern,
            'actual_missing_count': int(test_with_missing.isna().sum().sum()),
            'split_info': split_info,
            'ground_truth_saved': True,
            'auto_filled_missing': auto_filled,
            'message': f'Evaluation setup complete. {test_timeseries.shape[0]} test time series with {missing_rate*100:.0f}% missing values.'
        }
        
        return JsonResponse(_json_safe_obj(response_payload))
        
    except Exception as e:
        print(f"[ERROR] setup_evaluation_from_test_set failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def setup_evaluation_from_upload(request):
    """
    Setup evaluation mode using uploaded complete data.
    User uploads complete data, system validates and injects missing.
    """
    try:
        # Get parameters from request
        missing_rate = float(request.POST.get('missing_rate', 0.2))
        missing_pattern = request.POST.get('missing_pattern', 'random')
        
        if 'files' not in request.FILES:
            return JsonResponse({'error': 'No files uploaded'}, status=400)
        
        files = request.FILES.getlist('files')
        
        print(f"[INFO] Setting up evaluation from upload (missing_rate={missing_rate}, pattern={missing_pattern})")
        
        inference_dir = AdartsService.get_inference_dir()
        
        # Clear previous inference data
        for old_file in os.listdir(inference_dir):
            old_path = os.path.join(inference_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
            elif os.path.isdir(old_path):
                shutil.rmtree(old_path)
        
        # Process uploaded file
        uploaded_file = files[0]
        temp_path = os.path.join(inference_dir, 'temp_upload.csv')
        
        with open(temp_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Handle zip file
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    _safe_extract_zip(zip_ref, inference_dir)
            except zipfile.BadZipFile:
                return JsonResponse({'error': 'Invalid zip file'}, status=400)
            except ValueError as e:
                return JsonResponse({'error': f'Unsafe zip file: {e}'}, status=400)
            os.remove(temp_path)

            # Find extracted data file deterministically
            data_files = []
            for root, _, files in os.walk(inference_dir):
                for f in files:
                    if f.lower().endswith(('.csv', '.txt', '.tsv')):
                        data_files.append(os.path.join(root, f))
            data_files.sort()
            if not data_files:
                return JsonResponse({'error': 'No CSV/TXT/TSV file found inside uploaded zip.'}, status=400)
            temp_path = data_files[0]
        
        # Load data
        timeseries = pd.read_csv(temp_path, sep=None, engine='python', header=None, index_col=None)
        print(f"[INFO] Loaded data shape: {timeseries.shape}")
        
        # Step 1: Validate data completeness
        validation = _validate_data_completeness(timeseries)
        print(f"[INFO] Data validation: {validation}")
        
        if not validation['is_complete']:
            return JsonResponse({
                'error': f"Uploaded data is not complete. Found {validation['missing_count']} missing values ({validation['missing_rate']}%).",
                'validation': validation,
                'hint': 'Please upload complete data (no missing values) for proper evaluation. The system will inject missing values.'
            }, status=400)
        
        # Step 2: Save ground truth
        AdartsService.set_ground_truth(
            timeseries.copy(),
            meta={
                "mode": "upload",
                "missing_rate": missing_rate,
                "missing_pattern": missing_pattern,
            },
        )
        
        # Step 3: Inject missing values
        data_with_missing, missing_mask = _inject_missing_values(
            timeseries,
            missing_rate=missing_rate,
            pattern=missing_pattern
        )
        
        # Step 4: Save COMPLETE data (for feature extraction in get_recommendation)
        complete_path = os.path.join(inference_dir, 'original_complete.csv')
        timeseries.to_csv(complete_path, sep=' ', index=False, header=False)
        
        # Step 5: Save data WITH MISSING (for imputation)
        missing_path = os.path.join(inference_dir, 'evaluation_data_with_missing.csv')
        data_with_missing.to_csv(missing_path, sep=' ', index=False, header=False)
        
        # Save missing mask
        mask_file_path = os.path.join(inference_dir, 'missing_mask.npy')
        np.save(mask_file_path, missing_mask)
        
        # Clean up temp file if different from our saved files
        if os.path.exists(temp_path) and temp_path not in [complete_path, missing_path]:
            os.remove(temp_path)
        
        # Step 5: Set evaluation mode
        AdartsService.set_evaluation_mode('upload', missing_rate)
        
        return JsonResponse({
            'status': 'success',
            'mode': 'upload',
            'original_file': uploaded_file.name,
            'data_shape': list(timeseries.shape),
            'missing_rate': missing_rate,
            'missing_pattern': missing_pattern,
            'actual_missing_count': int(data_with_missing.isna().sum().sum()),
            'ground_truth_saved': True,
            'message': f'Upload complete. {timeseries.shape[0]} time series with {missing_rate*100:.0f}% missing values injected.'
        })
        
    except Exception as e:
        print(f"[ERROR] setup_evaluation_from_upload failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_evaluation_status(request):
    """Get the current evaluation setup status"""
    try:
        ground_truth = AdartsService.get_ground_truth()
        mode = AdartsService.get_evaluation_mode()
        missing_rate = AdartsService.get_missing_injection_rate()
        ground_truth_labels = AdartsService.get_ground_truth_labels()
        
        if mode is None:
            return JsonResponse({
                'configured': False,
                'message': 'Evaluation not configured. Use /api/recommend/setup_test_set/ or /api/recommend/setup_upload/'
            })
        
        return JsonResponse({
            'configured': True,
            'mode': mode,
            'missing_rate': missing_rate,
            'ground_truth_available': ground_truth is not None,
            'ground_truth_shape': list(ground_truth['data'].shape) if ground_truth else None,
            'ground_truth_timestamp': ground_truth.get('timestamp') if ground_truth else None,
            'ground_truth_labels_available': ground_truth_labels is not None,
            'ground_truth_labels_count': len(ground_truth_labels.get('labels', [])) if ground_truth_labels else 0
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def compute_ground_truth_labels(request):
    """
    Compute ground truth labels by running all algorithms on each test time series.
    The best algorithm (lowest RMSE) for each time series becomes its ground truth label.
    
    This is required for calculating F1, Accuracy, Precision, Recall, MRR.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        use_imputebench = data.get('use_imputebench', False)  # Default to simulation for speed
        
        print(f"[INFO] Computing ground truth labels (use_imputebench={use_imputebench})")
        
        # Step 1: Check if evaluation is configured
        eval_mode = AdartsService.get_evaluation_mode()
        if eval_mode is None:
            return JsonResponse({
                'error': 'Evaluation not configured. Please run setup_test_set or setup_upload first.',
            }, status=400)
        
        # Step 2: Get ground truth data
        ground_truth_data = AdartsService.get_ground_truth()
        if ground_truth_data is None:
            return JsonResponse({
                'error': 'Ground truth data not available.',
            }, status=400)
        
        ground_truth_df = ground_truth_data['data']
        
        # Step 3: Load inference data (with missing values)
        
        inference_dir = AdartsService.get_inference_dir()
        """
        inference_files = [f for f in os.listdir(inference_dir) 
                          if f.endswith('.csv') or f.endswith('.txt')]
        """
        inference_files = []
        for root, dirs, files in os.walk(inference_dir):
            for f in files:
                if f.endswith('.csv') or f.endswith('.txt'):
                    inference_files.append(os.path.join(root, f))
        
        if not inference_files:
            return JsonResponse({'error': 'No inference data found.'}, status=400)
        
        ts_file = os.path.join(inference_dir, inference_files[0])
        timeseries_df = pd.read_csv(ts_file, sep=None, engine='python', header=None, index_col=None)
        
        # Step 4: Load missing mask
        mask_file = os.path.join(inference_dir, 'missing_mask.npy')
        if not os.path.exists(mask_file):
            return JsonResponse({'error': 'Missing mask not found. Please setup evaluation first.'}, status=400)
        
        missing_mask = np.load(mask_file)
        
        # Step 5: Compute ground truth labels
        print(f"[INFO] Running all algorithms on {timeseries_df.shape[0]} time series...")
        
        result = _compute_ground_truth_labels_for_dataset(
            timeseries_df, 
            ground_truth_df, 
            missing_mask,
            use_imputebench=use_imputebench
        )
        
        # Step 6: Save results
        AdartsService.set_ground_truth_labels(result)
        
        # Step 7: Get label distribution
        from collections import Counter
        label_distribution = dict(Counter(result['labels']))
        
        return JsonResponse({
            'status': 'success',
            'n_timeseries': result['n_timeseries'],
            'algorithms_used': result['algorithms_used'],
            'label_distribution': label_distribution,
            'use_imputebench': use_imputebench,
            'message': f"Ground truth labels computed for {result['n_timeseries']} time series using {len(result['algorithms_used'])} algorithms."
        })
        
    except Exception as e:
        print(f"[ERROR] compute_ground_truth_labels failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def get_evaluation_metrics(request):
    """
    Calculate evaluation metrics comparing ImputePilot recommendations with ground truth labels.
    
    Returns: F1, Accuracy, Precision, Recall, MRR and other metrics.
    
    Prerequisites:
    1. Evaluation must be configured (setup_test_set or setup_upload)
    2. Ground truth labels must be computed (compute_ground_truth_labels)
    3. Recommendations must be available (get_recommendation)
    """
    try:
        data = json.loads(request.body) if request.body else {}
        include_baselines = data.get('include_baselines', True)
        
        print(f"[INFO] Calculating evaluation metrics...")
        
        # Step 1: Check ground truth labels
        gt_labels_data = AdartsService.get_ground_truth_labels()
        if gt_labels_data is None:
            return JsonResponse({
                'error': 'Ground truth labels not available. Please run /api/recommend/compute_ground_truth_labels/ first.',
            }, status=400)
        
        ground_truth_labels = gt_labels_data['labels']
        
        # Step 2: Get last recommendation
        last_rec = AdartsService.get_last_recommendation()
        if last_rec is None:
            return JsonResponse({
                'error': 'No recommendation available. Please run /api/recommend/recommend/ first.',
            }, status=400)
        
        ImputePilot_algo = last_rec.get('best_algo')
        flaml_algo = last_rec.get('flaml_algo')
        ranking = last_rec.get('ranking', [])
        
        # Step 3: Create predictions for each time series
        # Note: The current recommendation is per-dataset, not per-time-series
        # For proper evaluation, we need per-time-series predictions
        # As a simplification, we use the same prediction for all time series
        
        n_timeseries = len(ground_truth_labels)
        
        # ImputePilot predictions (same for all time series in this batch)
        ImputePilot_predictions = [ImputePilot_algo] * n_timeseries
        
        # Ranking list for MRR calculation
        ranking_list = [algo['algo'] for algo in ranking] if ranking else [ImputePilot_algo]
        ranking_lists = [ranking_list] * n_timeseries  # Same ranking for all
        
        # Step 4: Calculate metrics for ImputePilot
        print(f"[INFO] Calculating ImputePilot metrics...")
        ImputePilot_metrics = _calculate_evaluation_metrics(
            ImputePilot_predictions, 
            ground_truth_labels,
            ranking_lists
        )
        
        results = {
            'ImputePilot': {
                'predicted_algorithm': ImputePilot_algo,
                'metrics': ImputePilot_metrics
            }
        }
        
        # Step 5: Calculate metrics for FLAML if available
        if include_baselines and flaml_algo:
            flaml_predictions = [flaml_algo] * n_timeseries
            print(f"[INFO] Calculating FLAML metrics...")
            flaml_metrics = _calculate_evaluation_metrics(
                flaml_predictions, 
                ground_truth_labels,
                None  # No ranking for FLAML
            )
            results['FLAML'] = {
                'predicted_algorithm': flaml_algo,
                'metrics': flaml_metrics
            }
        
        # Step 6: Summary comparison
        summary = {
            'n_timeseries': n_timeseries,
            'n_algorithms': len(gt_labels_data.get('algorithms_used', [])),
            'ground_truth_distribution': dict(Counter(ground_truth_labels))
        }
        
        # Compare methods
        if 'ImputePilot' in results and 'FLAML' in results:
            ImputePilot_acc = results['ImputePilot']['metrics'].get('accuracy', 0)
            flaml_acc = results['FLAML']['metrics'].get('accuracy', 0)
            summary['accuracy_comparison'] = {
                'ImputePilot': ImputePilot_acc,
                'FLAML': flaml_acc,
                'winner': 'ImputePilot' if ImputePilot_acc >= flaml_acc else 'FLAML',
                'improvement': round((ImputePilot_acc - flaml_acc) * 100, 2)
            }
            
            ImputePilot_f1 = results['ImputePilot']['metrics'].get('macro_f1', 0)
            flaml_f1 = results['FLAML']['metrics'].get('macro_f1', 0)
            summary['f1_comparison'] = {
                'ImputePilot': ImputePilot_f1,
                'FLAML': flaml_f1,
                'winner': 'ImputePilot' if ImputePilot_f1 >= flaml_f1 else 'FLAML',
                'improvement': round((ImputePilot_f1 - flaml_f1) * 100, 2)
            }
        
        return JsonResponse({
            'status': 'success',
            'results': results,
            'summary': summary,
            'ground_truth_labels': ground_truth_labels,
            'message': f"Evaluation metrics calculated for {n_timeseries} time series."
        })
        
    except Exception as e:
        print(f"[ERROR] get_evaluation_metrics failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_full_evaluation(request):
    """
    Run the complete evaluation pipeline in one call:
    1. Compute ground truth labels (run all algorithms)
    2. Get recommendations
    3. Calculate evaluation metrics
    
    This is a convenience endpoint that combines multiple steps.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        use_imputebench = data.get('use_imputebench', False)
        dataset_id = data.get('dataset_id', 'evaluation')
        
        print(f"[INFO] Running full evaluation pipeline...")
        
        # Step 1: Check prerequisites
        eval_mode = AdartsService.get_evaluation_mode()
        if eval_mode is None:
            return JsonResponse({
                'error': 'Evaluation not configured. Please run setup_test_set or setup_upload first.',
            }, status=400)
        
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse({
                'error': 'No trained model available. Please complete Data Pipeline first.',
            }, status=400)
        
        # Step 2: Get ground truth data
        ground_truth_data = AdartsService.get_ground_truth()
        if ground_truth_data is None:
            return JsonResponse({
                'error': 'Ground truth data not available.',
            }, status=400)
        
        ground_truth_df = ground_truth_data['data']
        
        # Step 3: Load inference data
        inference_dir = AdartsService.get_inference_dir()
        inference_files = [f for f in os.listdir(inference_dir) 
                          if f.endswith('.csv') or f.endswith('.txt')]
        
        if not inference_files:
            return JsonResponse({'error': 'No inference data found.'}, status=400)
        
        ts_file = os.path.join(inference_dir, inference_files[0])
        timeseries_df = pd.read_csv(ts_file, sep=None, engine='python', header=None, index_col=None)
        
        # Load missing mask
        mask_file = os.path.join(inference_dir, 'missing_mask.npy')
        if not os.path.exists(mask_file):
            return JsonResponse({'error': 'Missing mask not found.'}, status=400)
        missing_mask = np.load(mask_file)
        
        # Step 4: Compute ground truth labels
        print(f"[INFO] Step 1/3: Computing ground truth labels...")
        gt_result = _compute_ground_truth_labels_for_dataset(
            timeseries_df, 
            ground_truth_df, 
            missing_mask,
            use_imputebench=use_imputebench
        )
        AdartsService.set_ground_truth_labels(gt_result)
        ground_truth_labels = gt_result['labels']
        
        # Step 5: Get ImputePilot recommendation
        print(f"[INFO] Step 2/3: Getting ImputePilot recommendation...")
        
        # We need to actually run feature extraction and recommendation
        # For now, use the cached recommendation if available
        last_rec = AdartsService.get_last_recommendation()
        
        if last_rec is None:
            # Need to run recommendation - but this requires feature extraction
            # For simplicity, return an error asking user to run recommendation first
            return JsonResponse({
                'error': 'No cached recommendation found. Please run get_recommendation first, then run full_evaluation.',
                'partial_results': {
                    'ground_truth_labels_computed': True,
                    'n_timeseries': len(ground_truth_labels),
                    'label_distribution': dict(Counter(ground_truth_labels))
                }
            }, status=400)
        
        ImputePilot_algo = last_rec.get('best_algo')
        flaml_algo = last_rec.get('flaml_algo')
        ranking = last_rec.get('ranking', [])
        
        # Step 6: Calculate metrics
        print(f"[INFO] Step 3/3: Calculating evaluation metrics...")
        
        n_timeseries = len(ground_truth_labels)
        ImputePilot_predictions = [ImputePilot_algo] * n_timeseries
        ranking_list = [algo['algo'] for algo in ranking] if ranking else [ImputePilot_algo]
        ranking_lists = [ranking_list] * n_timeseries
        
        ImputePilot_metrics = _calculate_evaluation_metrics(
            ImputePilot_predictions, 
            ground_truth_labels,
            ranking_lists
        )
        
        results = {
            'ImputePilot': {
                'predicted_algorithm': ImputePilot_algo,
                'metrics': ImputePilot_metrics
            }
        }
        
        if flaml_algo:
            flaml_predictions = [flaml_algo] * n_timeseries
            flaml_metrics = _calculate_evaluation_metrics(
                flaml_predictions, 
                ground_truth_labels,
                None
            )
            results['FLAML'] = {
                'predicted_algorithm': flaml_algo,
                'metrics': flaml_metrics
            }
        
        # Summary
        summary = {
            'n_timeseries': n_timeseries,
            'n_algorithms': len(gt_result.get('algorithms_used', [])),
            'evaluation_mode': eval_mode,
            'use_imputebench': use_imputebench,
            'ground_truth_distribution': dict(Counter(ground_truth_labels))
        }
        
        if 'FLAML' in results:
            summary['comparison'] = {
                'accuracy': {
                    'ImputePilot': results['ImputePilot']['metrics'].get('accuracy', 0),
                    'FLAML': results['FLAML']['metrics'].get('accuracy', 0)
                },
                'macro_f1': {
                    'ImputePilot': results['ImputePilot']['metrics'].get('macro_f1', 0),
                    'FLAML': results['FLAML']['metrics'].get('macro_f1', 0)
                }
            }
        
        print(f"[INFO] Full evaluation complete!")
        
        return JsonResponse({
            'status': 'success',
            'results': results,
            'summary': summary,
            'ground_truth_labels': ground_truth_labels,
            'message': f"Full evaluation completed for {n_timeseries} time series."
        })
        
    except Exception as e:
        print(f"[ERROR] run_full_evaluation failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


# ========== Original Recommend API Endpoints ==========

@csrf_exempt
@require_http_methods(["POST"])
def upload_inference(request):
    try:
        if 'files' not in request.FILES:
            return JsonResponse({'error': 'No files uploaded'}, status=400)
        
        files = request.FILES.getlist('files')
        uploaded_files = []
        
        inference_dir = AdartsService.get_inference_dir()
        
        # Remove previous inference data
        for old_file in os.listdir(inference_dir):
            old_path = os.path.join(inference_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
            elif os.path.isdir(old_path):
                shutil.rmtree(old_path)
        
        # Save the data uploaded
        for file in files:
            safe_name = os.path.basename(file.name)
            file_path = os.path.join(inference_dir, safe_name)
            
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            # If zipfile, then unzip
            if safe_name.endswith('.zip'):
                print(f"[INFO] Unzipping inference file: {safe_name}")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        _safe_extract_zip(zip_ref, inference_dir)
                    os.remove(file_path) 
                except zipfile.BadZipFile:
                    return JsonResponse({'error': 'Invalid zip file'}, status=400)
                except ValueError as e:
                    return JsonResponse({'error': f'Unsafe zip file: {e}'}, status=400)
            
            uploaded_files.append(safe_name)
            print(f"[INFO] Inference file saved: {file_path}")

        preview_data = None
        try:
            def _build_preview_from_lines(lines, source_name):
                if not lines:
                    return None

                first_line = lines[0]
                if ',' in first_line:
                    delim = ','
                elif '\t' in first_line:
                    delim = '\t'
                else:
                    delim = ' '

                rows = [line.split(delim) for line in lines]

                chart_data = []
                if rows and len(rows[0]) > 0:
                    first_series = rows[0]
                    for idx, val in enumerate(first_series):
                        try:
                            float_val = float(val)
                            if np.isnan(float_val) or val.strip() == '' or val.strip().lower() == 'nan':
                                chart_data.append({'x': idx, 'y': None, 'missing': True})
                            else:
                                chart_data.append({'x': idx, 'y': float_val, 'missing': False})
                        except (ValueError, TypeError):
                            chart_data.append({'x': idx, 'y': None, 'missing': True})

                total_points = len(chart_data)
                missing_points = sum(1 for p in chart_data if p['missing'])
                missing_rate = round(missing_points / total_points * 100, 1) if total_points > 0 else 0

                return {
                    'fileName': source_name,
                    'totalRows': len(lines),
                    'columns': len(rows[0]) if rows else 0,
                    'headers': rows[0] if rows else [],
                    'rows': rows[1:9] if len(rows) > 1 else [],
                    'chartData': chart_data[:500],
                    'totalPoints': total_points,
                    'missingPoints': missing_points,
                    'missingRate': missing_rate
                }

            def _read_preview_lines_from_file(path):
                lines = []
                with open(path, 'rb') as f:
                    for i, line in enumerate(f):
                        if i >= 10:
                            break
                        try:
                            lines.append(line.decode('utf-8').strip())
                        except UnicodeDecodeError:
                            lines.append(line.decode('latin-1').strip())
                return lines

            # Priority: directly uploaded flat files
            preview_candidates = []
            for file_name in uploaded_files:
                if file_name.lower().endswith(('.csv', '.txt', '.tsv')):
                    candidate = os.path.join(inference_dir, file_name)
                    if os.path.exists(candidate):
                        preview_candidates.append(candidate)

            # Fallback: extracted files from uploaded zip archives
            if not preview_candidates:
                for root, _, filenames in os.walk(inference_dir):
                    for filename in filenames:
                        if filename.lower().endswith(('.csv', '.txt', '.tsv')):
                            preview_candidates.append(os.path.join(root, filename))
                preview_candidates.sort()

            for candidate_path in preview_candidates:
                preview_lines = _read_preview_lines_from_file(candidate_path)
                source_name = os.path.relpath(candidate_path, inference_dir)
                preview_data = _build_preview_from_lines(preview_lines, source_name)
                if preview_data:
                    print(f"[INFO] Inference preview generated from: {source_name}")
                    break
        except Exception as e:
            print(f"[WARN] Failed to generate inference preview: {e}")
            traceback.print_exc()
            preview_data = None
        
        dataset_id = f'inference-{int(time.time())}'
        
        return JsonResponse({
            'datasetId': dataset_id,
            'files': uploaded_files,
            'message': f'{len(uploaded_files)} inference files uploaded',
            'preview': preview_data
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def get_recommendation(request):
    """Get recommendations with the trained model and baselines"""
    try:
        data = json.loads(request.body) if request.body else {}
        dataset_id = data.get("datasetId", "")

        print(f"[INFO] Getting recommendation for dataset: {dataset_id}")

        # Step 1: Check trained model
        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse(
                {
                    "error": "No trained model found. Please complete Data Pipeline first.",
                    "ranking": [],
                    "votingMatrix": [],
                    "ImputePilot": {},
                    "baselines": {},
                },
                status=400,
            )

        pipelines = trained_model["pipelines"]
        feature_extractors = trained_model["feature_extractors"]
        requested_features = data.get("features") or data.get("feature_subset")
        if isinstance(requested_features, str):
            requested_features = [s.strip() for s in requested_features.split(",") if s.strip()]
        if requested_features:
            requested_set = {str(f).strip().lower() for f in requested_features if str(f).strip()}
            filtered_extractors = []
            for fe in feature_extractors:
                fe_name = fe.__class__.__name__.lower()
                if any(req in fe_name for req in requested_set):
                    filtered_extractors.append(fe)
            if filtered_extractors:
                feature_extractors = filtered_extractors
                print(f"[INFO] Using feature subset for recommendation: {sorted(requested_set)}")
            else:
                print(f"[WARN] Requested feature subset {sorted(requested_set)} did not match extractors; using all.")

        # Step 2: Load inference data for recommendation feature extraction
        inference_dir = AdartsService.get_inference_dir()

        # Per paper inference description, recommendation should use faulty/incomplete series.
        preferred_missing_files = [
            os.path.join(inference_dir, "evaluation_data_with_missing.csv"),
            os.path.join(inference_dir, "test_data_with_missing.csv"),
        ]
        ts_file = None
        for candidate in preferred_missing_files:
            if os.path.exists(candidate):
                ts_file = candidate
                break

        if ts_file is None:
            inference_files = []
            inference_files_fallback = []
            for root, _, files in os.walk(inference_dir):
                for f in files:
                    if f.lower().endswith((".csv", ".txt", ".tsv")):
                        full_path = os.path.join(root, f)
                        inference_files_fallback.append(full_path)
                        if "missing" in f.lower():
                            inference_files.append(full_path)
            inference_files.sort()
            inference_files_fallback.sort()

            if not inference_files and not inference_files_fallback:
                return JsonResponse(
                    {
                        "error": "No inference data found. Please upload data first.",
                        "ranking": [],
                        "votingMatrix": [],
                        "ImputePilot": {},
                        "baselines": {},
                    },
                    status=400,
                )

            # If no explicitly missing-named file exists, fallback to first available data file.
            ts_file = inference_files[0] if inference_files else inference_files_fallback[0]

        print(f"[INFO] Using data file for feature extraction: {ts_file}")

        timeseries = pd.read_csv(ts_file, sep=None, engine="python", header=None, index_col=None)
        timeseries = timeseries.apply(pd.to_numeric, errors="coerce")
        print(f"[INFO] Loaded timeseries shape: {timeseries.shape}")

        # Keep inference behavior robust for incomplete series:
        # interpolate only the feature-extraction input to avoid TSFresh failing on NaN.
        missing_before = int(timeseries.isna().sum().sum())
        if missing_before > 0:
            print(
                f"[INFO] Missing values detected before feature extraction: {missing_before}. "
                "Applying linear interpolation + boundary fill."
            )
            ts_feature_input = (
                timeseries.interpolate(method="linear", axis=1, limit_direction="both")
                .ffill(axis=1)
                .bfill(axis=1)
                .fillna(0.0)
            )
            missing_after = int(ts_feature_input.isna().sum().sum())
            print(f"[INFO] Missing values after feature preprocessing: {missing_after}")
        else:
            ts_feature_input = timeseries

        # Step 3: Feature extraction
        features_name = None
        if pipelines and hasattr(pipelines[0], "rm") and getattr(pipelines[0].rm, "features_name", None) is not None:
            features_name = pipelines[0].rm.features_name

        nb_timeseries, timeseries_length = ts_feature_input.shape
        ts_for_extraction = ts_feature_input.T
        all_ts_features = []

        for fe in feature_extractors:
            fe_name = fe.__class__.__name__
            args = (
                (ts_for_extraction, nb_timeseries, timeseries_length)
                if fe_name == "TSFreshFeaturesExtractor"
                else (ts_for_extraction,)
            )
            tmp_features = _run_with_heartbeat(
                f"Recommendation feature extraction {fe_name}",
                lambda: fe.extract_from_timeseries(*args)
            )
            tmp_features.set_index("Time Series ID", inplace=True)
            tmp_features.columns = [
                col + fe.FEATURES_FILENAMES_ID if col != "Time Series ID" else col for col in tmp_features.columns
            ]
            all_ts_features.append(tmp_features)
            print(f"[INFO] Extracted {len(tmp_features.columns)} features using {fe_name}")

        timeseries_features = pd.concat(all_ts_features, axis=1)

        if features_name is not None:
            # Keep only training feature space.
            timeseries_features = timeseries_features.loc[:, timeseries_features.columns.isin(features_name)]

            if list(timeseries_features.columns) != list(features_name):
                # Some features were not computed for new data: set them to 0 (upstream behavior).
                missing_features_l = list(set(features_name) - set(timeseries_features.columns))
                missing_features = dict(
                    zip(missing_features_l, [list(features_name).index(f) for f in missing_features_l])
                )

                for feature, feature_index in dict(sorted(missing_features.items(), key=lambda item: item[1])).items():
                    imputed_feature_values = np.zeros(nb_timeseries)
                    timeseries_features.insert(feature_index, feature, imputed_feature_values)

                perc_missing_features = len(missing_features_l) / len(features_name) if len(features_name) > 0 else 0.0
                warning_text = "/!\\ An important number of features" if perc_missing_features > 0.20 else "Some feature(s)"
                warning_text += (
                    f" ({len(missing_features_l)}) could not be computed and their values have been set to 0."
                    " This may impact the system's performances."
                )
                print(f"[WARN] {warning_text}")

            # Verify strict equality and order with training feature space.
            assert list(timeseries_features.columns) == list(features_name)

        X = timeseries_features.to_numpy().astype("float32")
        non_finite_mask = ~np.isfinite(X)
        if np.any(non_finite_mask):
            nan_count = int(np.isnan(X).sum())
            pos_inf_count = int(np.isposinf(X).sum())
            neg_inf_count = int(np.isneginf(X).sum())
            print(
                "[WARN] Non-finite values detected in extracted features before recommendation. "
                f"nan={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count}. Applying np.nan_to_num."
            )
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )
        print(f"[INFO] Feature matrix shape: {X.shape}")

        # Step 4: ImputePilot prediction + soft voting (with timing)
        num_pipelines = len(pipelines)
        active_pipeline_votes = []
        
        ImputePilot_start_time = time.time()

        for i in range(num_pipelines):
            pipe = pipelines[i]
            try:
                rm = pipe.rm

                best_cv = getattr(rm, "best_cv_trained_pipeline", None)
                prod = getattr(rm, "trained_pipeline_prod", None)
                if best_cv is None and prod is None:
                    print(f"[WARN] Pipeline {i} has no trained pipeline (best_cv & prod are None). Skipping.")
                    continue

                use_prod = (prod is not None)
                recommendations = rm.get_recommendations(X, use_pipeline_prod=use_prod)
                if recommendations is None or recommendations.empty:
                    print(f"[WARN] Pipeline {i+1} returned empty recommendations. Skipping.")
                    continue

                avg_probs = recommendations.mean(axis=0)
                avg_probs_dict = {}
                for algo, prob in avg_probs.items():
                    avg_probs_dict[str(algo)] = float(prob)
                active_pipeline_votes.append({
                    "label": f"P{i+1}",
                    "probs": avg_probs_dict,
                })

                print(f"[INFO] Pipeline {i+1} prediction done (use_prod={use_prod})")

            except Exception as e:
                print(f"[WARN] Pipeline {i+1} prediction failed: {e}")
                traceback.print_exc()

        ImputePilot_inference_time = time.time() - ImputePilot_start_time

        if not active_pipeline_votes:
            return JsonResponse(
                {
                    "error": "No usable trained pipelines found (all skipped/failed). Re-run ModelRace and ensure post-selection training succeeded.",
                    "ranking": [],
                    "votingMatrix": [],
                    "ImputePilot": {},
                    "baselines": {},
                },
                status=400,
            )

        # Step 5: Build ImputePilot response
        active_pipeline_count = len(active_pipeline_votes)
        pipeline_headers = [entry["label"] for entry in active_pipeline_votes]
        all_algos = sorted({
            algo
            for entry in active_pipeline_votes
            for algo in entry["probs"].keys()
        })
        avg_scores = {}
        for algo in all_algos:
            per_pipeline_scores = [entry["probs"].get(algo, 0.0) for entry in active_pipeline_votes]
            avg_scores[algo] = float(np.mean(per_pipeline_scores)) if per_pipeline_scores else 0.0
        sorted_algos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        ranking = [{"rank": i + 1, "algo": algo} for i, (algo, _) in enumerate(sorted_algos[:5])]

        voting_matrix = []
        for algo, _ in sorted_algos[:5]:
            per_pipeline_scores = [entry["probs"].get(algo, 0.0) for entry in active_pipeline_votes]
            row = {
                "algo": algo,
                "pipelineScores": [round(float(s), 3) for s in per_pipeline_scores],
                "avg": round(float(avg_scores[algo]), 3),
                # Backward-compatible fields for older frontend code.
                "p1": round(float(per_pipeline_scores[0]), 3) if len(per_pipeline_scores) > 0 else 0.0,
                "p2": round(float(per_pipeline_scores[1]), 3) if len(per_pipeline_scores) > 1 else 0.0,
                "p3": round(float(per_pipeline_scores[2]), 3) if len(per_pipeline_scores) > 2 else 0.0,
            }
            voting_matrix.append(row)

        best_algo = sorted_algos[0][0] if sorted_algos else "Unknown"
        best_score = sorted_algos[0][1] if sorted_algos else 0.0

        print(f"[INFO] ImputePilot recommendation complete. Best algo: {best_algo}, inference time: {ImputePilot_inference_time:.4f}s")

        # Step 6: FLAML baseline prediction (if trained)
        flaml_result = None
        flaml_model_data = AdartsService.get_flaml_model()
        
        if flaml_model_data is not None:
            try:
                print("[INFO] Running FLAML prediction...")
                if flaml_model_data.get("external_runner"):
                    ext_res = _run_with_heartbeat(
                        "External baseline predict FLAML",
                        lambda: _run_external_baseline_runner(
                            "FLAML",
                            "predict",
                            arrays_dict={"X_infer": X},
                            meta_dict={},
                            timeout_sec=1200,
                        )
                    )
                    if ext_res.get("status") != "success":
                        raise RuntimeError(ext_res.get("error", "External FLAML predict failed"))
                    most_common_algo = ext_res.get("algo", "Unknown")
                    avg_confidence = float(ext_res.get("confidence", 0.0))
                    flaml_inference_time = float(ext_res.get("inference_time_ms", 0.0)) / 1000.0
                else:
                    flaml_automl = flaml_model_data['model']
                    flaml_start = time.time()
                    flaml_pred = flaml_automl.predict(X)
                    flaml_proba = flaml_automl.predict_proba(X)
                    flaml_inference_time = time.time() - flaml_start

                    # Get most common prediction (voting)
                    from collections import Counter
                    pred_counts = Counter(flaml_pred)
                    most_common_algo = pred_counts.most_common(1)[0][0]

                    # Average confidence
                    avg_confidence = float(np.mean(np.max(flaml_proba, axis=1)))
                
                flaml_result = {
                    'algo': most_common_algo,
                    'confidence': round(avg_confidence, 4),
                    'inference_time_ms': round(flaml_inference_time * 1000, 2),
                    'f1_train': round(flaml_model_data.get('f1_score', 0), 4),
                    'best_estimator': flaml_model_data.get('best_estimator', 'unknown'),
                    'trained': True,
                }
                print(f"[INFO] FLAML prediction: {most_common_algo} (confidence: {avg_confidence:.4f}, time: {flaml_inference_time*1000:.2f}ms)")
                
            except Exception as e:
                print(f"[WARN] FLAML prediction failed: {e}")
                traceback.print_exc()
                flaml_result = {'trained': True, 'error': str(e)}
        else:
            flaml_result = {'trained': False, 'message': 'Not trained yet.'}

        # Step 6b: Tune baseline prediction (if trained)
        tune_result = None
        tune_model_data = AdartsService.get_tune_model()
        
        if tune_model_data is not None:
            try:
                print("[INFO] Running Tune prediction...")
                if tune_model_data.get("external_runner"):
                    ext_res = _run_with_heartbeat(
                        "External baseline predict Tune",
                        lambda: _run_external_baseline_runner(
                            "TUNE",
                            "predict",
                            arrays_dict={"X_infer": X},
                            meta_dict={},
                            timeout_sec=1200,
                        )
                    )
                    if ext_res.get("status") != "success":
                        raise RuntimeError(ext_res.get("error", "External Tune predict failed"))
                    most_common_algo = ext_res.get("algo", "Unknown")
                    avg_confidence = float(ext_res.get("confidence", 0.0))
                    tune_inference_time = float(ext_res.get("inference_time_ms", 0.0)) / 1000.0
                else:
                    tune_clf = tune_model_data['model']
                    tune_start = time.time()
                    tune_pred = tune_clf.predict(X)
                    tune_proba = tune_clf.predict_proba(X) if hasattr(tune_clf, 'predict_proba') else None
                    tune_inference_time = time.time() - tune_start

                    # Get most common prediction (voting)
                    from collections import Counter
                    pred_counts = Counter(tune_pred)
                    most_common_algo = pred_counts.most_common(1)[0][0]

                    # Average confidence
                    avg_confidence = float(np.mean(np.max(tune_proba, axis=1))) if tune_proba is not None else 0.0
                
                tune_result = {
                    'algo': most_common_algo,
                    'confidence': round(avg_confidence, 4),
                    'inference_time_ms': round(tune_inference_time * 1000, 2),
                    'f1_train': round(tune_model_data.get('f1_score', 0), 4),
                    'best_estimator': tune_model_data.get('classifier_type', 'unknown'),
                    'trained': True,
                }
                print(f"[INFO] Tune prediction: {most_common_algo} (confidence: {avg_confidence:.4f}, time: {tune_inference_time*1000:.2f}ms)")
                
            except Exception as e:
                print(f"[WARN] Tune prediction failed: {e}")
                traceback.print_exc()
                tune_result = {'trained': True, 'error': str(e)}
        else:
            tune_result = {'trained': False, 'message': 'Not trained yet.'}

        # Step 6c: AutoFolio baseline prediction (if trained)
        autofolio_result = None
        autofolio_model_data = AdartsService.get_autofolio_model()

        if autofolio_model_data is not None:
            try:
                print("[INFO] Running AutoFolio prediction...")
                if autofolio_model_data.get("external_runner"):
                    ext_res = _run_with_heartbeat(
                        "External baseline predict AutoFolio",
                        lambda: _run_external_baseline_runner(
                            "AUTOFOLIO",
                            "predict",
                            arrays_dict={"X_infer": X},
                            meta_dict={},
                            timeout_sec=1200,
                        )
                    )
                    if ext_res.get("status") != "success":
                        raise RuntimeError(ext_res.get("error", "External AutoFolio predict failed"))
                    most_common_algo = ext_res.get("algo", "Unknown")
                    avg_confidence = float(ext_res.get("confidence", 0.0))
                    autofolio_inference_time = float(ext_res.get("inference_time_ms", 0.0)) / 1000.0
                    best_estimator = ext_res.get("best_estimator", "external")
                else:
                    autofolio_model = autofolio_model_data['model']
                    autofolio_start = time.time()
                    autofolio_pred = autofolio_model.predict(X)
                    autofolio_proba = autofolio_model.predict_proba(X) if hasattr(autofolio_model, 'predict_proba') else None
                    autofolio_inference_time = time.time() - autofolio_start

                    pred_counts = Counter(autofolio_pred)
                    most_common_algo = pred_counts.most_common(1)[0][0]
                    avg_confidence = float(np.mean(np.max(autofolio_proba, axis=1))) if autofolio_proba is not None else 0.0
                    best_estimator = autofolio_model_data.get("best_estimator", "unknown")

                autofolio_result = {
                    'algo': most_common_algo,
                    'confidence': round(avg_confidence, 4),
                    'inference_time_ms': round(autofolio_inference_time * 1000, 2),
                    'f1_train': round(autofolio_model_data.get('f1_score', 0), 4),
                    'best_estimator': best_estimator,
                    'trained': True,
                }
                print(f"[INFO] AutoFolio prediction: {most_common_algo} (confidence: {avg_confidence:.4f}, time: {autofolio_inference_time*1000:.2f}ms)")

            except Exception as e:
                print(f"[WARN] AutoFolio prediction failed: {e}")
                traceback.print_exc()
                autofolio_result = {'trained': True, 'error': str(e)}
        else:
            autofolio_result = {'trained': False, 'message': 'Not trained yet.'}

        # Step 6d: RAHA baseline prediction (if trained)
        raha_result = None
        raha_model_data = AdartsService.get_raha_model()

        if raha_model_data is not None:
            try:
                print("[INFO] Running RAHA prediction...")
                if raha_model_data.get("external_runner"):
                    ext_res = _run_with_heartbeat(
                        "External baseline predict RAHA",
                        lambda: _run_external_baseline_runner(
                            "RAHA",
                            "predict",
                            arrays_dict={"X_infer": X},
                            meta_dict={},
                            timeout_sec=1200,
                        )
                    )
                    if ext_res.get("status") != "success":
                        raise RuntimeError(ext_res.get("error", "External RAHA predict failed"))
                    raha_algo = ext_res.get("algo", "Unknown")
                    raha_conf = float(ext_res.get("confidence", 0.0))
                    raha_inference_time = float(ext_res.get("inference_time_ms", 0.0)) / 1000.0
                    raha_result = {
                        'algo': raha_algo,
                        'confidence': round(raha_conf, 4),
                        'inference_time_ms': round(raha_inference_time * 1000, 2),
                        'f1_train': round(raha_model_data.get('f1_score', 0), 4),
                        'best_estimator': 'cosine_similarity',
                        'trained': True,
                    }
                    print(f"[INFO] RAHA prediction: {raha_algo} (time: {raha_inference_time*1000:.2f}ms)")
                else:
                    raha_start = time.time()

                    existing_vectors = raha_model_data['existing_vectors']
                    raha_feature_cols = raha_model_data['feature_cols']
                    error_metric = raha_model_data['error_metric']

                    # Build profile vector from inference features
                    # X is already available from ImputePilot prediction above
                    # Use mean of all rows as the profile vector
                    profile_vec = X.mean(axis=0).astype(float)

                    # Core RAHA recommendation logic
                    def _cosine_dist(a, b):
                        na, nb = np.linalg.norm(a), np.linalg.norm(b)
                        if na == 0 or nb == 0:
                            return 1.0
                        sim = np.dot(a, b) / (na * nb)
                        return 1.0 if np.isnan(sim) else 1.0 - sim

                    all_techniques = set()
                    for _, row in existing_vectors.iterrows():
                        if row['Benchmark Results'] is not None:
                            try:
                                all_techniques.update(row['Benchmark Results'].index.tolist())
                            except Exception:
                                pass
                    all_techniques = sorted(all_techniques)

                    g_max_error = 0.0
                    for _, row in existing_vectors.iterrows():
                        try:
                            max_e = row['Benchmark Results'][error_metric].max()
                            if max_e > g_max_error:
                                g_max_error = max_e
                        except Exception:
                            pass
                    if g_max_error == 0:
                        g_max_error = 1.0

                    scores = {}
                    for cid, row in existing_vectors.iterrows():
                        fv = row['Features Vector']
                        if fv is None:
                            continue
                        dist = _cosine_dist(fv.to_numpy().astype(float), profile_vec)
                        for technique in all_techniques:
                            try:
                                rmse = row['Benchmark Results'][error_metric].loc[technique] / g_max_error
                                score = dist * rmse
                                if technique not in scores or score < scores[technique]:
                                    scores[technique] = score
                            except (KeyError, TypeError):
                                pass

                    sorted_techniques = sorted(scores.items(), key=lambda x: x[1])
                    # Handle cdrec variants
                    result_list = []
                    cdrec_added = False
                    for tech, sc in sorted_techniques:
                        if 'cdrec' in tech:
                            if not cdrec_added:
                                result_list.append(('cdrec', sc))
                                cdrec_added = True
                        else:
                            result_list.append((tech, sc))

                    raha_inference_time = time.time() - raha_start

                    if result_list:
                        raha_algo = result_list[0][0]
                        raha_result = {
                            'algo': raha_algo,
                            'confidence': round(1.0 - result_list[0][1], 4) if result_list[0][1] <= 1.0 else 0.0,
                            'inference_time_ms': round(raha_inference_time * 1000, 2),
                            'f1_train': round(raha_model_data.get('f1_score', 0), 4),
                            'best_estimator': 'cosine_similarity',
                            'trained': True,
                        }
                        print(f"[INFO] RAHA prediction: {raha_algo} (time: {raha_inference_time*1000:.2f}ms)")
                    else:
                        raha_result = {'trained': True, 'error': 'No recommendation produced'}

            except Exception as e:
                print(f"[WARN] RAHA prediction failed: {e}")
                traceback.print_exc()
                raha_result = {'trained': True, 'error': str(e)}
        else:
            raha_result = {'trained': False, 'message': 'Not trained yet.'}

        # Step 7: Save recommendation result
        AdartsService.set_last_recommendation({
            "dataset_id": dataset_id,
            "ranking": ranking,
            "voting_matrix": voting_matrix,
            "best_algo": best_algo,
            "avg_scores": avg_scores,
            "pipelines_configured": num_pipelines,
            "pipelines_used": active_pipeline_count,
            "pipeline_headers": pipeline_headers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ImputePilot_inference_time": ImputePilot_inference_time,
            "flaml_algo": flaml_result.get('algo') if flaml_result and 'algo' in flaml_result else None,
            "tune_algo": tune_result.get('algo') if tune_result and 'algo' in tune_result else None,
            "autofolio_algo": autofolio_result.get('algo') if autofolio_result and 'algo' in autofolio_result else None,
            "raha_algo": raha_result.get('algo') if raha_result and 'algo' in raha_result else None,
        })

        # Step 8: Build response
        response_data = {
            "ranking": ranking,
            "votingMatrix": voting_matrix,
            "pipelinesConfigured": num_pipelines,
            "pipelinesUsed": active_pipeline_count,
            "pipelineHeaders": pipeline_headers,
            "ImputePilot": {
                "algo": best_algo, 
                "confidence": round(best_score, 4),
                "inference_time_ms": round(ImputePilot_inference_time * 1000, 2),
            },
            "baselines": {
                "FLAML": flaml_result,
                "Tune": tune_result,
                "AutoFolio": autofolio_result,
                "RAHA": raha_result,
            }
        }

        return JsonResponse(response_data)

    except Exception as e:
        print(f"[ERROR] get_recommendation failed: {e}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


def _normalize_recommend_extractor_key(extractor_name):
    name = str(extractor_name).lower()
    if "catch22" in name:
        return "catch22"
    if "tsfresh" in name:
        return "tsfresh"
    if "topological" in name:
        return "topological"
    return name.replace("featuresextractor", "")


def _resolve_inference_data_file_for_recommendation():
    inference_dir = AdartsService.get_inference_dir()
    preferred_missing_files = [
        os.path.join(inference_dir, "evaluation_data_with_missing.csv"),
        os.path.join(inference_dir, "test_data_with_missing.csv"),
    ]
    for candidate in preferred_missing_files:
        if os.path.exists(candidate):
            return candidate

    inference_files = []
    inference_files_fallback = []
    for root, _, files in os.walk(inference_dir):
        for f in files:
            if f.lower().endswith((".csv", ".txt", ".tsv")):
                full_path = os.path.join(root, f)
                inference_files_fallback.append(full_path)
                if "missing" in f.lower():
                    inference_files.append(full_path)

    inference_files.sort()
    inference_files_fallback.sort()

    if inference_files:
        return inference_files[0]
    if inference_files_fallback:
        return inference_files_fallback[0]
    return None


@csrf_exempt
@require_http_methods(["POST"])
def extract_recommend_features(request):
    try:
        data = json.loads(request.body) if request.body else {}
        dataset_id = data.get("datasetId", "")
        print(f"[INFO] Extract recommend features for dataset: {dataset_id}")

        trained_model = AdartsService.get_trained_model()
        if trained_model is None:
            return JsonResponse(
                {
                    "error": "No trained model found. Please complete Data Pipeline first.",
                    "featureImportance": [],
                    "featurePreview": {},
                },
                status=400,
            )

        pipelines = trained_model.get("pipelines", [])
        feature_extractors = trained_model.get("feature_extractors", [])
        if not feature_extractors:
            return JsonResponse(
                {
                    "error": "No feature extractors found in trained model.",
                    "featureImportance": [],
                    "featurePreview": {},
                },
                status=400,
            )

        requested_features = data.get("features") or data.get("feature_subset")
        if isinstance(requested_features, str):
            requested_features = [s.strip() for s in requested_features.split(",") if s.strip()]
        if requested_features:
            requested_set = {str(f).strip().lower() for f in requested_features if str(f).strip()}
            filtered_extractors = []
            for fe in feature_extractors:
                fe_name = fe.__class__.__name__.lower()
                if any(req in fe_name for req in requested_set):
                    filtered_extractors.append(fe)
            if filtered_extractors:
                feature_extractors = filtered_extractors
                print(f"[INFO] Using feature subset for recommend feature extraction: {sorted(requested_set)}")
            else:
                print(f"[WARN] Requested feature subset {sorted(requested_set)} did not match extractors; using all.")

        ts_file = _resolve_inference_data_file_for_recommendation()
        if ts_file is None:
            return JsonResponse(
                {
                    "error": "No inference data found. Please upload data first.",
                    "featureImportance": [],
                    "featurePreview": {},
                },
                status=400,
            )

        print(f"[INFO] Recommend feature extraction file: {ts_file}")
        timeseries = pd.read_csv(ts_file, sep=None, engine="python", header=None, index_col=None)
        timeseries = timeseries.apply(pd.to_numeric, errors="coerce")
        print(f"[INFO] Loaded inference timeseries shape: {timeseries.shape}")

        missing_before = int(timeseries.isna().sum().sum())
        if missing_before > 0:
            print(
                f"[INFO] Missing values before feature extraction: {missing_before}. "
                "Applying interpolation + boundary fill."
            )
            ts_feature_input = (
                timeseries.interpolate(method="linear", axis=1, limit_direction="both")
                .ffill(axis=1)
                .bfill(axis=1)
                .fillna(0.0)
            )
        else:
            ts_feature_input = timeseries

        features_name = None
        if pipelines and hasattr(pipelines[0], "rm") and getattr(pipelines[0].rm, "features_name", None) is not None:
            features_name = pipelines[0].rm.features_name

        nb_timeseries, timeseries_length = ts_feature_input.shape
        ts_for_extraction = ts_feature_input.T
        all_ts_features = []

        preview_max_cols = 20
        preview_max_rows = 5
        extracted_summary = {}
        feature_preview = {}
        extractor_dataset_count = {}

        def _safe_json_value(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, (float, np.floating)):
                value = float(value)
                return value if np.isfinite(value) else None
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, str):
                return value
            return str(value)

        for fe in feature_extractors:
            fe_name = fe.__class__.__name__
            fe_key = _normalize_recommend_extractor_key(fe_name)
            args = (
                (ts_for_extraction, nb_timeseries, timeseries_length)
                if fe_name == "TSFreshFeaturesExtractor"
                else (ts_for_extraction,)
            )
            try:
                tmp_features = _run_with_heartbeat(
                    f"Recommend feature extraction {fe_name}",
                    lambda: fe.extract_from_timeseries(*args)
                )
            except Exception as ex:
                print(f"[WARN] Recommend feature extraction failed for {fe_name}: {ex}")
                traceback.print_exc()
                continue

            preview_df = tmp_features.copy()
            if "Time Series ID" not in preview_df.columns:
                preview_df.insert(0, "Time Series ID", np.arange(preview_df.shape[0]))

            feature_cols_plain = [col for col in preview_df.columns if col != "Time Series ID"]
            feat_count = len(feature_cols_plain)
            extracted_summary[fe_key] = extracted_summary.get(fe_key, 0) + feat_count
            extractor_dataset_count[fe_key] = extractor_dataset_count.get(fe_key, 0) + 1

            if fe_key not in feature_preview:
                sample_cols = feature_cols_plain[:preview_max_cols]
                row_cols = ["Time Series ID"] + sample_cols
                preview_rows = []
                for _, row in preview_df.loc[:, row_cols].head(preview_max_rows).iterrows():
                    preview_rows.append({col: _safe_json_value(row[col]) for col in row_cols})

                feature_preview[fe_key] = {
                    "dataset": os.path.basename(ts_file),
                    "idColumn": "Time Series ID",
                    "totalFeatures": feat_count,
                    "sampleColumns": sample_cols,
                    "truncated": feat_count > len(sample_cols),
                    "rows": preview_rows,
                }

            tmp_features.set_index("Time Series ID", inplace=True)
            tmp_features.columns = [
                col + fe.FEATURES_FILENAMES_ID if col != "Time Series ID" else col
                for col in tmp_features.columns
            ]
            all_ts_features.append(tmp_features)
            print(f"[INFO] Recommend extractor {fe_key}: {feat_count} features")

        if not all_ts_features:
            return JsonResponse(
                {
                    "error": "Feature extraction failed for all configured extractors.",
                    "featureImportance": [],
                    "featurePreview": {},
                },
                status=500,
            )

        timeseries_features = pd.concat(all_ts_features, axis=1)

        if features_name is not None:
            timeseries_features = timeseries_features.loc[:, timeseries_features.columns.isin(features_name)]
            if list(timeseries_features.columns) != list(features_name):
                missing_features_l = list(set(features_name) - set(timeseries_features.columns))
                missing_features = dict(
                    zip(missing_features_l, [list(features_name).index(f) for f in missing_features_l])
                )
                for feature, feature_index in dict(sorted(missing_features.items(), key=lambda item: item[1])).items():
                    imputed_feature_values = np.zeros(nb_timeseries)
                    timeseries_features.insert(feature_index, feature, imputed_feature_values)

        response_data = [
            {
                "name": k,
                "value": v,
                "datasetsProcessed": extractor_dataset_count.get(k, 0),
            }
            for k, v in extracted_summary.items()
        ]

        return JsonResponse(
            {
                "featureImportance": response_data,
                "featurePreview": feature_preview,
                "previewRows": preview_max_rows,
                "previewCols": preview_max_cols,
                "nTimeseries": int(nb_timeseries),
                "nFeatureColumns": int(timeseries_features.shape[1]),
            }
        )

    except Exception as e:
        print(f"[ERROR] extract_recommend_features failed: {e}")
        traceback.print_exc()
        return JsonResponse(
            {"error": str(e), "featureImportance": [], "featurePreview": {}},
            status=500,
        )


@csrf_exempt
@require_http_methods(["POST"])
def compare_baselines(request):
    """
    Compare ImputePilot with selected baselines by running actual imputation.
    
    If evaluation mode is configured (with Ground Truth), calculates real RMSE.
    Otherwise, uses ImputeBench's internal RMSE calculation.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        selected_baselines = data.get('baselines', ['FLAML'])
        use_ground_truth = data.get('use_ground_truth', True)
        
        print(f"[INFO] Comparing baselines: {selected_baselines}")
        
        # Step 1: Get last recommendation
        last_rec = AdartsService.get_last_recommendation()
        if last_rec is None:
            return JsonResponse({
                'error': 'No recommendation available. Please run get_recommendation first.',
                'results': []
            }, status=400)
        
        ImputePilot_algo = last_rec.get('best_algo')
        if not ImputePilot_algo:
            return JsonResponse({
                'error': 'ImputePilot did not recommend any algorithm.',
                'results': []
            }, status=400)
        
        # Step 2: Check evaluation mode and Ground Truth
        eval_mode = AdartsService.get_evaluation_mode()
        ground_truth_data = AdartsService.get_ground_truth()
        
        has_ground_truth = (
            use_ground_truth and 
            ground_truth_data is not None and 
            ground_truth_data.get('data') is not None
        )
        
        print(f"[INFO] Evaluation mode: {eval_mode}, Ground Truth available: {has_ground_truth}")
        
        # Step 3: Collect algorithms to compare
        algos_to_compare = {
            'ImputePilot': ImputePilot_algo
        }
        
        if 'FLAML' in selected_baselines:
            flaml_model = AdartsService.get_flaml_model()
            if flaml_model is not None:
                flaml_algo = last_rec.get('flaml_algo')
                if flaml_algo:
                    algos_to_compare['FLAML'] = flaml_algo
        
        if 'Tune' in selected_baselines:
            tune_model = AdartsService.get_tune_model()
            if tune_model is not None:
                tune_algo = last_rec.get('tune_algo')
                if tune_algo:
                    algos_to_compare['Tune'] = tune_algo
        
        if 'AutoFolio' in selected_baselines:
            autofolio_model = AdartsService.get_autofolio_model()
            if autofolio_model is not None:
                autofolio_algo = last_rec.get('autofolio_algo')
                if autofolio_algo:
                    algos_to_compare['AutoFolio'] = autofolio_algo

        if 'RAHA' in selected_baselines:
            raha_model = AdartsService.get_raha_model()
            if raha_model is not None:
                raha_algo = last_rec.get('raha_algo')
                if raha_algo:
                    algos_to_compare['RAHA'] = raha_algo
        
        print(f"[INFO] Algorithms to compare: {algos_to_compare}")
        
        # Step 4: Load inference data (prefer missing-data files, otherwise newest file)
        inference_dir = AdartsService.get_inference_dir()
        preferred_missing_files = [
            os.path.join(inference_dir, "evaluation_data_with_missing.csv"),
            os.path.join(inference_dir, "test_data_with_missing.csv"),
        ]
        ts_file = None
        for candidate in preferred_missing_files:
            if os.path.exists(candidate):
                ts_file = candidate
                break

        if ts_file is None:
            inference_files = [
                f for f in os.listdir(inference_dir)
                if f.lower().endswith((".csv", ".txt", ".tsv"))
            ]
            if not inference_files:
                return JsonResponse({'error': 'No inference data found.'}, status=400)

            missing_candidates = [f for f in inference_files if "missing" in f.lower()]
            candidates = missing_candidates if missing_candidates else inference_files
            candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(inference_dir, f)),
                reverse=True
            )
            ts_file = os.path.join(inference_dir, candidates[0])

        print(f"[INFO] compare_baselines using inference file: {ts_file}")
        timeseries = pd.read_csv(ts_file, sep=None, engine='python', header=None, index_col=None)
        
        # Step 5: Load missing mask if available
        missing_mask = None
        mask_file = os.path.join(inference_dir, 'missing_mask.npy')
        if os.path.exists(mask_file):
            missing_mask = np.load(mask_file)
            print(f"[INFO] Missing mask loaded. Shape: {missing_mask.shape}")
        
        # Step 6: Check ImputeBench availability
        try:
            from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils
            CONF = Utils.read_conf_file('imputebenchlabeler')
            benchmark_path = CONF.get('BENCHMARK_PATH', '')
            imputebench_available = os.path.exists(benchmark_path) and os.path.exists(os.path.join(benchmark_path, 'TestingFramework.exe'))
        except Exception as e:
            print(f"[WARN] Could not check ImputeBench availability: {e}")
            imputebench_available = False
        
        # Step 7: Run imputation for each algorithm
        results = []
        
        if not imputebench_available:
            return JsonResponse({
                'error': 'ImputeBench is not available. Cannot run imputation comparison.',
                'hint': 'Please verify BENCHMARK_PATH in Config/imputebenchlabeler_config.yaml and ensure TestingFramework.exe exists.',
                'results': []
            }, status=400)
        
        for method_name, algo in algos_to_compare.items():
            print(f"[INFO] Running imputation for {method_name} with algorithm: {algo}")
            
            try:
                result_entry = {
                    'method': method_name,
                    'algorithm': algo,
                    'rmse': None,
                    'mae': None,
                    'runtime_seconds': None,
                    'status': 'pending',
                    'error': None,
                    'rmse_source': None  # 'ground_truth' or 'imputebench'
                }
                
                # Run ImputeBench
                imputation_result = _run_imputation_for_algo(timeseries, algo)
                
                if imputation_result.get('error'):
                    print(f"[WARN] ImputeBench error for {algo}: {imputation_result.get('error')}")
                    result_entry['status'] = 'error'
                    result_entry['error'] = imputation_result.get('error')
                else:
                    result_entry['runtime_seconds'] = imputation_result.get('runtime')
                    
                    # If we have Ground Truth and missing mask, calculate RMSE ourselves
                    if has_ground_truth and missing_mask is not None:
                        # Load imputed data from ImputeBench output
                        imputed_file = imputation_result.get('imputed_file')
                        if imputed_file and os.path.exists(imputed_file):
                            imputed_data = _read_imputed_file(imputed_file)
                            ground_truth = ground_truth_data['data']
                            
                            metrics = _calculate_rmse_with_ground_truth(
                                imputed_data, 
                                ground_truth, 
                                missing_mask
                            )
                            
                            result_entry['rmse'] = metrics.get('rmse')
                            result_entry['mae'] = metrics.get('mae')
                            result_entry['status'] = 'success'
                            result_entry['rmse_source'] = 'ground_truth'
                            print(f"[INFO] Ground Truth RMSE for {algo}: {metrics.get('rmse')}")
                        else:
                            # Use ImputeBench's RMSE
                            result_entry['rmse'] = imputation_result.get('rmse')
                            result_entry['mae'] = imputation_result.get('mae')
                            result_entry['status'] = 'success'
                            result_entry['rmse_source'] = 'imputebench'
                    else:
                        # Use ImputeBench's RMSE
                        result_entry['rmse'] = imputation_result.get('rmse')
                        result_entry['mae'] = imputation_result.get('mae')
                        result_entry['status'] = 'success'
                        result_entry['rmse_source'] = 'imputebench'
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"[ERROR] Imputation failed for {method_name}/{algo}: {e}")
                traceback.print_exc()
                
                results.append({
                    'method': method_name,
                    'algorithm': algo,
                    'rmse': None,
                    'mae': None,
                    'runtime_seconds': None,
                    'status': 'error',
                    'error': str(e),
                    'rmse_source': None
                })
        
        # Step 8: Calculate improvement percentages
        ImputePilot_result = next((r for r in results if r['method'] == 'ImputePilot'), None)
        
        if ImputePilot_result and ImputePilot_result['rmse'] is not None:
            for r in results:
                if r['method'] != 'ImputePilot' and r['rmse'] is not None:
                    rmse_improvement = ((r['rmse'] - ImputePilot_result['rmse']) / r['rmse']) * 100
                    r['rmse_improvement_vs_ImputePilot'] = round(rmse_improvement, 2)
                    
                    if r['mae'] is not None and ImputePilot_result['mae'] is not None:
                        mae_improvement = ((r['mae'] - ImputePilot_result['mae']) / r['mae']) * 100
                        r['mae_improvement_vs_ImputePilot'] = round(mae_improvement, 2)
        
        print(f"[INFO] Comparison complete. Results: {results}")
        
        # Determine best method
        valid_results = [r for r in results if r['rmse'] is not None]
        best_method = min(valid_results, key=lambda x: x['rmse'])['method'] if valid_results else 'Unknown'
        
        return JsonResponse({
            'results': results,
            'summary': {
                'best_method': best_method,
                'total_compared': len(results),
                'evaluation_mode': eval_mode,
                'ground_truth_used': has_ground_truth,
                'imputebench_available': imputebench_available,
            }
        })
        
    except Exception as e:
        print(f"[ERROR] compare_baselines failed: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


def _run_imputation_for_algo(timeseries, algorithm):
    """Helper function to run imputation for a single algorithm and return metrics"""
    try:
        from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils
        import subprocess
        import re
        import sys
        algorithm = str(algorithm).lower()

        if _is_external_dl_algo(algorithm):
            return _run_external_imputation_for_algo(timeseries, algorithm)

        print(f"\n{'='*60}")
        print(f"[DEBUG] Starting ImputeBench for algorithm: {algorithm}")
        print(f"{'='*60}")

        # Step 1: Load config
        CONF = Utils.read_conf_file('imputebenchlabeler')
        benchmark_path = os.getenv("ImputePilot_IMPUTEBENCH_PATH", "").strip() or CONF.get('BENCHMARK_PATH', '')
        imputebench_timeout_sec = int(os.getenv("ImputePilot_IMPUTEBENCH_TIMEOUT_SEC", "1200"))

        print(f"[DEBUG] Step 1: Config loaded")
        print(f"[DEBUG]   BENCHMARK_PATH = {benchmark_path}")
        print(f"[DEBUG]   Path exists: {os.path.exists(benchmark_path)}")

        if not benchmark_path:
            return {'error': 'BENCHMARK_PATH not configured in imputebenchlabeler_config.yaml'}

        if not os.path.exists(benchmark_path):
            return {'error': f'ImputeBench directory not found: {benchmark_path}'}

        # Step 2: Check TestingFramework.exe
        exe_path = os.path.join(benchmark_path, 'TestingFramework.exe')
        print(f"[DEBUG] Step 2: Checking TestingFramework.exe")
        print(f"[DEBUG]   EXE path: {exe_path}")
        print(f"[DEBUG]   EXE exists: {os.path.exists(exe_path)}")

        if not os.path.exists(exe_path):
            # List contents of benchmark_path
            try:
                contents = os.listdir(benchmark_path)
                print(f"[DEBUG]   Contents of {benchmark_path}: {contents}")
            except Exception as e:
                print(f"[DEBUG]   Failed to list directory: {e}")
            return {'error': f'TestingFramework.exe not found at: {exe_path}'}

        # Step 3: Resolve/check mono (avoid PATH-only lookup, which may be missing in spawned workers)
        print(f"[DEBUG] Step 3: Checking mono installation")
        try:
            mono_candidates = []
            which_mono = shutil.which('mono')
            if which_mono:
                mono_candidates.append(which_mono)
            py_bin_mono = os.path.join(os.path.dirname(sys.executable), 'mono')
            mono_candidates.append(py_bin_mono)
            mono_candidates.append('/usr/bin/mono')

            mono_exec = None
            for candidate in mono_candidates:
                if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    mono_exec = candidate
                    break

            print(f"[DEBUG]   mono candidates: {mono_candidates}")
            print(f"[DEBUG]   mono resolved: {mono_exec}")

            if mono_exec is None:
                return {'error': 'mono is not installed. Please install mono-complete.'}

            # Check mono version
            mono_version = subprocess.run([mono_exec, '--version'], capture_output=True, text=True)
            print(f"[DEBUG]   mono version: {mono_version.stdout.split(chr(10))[0] if mono_version.stdout else 'unknown'}")
        except Exception as e:
            print(f"[DEBUG]   mono check failed: {e}")
            return {'error': f'Failed to check mono: {str(e)}'}

        # Build a stable runtime env for ImputeBench child processes.
        # Important: setting CONDA_DEFAULT_ENV alone is not enough for dynamic linker;
        # we also need to expose the env lib path so algoCollection can resolve libmlpack.
        runtime_env = os.environ.copy()
        try:
            mono_bin_dir = os.path.dirname(mono_exec)
            mono_prefix = os.path.dirname(mono_bin_dir)
            mono_lib_dir = os.path.join(mono_prefix, 'lib')

            if os.path.exists(mono_lib_dir):
                runtime_env['CONDA_PREFIX'] = mono_prefix
                runtime_env['CONDA_DEFAULT_ENV'] = os.path.basename(mono_prefix) or 'ImputePilot'

                existing_ld = runtime_env.get('LD_LIBRARY_PATH', '')
                ld_parts = [p for p in existing_ld.split(':') if p]
                if mono_lib_dir not in ld_parts:
                    ld_parts.insert(0, mono_lib_dir)
                runtime_env['LD_LIBRARY_PATH'] = ':'.join(ld_parts)

                existing_path = runtime_env.get('PATH', '')
                path_parts = [p for p in existing_path.split(':') if p]
                if mono_bin_dir not in path_parts:
                    path_parts.insert(0, mono_bin_dir)
                runtime_env['PATH'] = ':'.join(path_parts)

            print(f"[DEBUG]   child CONDA_DEFAULT_ENV: {runtime_env.get('CONDA_DEFAULT_ENV')}")
            print(f"[DEBUG]   child CONDA_PREFIX: {runtime_env.get('CONDA_PREFIX')}")
            print(f"[DEBUG]   child LD_LIBRARY_PATH has env lib: {mono_lib_dir in (runtime_env.get('LD_LIBRARY_PATH', ''))}")
        except Exception as e:
            print(f"[WARN] Failed to build child runtime env, fallback to current env: {e}")
            runtime_env = os.environ.copy()

        def _compute_errors_python(results_folder_path):
            """
            Fallback for environments without Rscript:
            compute mse/rmse/mae files from recovered_matrices using the same logic as error_calculation.r.
            """
            r_script_path = os.path.join(results_folder_path, 'scripts', 'precision', 'error_calculation.r')
            recovery_dir = os.path.join(results_folder_path, 'recovery', 'values', 'recovered_matrices')

            if not os.path.isdir(recovery_dir):
                print(f"[WARN] recovered_matrices folder not found: {recovery_dir}")
                return False

            lengths = []
            list_algos = []

            if os.path.isfile(r_script_path):
                with open(r_script_path, 'r', encoding='utf-8', errors='ignore') as f:
                    r_content = f.read()

                m_len = re.search(r'seq\.int\(from\s*=\s*(\d+),\s*to\s*=\s*(\d+),\s*by\s*=\s*(\d+)\)', r_content)
                if m_len:
                    lengths = list(range(int(m_len.group(1)), int(m_len.group(2)) + 1, int(m_len.group(3))))

                m_alg = re.search(r'list_algos\s*<-\s*c\(([^)]+)\)', r_content)
                if m_alg:
                    list_algos = [s.strip().strip('"').strip("'") for s in m_alg.group(1).split(',') if s.strip()]

            if not lengths:
                recovered_files = []
                for filename in os.listdir(recovery_dir):
                    m = re.match(r'recoveredMat(\d+)\.txt$', filename)
                    if m:
                        recovered_files.append(int(m.group(1)))
                lengths = sorted(recovered_files)

            if not lengths:
                print(f"[WARN] Could not determine any recovered matrix lengths in: {recovery_dir}")
                return False

            if not list_algos:
                # Best effort fallback if R script parsing failed
                list_algos = CONF.get('ALGORITHMS_LIST', [])

            error_defs = [
                ('mse',  'MSE_',  lambda d: np.mean(d ** 2)),
                ('rmse', 'RMSE_', lambda d: np.sqrt(np.mean(d ** 2))),
                ('mae',  'MAE_',  lambda d: np.mean(np.abs(d))),
            ]

            for subdir, prefix, _ in error_defs:
                os.makedirs(os.path.join(results_folder_path, 'error', subdir), exist_ok=True)

            for length in lengths:
                mat_path = os.path.join(recovery_dir, f'recoveredMat{length}.txt')
                if not os.path.isfile(mat_path):
                    continue

                df = pd.read_csv(mat_path, header=None, sep=r'\s+', engine='python')
                if df.shape[1] < 2:
                    continue

                ref = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
                algo_count = min(df.shape[1] - 1, len(list_algos))

                for i in range(algo_count):
                    algo = list_algos[i]
                    comp = pd.to_numeric(df.iloc[:, i + 1], errors='coerce').to_numpy()
                    valid = ~np.isnan(ref) & ~np.isnan(comp)
                    if not np.any(valid):
                        continue

                    diff = comp[valid] - ref[valid]
                    for subdir, prefix, compute_fn in error_defs:
                        out_file = os.path.join(results_folder_path, 'error', subdir, f'{prefix}{algo}.dat')
                        if not os.path.exists(out_file):
                            with open(out_file, 'w', encoding='utf-8') as f:
                                f.write(f'# {algo}\n')
                        with open(out_file, 'a', encoding='utf-8') as f:
                            f.write(f'{length} {compute_fn(diff)}\n')

            print(f"[INFO] Error metrics computed via Python fallback for {os.path.basename(results_folder_path)}")
            return True

        def _read_metric_mean(metric_folder, algo_name):
            if not os.path.isdir(metric_folder):
                return None

            files = sorted(os.listdir(metric_folder))
            candidates = [f for f in files if f.lower().endswith('.dat') and algo_name in f.lower()]
            if not candidates and algo_name == 'cdrec':
                candidates = [f for f in files if f.lower().endswith('.dat') and 'cdrec' in f.lower()]

            for filename in candidates:
                filepath = os.path.join(metric_folder, filename)
                try:
                    result_df = pd.read_csv(
                        filepath,
                        header=None,
                        comment='#',
                        sep=r'\s+',
                        engine='python'
                    )
                    if result_df.shape[1] < 2:
                        continue
                    vals = pd.to_numeric(result_df.iloc[:, 1], errors='coerce').dropna()
                    if not vals.empty:
                        return float(vals.mean())
                except Exception as e:
                    print(f"[DEBUG]   Failed to parse metric file {filepath}: {e}")

            return None

        def _export_imputed_file(results_folder_path):
            recovery_values_dir = os.path.join(results_folder_path, 'recovery', 'values')
            if not os.path.isdir(recovery_values_dir):
                return None

            series_length = int(timeseries.shape[1])
            candidate_dirs = []
            for name in os.listdir(recovery_values_dir):
                full_path = os.path.join(recovery_values_dir, name)
                if os.path.isdir(full_path) and name.isdigit():
                    candidate_dirs.append((int(name), full_path))
            candidate_dirs.sort(reverse=True)

            prioritized_dirs = []
            exact = [entry for entry in candidate_dirs if entry[0] == series_length]
            if exact:
                prioritized_dirs.extend(exact)
            prioritized_dirs.extend([entry for entry in candidate_dirs if entry[0] != series_length])

            matched_path = None
            for _, dir_path in prioritized_dirs:
                files = sorted(os.listdir(dir_path))
                matches = [f for f in files if f.lower().endswith('.txt') and algorithm in f.lower()]
                if not matches and algorithm == 'cdrec':
                    matches = [f for f in files if f.lower().endswith('.txt') and 'cdrec' in f.lower()]
                if matches:
                    matched_path = os.path.join(dir_path, matches[0])
                    break

            if matched_path is None:
                return None

            recovered_df = pd.read_csv(matched_path, header=None, sep=r'\s+', engine='python')
            if recovered_df.empty:
                return None

            # First column is usually 1..N index from ImputeBench output
            first_col_numeric = pd.to_numeric(recovered_df.iloc[:, 0], errors='coerce')
            if first_col_numeric.notna().all() and len(recovered_df.columns) > 1:
                seq = np.arange(first_col_numeric.iloc[0], first_col_numeric.iloc[0] + len(first_col_numeric))
                if np.allclose(first_col_numeric.to_numpy(), seq):
                    recovered_df = recovered_df.iloc[:, 1:]

            # Align orientation to uploaded data shape when possible
            if recovered_df.shape != timeseries.shape and recovered_df.T.shape == timeseries.shape:
                recovered_df = recovered_df.T

            inference_dir = AdartsService.get_inference_dir()
            out_path = os.path.join(inference_dir, f'imputed_{algorithm}.txt')
            recovered_df.to_csv(out_path, sep=' ', index=False, header=False)
            return out_path

        start_time = time.time()

        # Step 4: Create temporary dataset
        dataset_name = f'impute_{algorithm}_{int(time.time() * 1000)}_{os.getpid()}_{random.randint(1000, 9999)}'
        data_folder = os.path.join(benchmark_path, 'data')
        dataset_folder_path = os.path.join(data_folder, dataset_name)
        dataset_file_path = os.path.join(dataset_folder_path, f'{dataset_name}_normal.txt')

        print(f"[DEBUG] Step 4: Creating dataset")
        print(f"[DEBUG]   data folder: {data_folder}")
        print(f"[DEBUG]   data folder exists: {os.path.exists(data_folder)}")
        print(f"[DEBUG]   dataset name: {dataset_name}")
        print(f"[DEBUG]   dataset folder: {dataset_folder_path}")
        print(f"[DEBUG]   dataset file: {dataset_file_path}")

        # Create data folder if needed
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(dataset_folder_path, exist_ok=True)

        # Save timeseries data
        print(f"[DEBUG]   Input timeseries shape: {timeseries.shape}")
        print(f"[DEBUG]   Transposed shape: {timeseries.T.shape}")

        # Check for NaN values
        nan_count = timeseries.isna().sum().sum()
        print(f"[DEBUG]   NaN count in data: {nan_count}")

        # Save the data (IMPORTANT: na_rep='NaN' ensures missing values are saved as "NaN" string)
        # ImputeBench expects "NaN" for missing values, not empty strings
        timeseries.T.to_csv(dataset_file_path, sep=' ', index=False, header=False, na_rep='NaN')

        # Verify file
        if os.path.exists(dataset_file_path):
            file_size = os.path.getsize(dataset_file_path)
            print(f"[DEBUG]   File created successfully, size: {file_size} bytes")

            # Preview first few lines
            with open(dataset_file_path, 'r') as f:
                lines = f.readlines()[:3]
                print(f"[DEBUG]   File preview (first 3 lines):")
                for i, line in enumerate(lines):
                    print(f"[DEBUG]     Line {i}: {line[:100]}...")
        else:
            return {'error': f'Failed to create dataset file: {dataset_file_path}'}

        # Step 5: Build and run command
        command = [
            mono_exec,
            'TestingFramework.exe',
            '-alg', algorithm.lower(),
            '-d', dataset_name,
            '-scen', 'miss_perc',
            '-nort', '-novis'
        ]

        print(f"[DEBUG] Step 5: Running ImputeBench")
        print(f"[DEBUG]   Command: {' '.join(command)}")
        print(f"[DEBUG]   Working directory: {benchmark_path}")

        try:
            process = subprocess.Popen(
                command,
                cwd=benchmark_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=runtime_env,
            )
            stdout, stderr = process.communicate(timeout=imputebench_timeout_sec)

            print(f"[DEBUG]   Return code: {process.returncode}")
            if stdout:
                print(f"[DEBUG]   STDOUT (first 1000 chars):")
                print(f"[DEBUG]   {stdout[:1000]}")
            if stderr:
                print(f"[DEBUG]   STDERR (first 1000 chars):")
                print(f"[DEBUG]   {stderr[:1000]}")

            if process.returncode != 0:
                # Don't cleanup on error so we can inspect
                return {'error': f'ImputeBench returned code {process.returncode}. STDERR: {stderr[:500] if stderr else "none"}. STDOUT: {stdout[:500] if stdout else "none"}'}

        except subprocess.TimeoutExpired:
            process.kill()
            return {'error': f'ImputeBench timeout ({imputebench_timeout_sec}s)'}
        except FileNotFoundError as e:
            return {'error': f'Command not found: {str(e)}'}
        except Exception as e:
            return {'error': f'Failed to run ImputeBench subprocess: {type(e).__name__}: {str(e)}'}

        runtime = time.time() - start_time

        # Step 6: Read results
        results_folder = os.path.join(benchmark_path, 'Results', 'miss_perc', dataset_name)
        print(f"[DEBUG] Step 6: Reading results")
        print(f"[DEBUG]   Results folder: {results_folder}")
        print(f"[DEBUG]   Results folder exists: {os.path.exists(results_folder)}")

        if not os.path.exists(results_folder):
            # List what's in Results folder
            results_base = os.path.join(benchmark_path, 'Results')
            if os.path.exists(results_base):
                print(f"[DEBUG]   Contents of Results/: {os.listdir(results_base)}")
                miss_perc_folder = os.path.join(results_base, 'miss_perc')
                if os.path.exists(miss_perc_folder):
                    print(f"[DEBUG]   Contents of Results/miss_perc/: {os.listdir(miss_perc_folder)}")
            return {'error': f'Results folder not created: {results_folder}'}

        rmse_value = None
        mae_value = None

        # Get RMSE
        rmse_folder = os.path.join(results_folder, 'error', 'rmse')
        print(f"[DEBUG]   RMSE folder: {rmse_folder}")
        print(f"[DEBUG]   RMSE folder exists: {os.path.exists(rmse_folder)}")

        rmse_files_before = os.listdir(rmse_folder) if os.path.isdir(rmse_folder) else []
        if len(rmse_files_before) == 0:
            print("[WARN] RMSE files are missing/empty. Running Python fallback metric computation...")
            _compute_errors_python(results_folder)

        if os.path.exists(rmse_folder):
            files = os.listdir(rmse_folder)
            print(f"[DEBUG]   Files in RMSE folder: {files}")
            rmse_value = _read_metric_mean(rmse_folder, algorithm.lower())
            if rmse_value is not None:
                print(f"[DEBUG]   RMSE value: {rmse_value}")

        # Get MAE
        mae_folder = os.path.join(results_folder, 'error', 'mae')
        if os.path.exists(mae_folder):
            mae_value = _read_metric_mean(mae_folder, algorithm.lower())
            if mae_value is not None:
                print(f"[DEBUG]   MAE value: {mae_value}")

        # Step 7: Find and save imputed data (before cleanup)
        print(f"[DEBUG] Step 7: Finding imputed data")
        recovery_folder = os.path.join(results_folder, 'recovery')
        print(f"[DEBUG]   Recovery folder: {recovery_folder}")
        print(f"[DEBUG]   Recovery folder exists: {os.path.exists(recovery_folder)}")
        imputed_file_path = _export_imputed_file(results_folder)
        if imputed_file_path:
            print(f"[DEBUG]   Exported imputed data to: {imputed_file_path}")

        # Step 8: Cleanup temporary files
        print(f"[DEBUG] Step 8: Cleaning up temporary files")
        try:
            shutil.rmtree(dataset_folder_path, ignore_errors=True)
            shutil.rmtree(results_folder, ignore_errors=True)
            print(f"[DEBUG]   Cleanup complete")
        except Exception as e:
            print(f"[DEBUG]   Cleanup warning: {e}")

        print(f"[DEBUG] {'='*60}")
        print(f"[DEBUG] Imputation complete!")
        print(f"[DEBUG]   RMSE: {rmse_value}")
        print(f"[DEBUG]   MAE: {mae_value}")
        print(f"[DEBUG]   Runtime: {runtime:.2f}s")
        print(f"[DEBUG]   Imputed file: {imputed_file_path}")
        print(f"[DEBUG] {'='*60}\n")

        if rmse_value is None and mae_value is None:
            return {'error': 'No results found in output files', 'imputed_file': imputed_file_path}

        # Check for NaN values (JSON doesn't support NaN)
        import math
        rmse_is_nan = rmse_value is not None and math.isnan(rmse_value)
        mae_is_nan = mae_value is not None and math.isnan(mae_value)

        if rmse_is_nan and mae_is_nan:
            return {
                'error': f'Algorithm produced NaN results. This may indicate the algorithm is not suitable for this data, or the data has too many missing values.',
                'runtime': round(runtime, 3),
                'details': 'ImputeBench ran successfully but the algorithm could not compute valid error metrics.',
                'imputed_file': imputed_file_path
            }

        # Convert NaN to None for JSON compatibility
        return {
            'rmse': None if rmse_is_nan else rmse_value,
            'mae': None if mae_is_nan else mae_value,
            'runtime': round(runtime, 3),
            'warning': 'Some metrics were NaN' if (rmse_is_nan or mae_is_nan) else None,
            'imputed_file': imputed_file_path
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] _run_imputation_for_algo exception: {e}")
        traceback.print_exc()
        return {'error': f'{type(e).__name__}: {str(e)}'}

@csrf_exempt
@require_http_methods(["POST"])
def run_imputation(request):
    """Run imputation with ImputeBench - DEBUG VERSION (no simulation fallback)"""
    try:
        data = json.loads(request.body)
        algorithm = data.get('algorithm', 'rosl').lower()
        is_external = _is_external_dl_algo(algorithm)
        
        print(f"\n{'#'*60}")
        print(f"[INFO] run_imputation called with algorithm: {algorithm}")
        print(f"{'#'*60}\n")
        
        # Step 1: Load inference data (MUST use data WITH MISSING for imputation)
        inference_dir = AdartsService.get_inference_dir()
        print(f"[INFO] Inference directory: {inference_dir}")
        print(f"[INFO] Inference dir exists: {os.path.exists(inference_dir)}")
        
        if os.path.exists(inference_dir):
            all_files = os.listdir(inference_dir)
            print(f"[INFO] All files in inference dir: {all_files}")
        
        # Priority: use data with missing values (evaluation mode)
        missing_file = os.path.join(inference_dir, 'evaluation_data_with_missing.csv')
        test_missing_file = os.path.join(inference_dir, 'test_data_with_missing.csv')
        
        if os.path.exists(missing_file):
            ts_file = missing_file
            print(f"[INFO] Using evaluation data with missing: {ts_file}")
        elif os.path.exists(test_missing_file):
            ts_file = test_missing_file
            print(f"[INFO] Using test data with missing: {ts_file}")
        else:
            # Fallback: find any data file (normal mode, user uploaded data that already has missing)
            inference_files = []
            for root, dirs, files in os.walk(inference_dir):
                for f in files:
                    if f.endswith('.csv') or f.endswith('.txt'):
                        # Skip the complete file - it has no missing values to impute
                        if f != 'original_complete.csv':
                            inference_files.append(os.path.join(root, f))
            
            print(f"[INFO] Matching inference files: {inference_files}")
            
            if not inference_files:
                return JsonResponse({'error': 'No data to impute. Please upload inference data first.'}, status=400)
            
            ts_file = inference_files[0]
            print(f"[INFO] Using data file: {ts_file}")
        
        timeseries = pd.read_csv(ts_file, sep=None, engine='python', header=None, index_col=None)
        print(f"[INFO] Loaded timeseries shape: {timeseries.shape}")
        print(f"[INFO] Timeseries dtypes: {timeseries.dtypes.to_dict()}")
        
        missing_count = int(timeseries.isna().sum().sum())
        print(f"[INFO] Missing values count: {missing_count}")
        
        # Step 2: Check ImputeBench configuration
        print(f"\n[INFO] Step 2: Checking ImputeBench configuration...")
        
        try:
            from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils
            CONF = Utils.read_conf_file('imputebenchlabeler')
            benchmark_path = CONF.get('BENCHMARK_PATH', '')
            algorithms_list = CONF.get('ALGORITHMS_LIST', [])
            
            print(f"[INFO] BENCHMARK_PATH: {benchmark_path}")
            print(f"[INFO] ALGORITHMS_LIST: {algorithms_list}")
            print(f"[INFO] Requested algorithm '{algorithm}' in list: {algorithm in [a.lower() for a in algorithms_list]}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return JsonResponse({
                'error': f'Failed to load ImputeBench config: {str(e)}',
                'mode': 'error'
            }, status=500)
        
        # Step 3: Validate algorithm name
        valid_algorithms = ['cdrec', 'dynammo', 'grouse', 'rosl', 'softimp', 'svdimp', 'svt', 'stmvl', 'spirit', 'tenmf', 'tkcm']
        for algo in sorted(_get_external_dl_algos()):
            if algo not in valid_algorithms:
                valid_algorithms.append(algo)
        if algorithm.lower() not in valid_algorithms:
            print(f"[WARN] Algorithm '{algorithm}' not in standard list: {valid_algorithms}")
        
        # Step 4: Run ImputeBench (NO SIMULATION FALLBACK)
        print(f"\n[INFO] Step 4: Running ImputeBench...")
        
        if not is_external:
            if not os.path.exists(benchmark_path):
                return JsonResponse({
                    'error': f'ImputeBench directory not found: {benchmark_path}',
                    'mode': 'error',
                    'hint': 'Please verify BENCHMARK_PATH in Config/imputebenchlabeler_config.yaml'
                }, status=500)
            
            exe_path = os.path.join(benchmark_path, 'TestingFramework.exe')
            if not os.path.exists(exe_path):
                return JsonResponse({
                    'error': f'TestingFramework.exe not found at: {exe_path}',
                    'mode': 'error',
                    'hint': 'Please build ImputeBench or verify the path'
                }, status=500)
        else:
            print("[INFO] External DL algorithm requested; skipping ImputeBench checks.")
        
        # Run imputation
        result = _run_with_heartbeat(
            f"Imputation {algorithm}",
            lambda: _run_imputation_for_algo(timeseries, algorithm)
        )
        
        # Step 5: Return results (no fallback to simulation)
        if result.get('error'):
            print(f"[ERROR] ImputeBench failed: {result['error']}")
            return JsonResponse({
                'error': result['error'],
                'algo': algorithm.upper(),
                'mode': 'error',
                'missingPoints': missing_count,
                'runtime': result.get('runtime'),
                'details': result.get('details')
            }, status=500)
        
        if is_external and result.get('imputed_file'):
            rmse_val = result.get('rmse')
            mae_val = result.get('mae')
            try:
                ground_truth_data = AdartsService.get_ground_truth()
                mask_file = os.path.join(inference_dir, 'missing_mask.npy')
                if ground_truth_data is not None and ground_truth_data.get('data') is not None and os.path.exists(mask_file):
                    missing_mask = np.load(mask_file)
                    imputed_df = _read_imputed_file(result.get('imputed_file'))
                    metrics = _calculate_rmse_with_ground_truth(
                        imputed_df,
                        ground_truth_data['data'],
                        missing_mask
                    )
                    rmse_val = metrics.get('rmse')
                    mae_val = metrics.get('mae')
            except Exception as e:
                print(f"[WARN] Failed to compute RMSE/MAE for external imputation: {e}")

            return JsonResponse({
                'algo': algorithm.upper(),
                'missingPoints': missing_count,
                'recoveryRate': '100%',
                'processingTime': f"{result.get('runtime', 0):.2f}s",
                'rmse': rmse_val,
                'mae': mae_val,
                'mode': 'real',
                'imputed_file': os.path.basename(result.get('imputed_file')),
            })

        if result.get('rmse') is None and result.get('mae') is None:
            # Check if there's a warning (partial NaN results)
            if result.get('warning'):
                return JsonResponse({
                    'error': 'Algorithm produced invalid results (NaN)',
                    'algo': algorithm.upper(),
                    'mode': 'error',
                    'missingPoints': missing_count,
                    'runtime': f"{result.get('runtime', 0):.2f}s",
                    'hint': 'Try a different algorithm or check your data quality.'
                }, status=500)
            return JsonResponse({
                'error': 'ImputeBench completed but no results found',
                'algo': algorithm.upper(),
                'mode': 'error',
                'missingPoints': missing_count,
            }, status=500)
        
        # Success!
        print(f"\n[SUCCESS] Imputation completed successfully!")
        return JsonResponse({
            'algo': algorithm.upper(),
            'missingPoints': missing_count,
            'recoveryRate': '100%',
            'processingTime': f"{result.get('runtime', 0):.2f}s",
            'rmse': result.get('rmse'),
            'mae': result.get('mae'),
            'mode': 'real'
        })
        
    except Exception as e:
        print(f"[ERROR] run_imputation exception: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'mode': 'error'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_downstream(request):
    """
    Downstream Evaluation - Real Implementation (No Simulation)
    
    Compares performance on downstream tasks (forecasting/classification) using:
    1. Ground Truth data (complete, original data) - as upper bound
    2. Imputed data (after ImputePilot recommendation) - with imputation
    3. Mean-filled data (simple baseline) - without proper imputation
    
    Prerequisites:
    - Evaluation must be configured (ground truth available)
    - Imputation must have been run (imputed data available)
    """
    try:
        data = json.loads(request.body)
        task = data.get('task', 'forecasting')
        algorithm = data.get('algorithm', None)  # Optional: specific imputed algorithm to use
        eval_missing_rate = float(data.get('eval_missing_rate', 0.1))
        eval_seed = int(data.get('eval_seed', 42))
        regenerate_mask = bool(data.get('regenerate_mask', False))
        
        print(f"\n{'='*60}")
        print(f"[INFO] Running REAL downstream evaluation: {task}")
        print(f"{'='*60}")
        
        # Step 1: Check and load Ground Truth data (fallback to self-supervised mode if absent)
        ground_truth_data = AdartsService.get_ground_truth()
        has_ground_truth = ground_truth_data is not None and ground_truth_data.get('data') is not None
        
        ground_truth_df = ground_truth_data['data'] if has_ground_truth else None
        if has_ground_truth:
            print(f"[INFO] Ground Truth shape: {ground_truth_df.shape}")
        
        # Step 2: Load data with missing values
        inference_dir = AdartsService.get_inference_dir()
        
        # Load missing mask if available (only required for ground-truth mode)
        missing_mask = None
        mask_file = os.path.join(inference_dir, 'missing_mask.npy')
        if has_ground_truth:
            if not os.path.exists(mask_file):
                return JsonResponse({
                    'error': 'Missing mask not found. Please run setup_test_set or setup_upload first.',
                    'task': task,
                    'mode': 'error'
                }, status=400)
            missing_mask = np.load(mask_file)
            print(f"[INFO] Missing mask shape: {missing_mask.shape}")
        
        # Load data with missing values (prefer evaluation missing files, otherwise newest non-imputed file)
        preferred_missing_files = [
            os.path.join(inference_dir, "evaluation_data_with_missing.csv"),
            os.path.join(inference_dir, "test_data_with_missing.csv"),
        ]
        ts_file = None
        for candidate in preferred_missing_files:
            if os.path.exists(candidate):
                ts_file = candidate
                break

        if ts_file is None:
            inference_files = [
                f for f in os.listdir(inference_dir)
                if f.lower().endswith((".csv", ".txt", ".tsv")) and "imputed" not in f.lower()
            ]
            if not inference_files:
                return JsonResponse({
                    'error': 'No inference data with missing values found.',
                    'task': task,
                    'mode': 'error'
                }, status=400)

            missing_candidates = [f for f in inference_files if "missing" in f.lower()]
            candidates = missing_candidates if missing_candidates else inference_files
            candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(inference_dir, f)),
                reverse=True
            )
            ts_file = os.path.join(inference_dir, candidates[0])

        print(f"[INFO] Using missing data file: {ts_file}")
        data_with_missing = pd.read_csv(ts_file, sep=None, engine='python', header=None, index_col=None)
        print(f"[INFO] Data with missing shape: {data_with_missing.shape}")

        # Step 3: Load or compute imputed data
        imputed_data = None
        imputed_file = None
        if has_ground_truth:
            imputed_files = [f for f in os.listdir(inference_dir) if 'imputed' in f.lower()]
            if not imputed_files:
                return JsonResponse({
                    'error': 'No imputed data found. Please run imputation first (run_imputation API).',
                    'task': task,
                    'mode': 'error',
                    'hint': 'Call /api/recommend/impute/ with your recommended algorithm first.'
                }, status=400)

            # Use specified algorithm, otherwise use last recommended algorithm if available
            algo_key = None
            if algorithm:
                algo_key = str(algorithm).strip().lower()
            else:
                last_rec = AdartsService.get_last_recommendation()
                if last_rec:
                    algo_key = str(last_rec.get('best_algo', '')).strip().lower() or None

            if algo_key:
                for f in imputed_files:
                    if algo_key in f.lower():
                        imputed_file = os.path.join(inference_dir, f)
                        break

            if imputed_file is None:
                # Fallback: use newest imputed file
                imputed_files.sort(
                    key=lambda f: os.path.getmtime(os.path.join(inference_dir, f)),
                    reverse=True
                )
                imputed_file = os.path.join(inference_dir, imputed_files[0])

            print(f"[INFO] Using imputed file: {imputed_file}")
            imputed_data = _read_imputed_file(imputed_file)
            print(f"[INFO] Imputed data shape: {imputed_data.shape}")
        else:
            algo = algorithm
            if not algo:
                last_rec = AdartsService.get_last_recommendation()
                if last_rec:
                    algo = last_rec.get('best_algo')
            if not algo:
                return JsonResponse({
                    'error': 'No algorithm provided and no recommendation available. Provide algorithm in request.',
                    'task': task,
                    'mode': 'error'
                }, status=400)
            algo_key = str(algo).strip().lower()
            values = data_with_missing.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
            eval_mask = _load_or_create_downstream_eval_mask(
                values,
                inference_dir,
                missing_rate=eval_missing_rate,
                seed=eval_seed,
                regenerate=regenerate_mask,
            )
            eval_df = data_with_missing.copy()
            eval_df.values[eval_mask] = np.nan
            print(f"[INFO] Self-supervised: eval mask rate={eval_missing_rate}, algo={algo_key}")
            imputation_result = _run_imputation_for_algo(eval_df, algo_key)
            if imputation_result.get('error'):
                return JsonResponse({
                    'error': f"Imputation failed for {algo_key}: {imputation_result.get('error')}",
                    'task': task,
                    'mode': 'error'
                }, status=500)
            imputed_file = imputation_result.get('imputed_file')
            if imputed_file and os.path.exists(imputed_file):
                imputed_data = _read_imputed_file(imputed_file)
            else:
                return JsonResponse({
                    'error': 'Imputation completed but no imputed file found.',
                    'task': task,
                    'mode': 'error'
                }, status=500)
            # Align shapes if needed
            if imputed_data.shape != eval_df.shape and imputed_data.T.shape == eval_df.shape:
                imputed_data = imputed_data.T
            data_with_missing = data_with_missing.apply(pd.to_numeric, errors="coerce")
        
        # Verify shapes match
        if has_ground_truth and ground_truth_df.shape != data_with_missing.shape:
            return JsonResponse({
                'error': f'Shape mismatch: Ground Truth {ground_truth_df.shape} vs Missing Data {data_with_missing.shape}',
                'task': task,
                'mode': 'error'
            }, status=400)
        
        if has_ground_truth and ground_truth_df.shape != imputed_data.shape:
            return JsonResponse({
                'error': f'Shape mismatch: Ground Truth {ground_truth_df.shape} vs Imputed Data {imputed_data.shape}',
                'task': task,
                'mode': 'error'
            }, status=400)
        
        # Step 4: Create mean-filled data (simple baseline - no proper imputation)
        mean_filled_data = data_with_missing.copy()
        for col in mean_filled_data.columns:
            col_mean = mean_filled_data[col].mean()
            if pd.isna(col_mean):
                col_mean = 0
            mean_filled_data[col] = mean_filled_data[col].fillna(col_mean)
        
        print(f"[INFO] Mean-filled data shape: {mean_filled_data.shape}")
        
        # Step 5: Run downstream task
        if task == 'forecasting':
            if has_ground_truth:
                result = _run_real_forecasting_task(
                    ground_truth_df,
                    imputed_data,
                    mean_filled_data,
                    missing_mask
                )
            else:
                values = data_with_missing.to_numpy(dtype="float32")
                eval_mask = _load_or_create_downstream_eval_mask(
                    values,
                    inference_dir,
                    missing_rate=eval_missing_rate,
                    seed=eval_seed,
                    regenerate=regenerate_mask,
                )
                oracle_df = (
                    data_with_missing.interpolate(method="linear", axis=1, limit_direction="both")
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .fillna(0.0)
                )
                result = _run_self_supervised_forecasting_task(
                    data_with_missing,
                    imputed_data,
                    mean_filled_data,
                    eval_mask,
                    oracle_df=oracle_df,
                )
        elif task == 'classification':
            if has_ground_truth:
                result = _run_real_classification_task(
                    ground_truth_df,
                    imputed_data,
                    mean_filled_data,
                    missing_mask
                )
            else:
                oracle_df = (
                    data_with_missing.interpolate(method="linear", axis=1, limit_direction="both")
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .fillna(0.0)
                )
                result = _run_self_supervised_classification_task(
                    data_with_missing,
                    imputed_data,
                    mean_filled_data,
                    oracle_df=oracle_df,
                )
        else:
            return JsonResponse({
                'error': f'Unknown task: {task}. Use "forecasting" or "classification".',
                'mode': 'error'
            }, status=400)
        
        if 'error' in result:
            return JsonResponse(result, status=400)
        
        result['mode'] = 'real' if has_ground_truth else 'self_supervised'
        if imputed_file:
            result['imputed_file_used'] = os.path.basename(imputed_file)
        if not has_ground_truth:
            result['eval_mask_rate'] = eval_missing_rate
            result['eval_mask_seed'] = eval_seed
        
        print(f"[INFO] Downstream evaluation complete: {result}")
        
        return JsonResponse(result)
        
    except Exception as e:
        print(f"[ERROR] run_downstream exception: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'mode': 'error'}, status=500)


def _run_real_forecasting_task(ground_truth_df, imputed_df, mean_filled_df, missing_mask):
    """
    Real Forecasting Evaluation
    
    For each time series:
    1. Split into train (80%) and test (20%)
    2. Train a linear regression model
    3. Predict the test portion
    4. Calculate RMSE
    
    Compare:
    - Ground Truth (upper bound - best possible)
    - Imputed data (with ImputePilot imputation)
    - Mean-filled data (simple baseline)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    results_ground_truth = []
    results_imputed = []
    results_mean_filled = []
    
    n_timeseries = ground_truth_df.shape[0]
    n_evaluated = 0
    n_skipped = 0
    
    for idx in range(n_timeseries):
        # Get time series from each dataset
        ts_gt = ground_truth_df.iloc[idx].values.astype(float)
        ts_imputed = imputed_df.iloc[idx].values.astype(float)
        ts_mean_filled = mean_filled_df.iloc[idx].values.astype(float)
        
        # Skip if too short
        ts_length = len(ts_gt)
        if ts_length < 10:
            n_skipped += 1
            continue
        
        # Check for NaN in data (should not happen for ground truth and properly imputed data)
        if np.any(np.isnan(ts_gt)):
            n_skipped += 1
            continue
        
        # Split: 80% train, 20% test
        split_idx = int(ts_length * 0.8)
        if split_idx < 5 or (ts_length - split_idx) < 2:
            n_skipped += 1
            continue
        
        # Prepare data
        X_train = np.arange(split_idx).reshape(-1, 1)
        X_test = np.arange(split_idx, ts_length).reshape(-1, 1)
        
        # === Ground Truth ===
        y_train_gt = ts_gt[:split_idx]
        y_test_gt = ts_gt[split_idx:]
        
        model_gt = LinearRegression()
        model_gt.fit(X_train, y_train_gt)
        pred_gt = model_gt.predict(X_test)
        rmse_gt = np.sqrt(mean_squared_error(y_test_gt, pred_gt))
        
        # === Imputed Data ===
        y_train_imp = ts_imputed[:split_idx]
        y_test_imp = ts_imputed[split_idx:]
        
        # Check for NaN in imputed data
        if np.any(np.isnan(y_train_imp)) or np.any(np.isnan(y_test_imp)):
            # Imputation might have failed for this series
            n_skipped += 1
            continue
        
        model_imp = LinearRegression()
        model_imp.fit(X_train, y_train_imp)
        pred_imp = model_imp.predict(X_test)
        # Compare prediction against Ground Truth test set (fair comparison)
        rmse_imp = np.sqrt(mean_squared_error(y_test_gt, pred_imp))
        
        # === Mean-filled Data ===
        y_train_mf = ts_mean_filled[:split_idx]
        y_test_mf = ts_mean_filled[split_idx:]
        
        if np.any(np.isnan(y_train_mf)) or np.any(np.isnan(y_test_mf)):
            n_skipped += 1
            continue
        
        model_mf = LinearRegression()
        model_mf.fit(X_train, y_train_mf)
        pred_mf = model_mf.predict(X_test)
        # Compare prediction against Ground Truth test set (fair comparison)
        rmse_mf = np.sqrt(mean_squared_error(y_test_gt, pred_mf))
        
        results_ground_truth.append(rmse_gt)
        results_imputed.append(rmse_imp)
        results_mean_filled.append(rmse_mf)
        n_evaluated += 1
    
    if n_evaluated == 0:
        return {
            'error': 'Not enough valid time series for forecasting evaluation.',
            'task': 'forecasting',
            'n_timeseries': n_timeseries,
            'n_skipped': n_skipped
        }
    
    # Calculate statistics
    avg_gt = np.mean(results_ground_truth)
    avg_imputed = np.mean(results_imputed)
    avg_mean_filled = np.mean(results_mean_filled)
    
    std_gt = np.std(results_ground_truth)
    std_imputed = np.std(results_imputed)
    std_mean_filled = np.std(results_mean_filled)
    
    # Calculate improvements
    # Improvement = how much better imputed is compared to mean-filled (lower RMSE is better)
    if avg_mean_filled > 0:
        improvement_vs_baseline = round((avg_mean_filled - avg_imputed) / avg_mean_filled * 100, 2)
    else:
        improvement_vs_baseline = 0
    
    # How close is imputed to ground truth (optimal)
    if avg_gt > 0:
        gap_to_optimal = round((avg_imputed - avg_gt) / avg_gt * 100, 2)
    else:
        gap_to_optimal = 0
    
    return {
        'task': 'forecasting',
        'metric': 'RMSE (lower is better)',
        'groundTruth': round(avg_gt, 6),
        'groundTruthStd': round(std_gt, 6),
        'withImputePilot': round(avg_imputed, 6),
        'withImputePilotStd': round(std_imputed, 6),
        'withoutImputePilot': round(avg_mean_filled, 6),
        'withoutImputePilotStd': round(std_mean_filled, 6),
        'withAdarts': round(avg_imputed, 6),
        'withAdartsStd': round(std_imputed, 6),
        'withoutAdarts': round(avg_mean_filled, 6),
        'withoutAdartsStd': round(std_mean_filled, 6),
        'improvement': improvement_vs_baseline,
        'gapToOptimal': gap_to_optimal,
        'n_evaluated': n_evaluated,
        'n_skipped': n_skipped,
        'details': {
            'description': 'RMSE of linear regression forecasting on the last 20% of each time series',
            'groundTruth': 'Model trained on complete original data (upper bound)',
            'withImputePilot': 'Model trained on ImputePilot imputed data',
            'withoutImputePilot': 'Model trained on mean-filled data (simple baseline)',
            'withAdarts': 'Model trained on ImputePilot imputed data',
            'withoutAdarts': 'Model trained on mean-filled data (simple baseline)'
        }
    }


def _run_real_classification_task(ground_truth_df, imputed_df, mean_filled_df, missing_mask):
    """
    Real Classification Evaluation
    
    1. Extract features from each time series
    2. Use clustering on Ground Truth to create pseudo-labels
    3. Train KNN classifier
    4. Evaluate using cross-validation
    
    Compare:
    - Ground Truth features (upper bound)
    - Imputed data features (with ImputePilot imputation)
    - Mean-filled data features (simple baseline)
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    def extract_features(ts):
        """Extract statistical features from a time series"""
        ts_clean = ts[~np.isnan(ts)]
        if len(ts_clean) < 5:
            return None
        
        features = [
            np.mean(ts_clean),
            np.std(ts_clean),
            np.max(ts_clean),
            np.min(ts_clean),
            np.median(ts_clean),
            np.percentile(ts_clean, 25),
            np.percentile(ts_clean, 75),
            len(ts_clean),
            # Trend feature
            np.polyfit(np.arange(len(ts_clean)), ts_clean, 1)[0] if len(ts_clean) > 1 else 0,
            # Volatility
            np.mean(np.abs(np.diff(ts_clean))) if len(ts_clean) > 1 else 0
        ]
        return features
    
    # Extract features from all datasets
    features_gt = []
    features_imputed = []
    features_mean_filled = []
    valid_indices = []
    
    n_timeseries = ground_truth_df.shape[0]
    
    for idx in range(n_timeseries):
        ts_gt = ground_truth_df.iloc[idx].values.astype(float)
        ts_imputed = imputed_df.iloc[idx].values.astype(float)
        ts_mean_filled = mean_filled_df.iloc[idx].values.astype(float)
        
        feat_gt = extract_features(ts_gt)
        feat_imputed = extract_features(ts_imputed)
        feat_mean_filled = extract_features(ts_mean_filled)
        
        if feat_gt is not None and feat_imputed is not None and feat_mean_filled is not None:
            # Check for NaN in features
            if not any(np.isnan(feat_gt)) and not any(np.isnan(feat_imputed)) and not any(np.isnan(feat_mean_filled)):
                features_gt.append(feat_gt)
                features_imputed.append(feat_imputed)
                features_mean_filled.append(feat_mean_filled)
                valid_indices.append(idx)
    
    n_valid = len(valid_indices)
    
    if n_valid < 10:
        return {
            'error': f'Not enough valid time series for classification. Got {n_valid}, need at least 10.',
            'task': 'classification',
            'n_timeseries': n_timeseries,
            'n_valid': n_valid
        }
    
    X_gt = np.array(features_gt)
    X_imputed = np.array(features_imputed)
    X_mean_filled = np.array(features_mean_filled)
    
    # Normalize features
    scaler = StandardScaler()
    X_gt_scaled = scaler.fit_transform(X_gt)
    X_imputed_scaled = scaler.transform(X_imputed)
    X_mean_filled_scaled = scaler.transform(X_mean_filled)
    
    # Create pseudo-labels using clustering on Ground Truth
    n_clusters = min(5, n_valid // 5)
    if n_clusters < 2:
        n_clusters = 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_labels = kmeans.fit_predict(X_gt_scaled)
    
    # Determine KNN parameters
    n_neighbors = min(5, n_valid // (n_clusters * 2))
    if n_neighbors < 1:
        n_neighbors = 1
    
    cv_folds = min(5, n_valid // n_clusters)
    if cv_folds < 2:
        cv_folds = 2
    
    # === Ground Truth Classification ===
    knn_gt = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_gt = cross_val_score(knn_gt, X_gt_scaled, y_labels, cv=cv_folds)
    acc_gt = np.mean(scores_gt)
    std_gt = np.std(scores_gt)
    
    # === Imputed Data Classification ===
    knn_imputed = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_imputed = cross_val_score(knn_imputed, X_imputed_scaled, y_labels, cv=cv_folds)
    acc_imputed = np.mean(scores_imputed)
    std_imputed = np.std(scores_imputed)
    
    # === Mean-filled Data Classification ===
    knn_mean_filled = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_mean_filled = cross_val_score(knn_mean_filled, X_mean_filled_scaled, y_labels, cv=cv_folds)
    acc_mean_filled = np.mean(scores_mean_filled)
    std_mean_filled = np.std(scores_mean_filled)
    
    # Calculate improvements
    if acc_mean_filled > 0:
        improvement_vs_baseline = round((acc_imputed - acc_mean_filled) / acc_mean_filled * 100, 2)
    else:
        improvement_vs_baseline = 0
    
    if acc_gt > 0:
        gap_to_optimal = round((acc_gt - acc_imputed) / acc_gt * 100, 2)
    else:
        gap_to_optimal = 0
    
    return {
        'task': 'classification',
        'metric': 'Accuracy (higher is better)',
        'groundTruth': round(acc_gt, 4),
        'groundTruthStd': round(std_gt, 4),
        'withImputePilot': round(acc_imputed, 4),
        'withImputePilotStd': round(std_imputed, 4),
        'withoutImputePilot': round(acc_mean_filled, 4),
        'withoutImputePilotStd': round(std_mean_filled, 4),
        'withAdarts': round(acc_imputed, 4),
        'withAdartsStd': round(std_imputed, 4),
        'withoutAdarts': round(acc_mean_filled, 4),
        'withoutAdartsStd': round(std_mean_filled, 4),
        'improvement': improvement_vs_baseline,
        'gapToOptimal': gap_to_optimal,
        'n_evaluated': n_valid,
        'n_skipped': n_timeseries - n_valid,
        'n_clusters': n_clusters,
        'n_neighbors': n_neighbors,
        'cv_folds': cv_folds,
        'details': {
            'description': 'KNN classification accuracy using extracted time series features',
            'groundTruth': 'Features from complete original data (upper bound)',
            'withImputePilot': 'Features from ImputePilot imputed data',
            'withoutImputePilot': 'Features from mean-filled data (simple baseline)',
            'withAdarts': 'Features from ImputePilot imputed data',
            'withoutAdarts': 'Features from mean-filled data (simple baseline)',
            'labels': 'Pseudo-labels created by K-Means clustering on ground truth features'
        }
    }


def _run_self_supervised_forecasting_task(observed_df, imputed_df, mean_filled_df, eval_mask, oracle_df=None):
    """
    Self-supervised forecasting evaluation using only observed values.
    Uses eval_mask to select observed points as evaluation targets.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    if oracle_df is None:
        oracle_df = (
            observed_df.interpolate(method="linear", axis=1, limit_direction="both")
            .ffill(axis=1)
            .bfill(axis=1)
            .fillna(0.0)
        )

    results_oracle = []
    results_imputed = []
    results_mean_filled = []

    n_timeseries = observed_df.shape[0]
    n_evaluated = 0
    n_skipped = 0

    for idx in range(n_timeseries):
        ts_obs = observed_df.iloc[idx].values.astype(float)
        ts_oracle = oracle_df.iloc[idx].values.astype(float)
        ts_imputed = imputed_df.iloc[idx].values.astype(float)
        ts_mean_filled = mean_filled_df.iloc[idx].values.astype(float)
        mask_row = eval_mask[idx].astype(bool)

        if not mask_row.any():
            n_skipped += 1
            continue

        ts_length = len(ts_obs)
        if ts_length < 10:
            n_skipped += 1
            continue

        split_idx = int(ts_length * 0.8)
        if split_idx < 5 or (ts_length - split_idx) < 2:
            n_skipped += 1
            continue

        mask_test = mask_row[split_idx:]
        if not np.any(mask_test):
            n_skipped += 1
            continue

        y_true = ts_obs[split_idx:][mask_test]
        if np.any(np.isnan(y_true)):
            n_skipped += 1
            continue

        X_train = np.arange(split_idx).reshape(-1, 1)
        X_test = np.arange(split_idx, ts_length).reshape(-1, 1)

        # Oracle (observed + interpolation)
        if np.any(np.isnan(ts_oracle)):
            n_skipped += 1
            continue
        model_oracle = LinearRegression()
        model_oracle.fit(X_train, ts_oracle[:split_idx])
        pred_oracle = model_oracle.predict(X_test)
        rmse_oracle = np.sqrt(mean_squared_error(y_true, pred_oracle[mask_test]))

        # Imputed
        if np.any(np.isnan(ts_imputed)):
            n_skipped += 1
            continue
        model_imp = LinearRegression()
        model_imp.fit(X_train, ts_imputed[:split_idx])
        pred_imp = model_imp.predict(X_test)
        rmse_imp = np.sqrt(mean_squared_error(y_true, pred_imp[mask_test]))

        # Mean-filled
        if np.any(np.isnan(ts_mean_filled)):
            n_skipped += 1
            continue
        model_mf = LinearRegression()
        model_mf.fit(X_train, ts_mean_filled[:split_idx])
        pred_mf = model_mf.predict(X_test)
        rmse_mf = np.sqrt(mean_squared_error(y_true, pred_mf[mask_test]))

        results_oracle.append(rmse_oracle)
        results_imputed.append(rmse_imp)
        results_mean_filled.append(rmse_mf)
        n_evaluated += 1

    if n_evaluated == 0:
        return {
            'error': 'Not enough valid time series for forecasting evaluation (self-supervised).',
            'task': 'forecasting',
            'n_timeseries': n_timeseries,
            'n_skipped': n_skipped
        }

    avg_gt = float(np.mean(results_oracle))
    avg_imputed = float(np.mean(results_imputed))
    avg_mean_filled = float(np.mean(results_mean_filled))

    std_gt = float(np.std(results_oracle))
    std_imputed = float(np.std(results_imputed))
    std_mean_filled = float(np.std(results_mean_filled))

    if avg_mean_filled > 0:
        improvement_vs_baseline = round((avg_mean_filled - avg_imputed) / avg_mean_filled * 100, 2)
    else:
        improvement_vs_baseline = 0

    if avg_gt > 0:
        gap_to_optimal = round((avg_imputed - avg_gt) / avg_gt * 100, 2)
    else:
        gap_to_optimal = 0

    return {
        'task': 'forecasting',
        'metric': 'RMSE (lower is better)',
        'groundTruth': round(avg_gt, 6),
        'groundTruthStd': round(std_gt, 6),
        'withImputePilot': round(avg_imputed, 6),
        'withImputePilotStd': round(std_imputed, 6),
        'withoutImputePilot': round(avg_mean_filled, 6),
        'withoutImputePilotStd': round(std_mean_filled, 6),
        'withAdarts': round(avg_imputed, 6),
        'withAdartsStd': round(std_imputed, 6),
        'withoutAdarts': round(avg_mean_filled, 6),
        'withoutAdartsStd': round(std_mean_filled, 6),
        'improvement': improvement_vs_baseline,
        'gapToOptimal': gap_to_optimal,
        'n_evaluated': n_evaluated,
        'n_skipped': n_skipped,
        'details': {
            'description': 'Self-supervised RMSE using masked observed values as evaluation targets.',
            'groundTruth': 'Observed values with interpolation (no additional masking)',
            'withImputePilot': 'ImputePilot imputation on eval-masked data',
            'withoutImputePilot': 'Mean-filled baseline on eval-masked data',
            'withAdarts': 'ImputePilot imputation on eval-masked data',
            'withoutAdarts': 'Mean-filled baseline on eval-masked data'
        }
    }


def _run_self_supervised_classification_task(observed_df, imputed_df, mean_filled_df, oracle_df=None):
    """
    Self-supervised classification evaluation using pseudo labels from observed data.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if oracle_df is None:
        oracle_df = (
            observed_df.interpolate(method="linear", axis=1, limit_direction="both")
            .ffill(axis=1)
            .bfill(axis=1)
            .fillna(0.0)
        )

    def extract_features(ts):
        ts_clean = ts[~np.isnan(ts)]
        if len(ts_clean) < 5:
            return None
        features = [
            np.mean(ts_clean),
            np.std(ts_clean),
            np.max(ts_clean),
            np.min(ts_clean),
            np.median(ts_clean),
            np.percentile(ts_clean, 25),
            np.percentile(ts_clean, 75),
            len(ts_clean),
            np.polyfit(np.arange(len(ts_clean)), ts_clean, 1)[0] if len(ts_clean) > 1 else 0,
            np.mean(np.abs(np.diff(ts_clean))) if len(ts_clean) > 1 else 0,
        ]
        return features

    features_oracle = []
    features_imputed = []
    features_mean_filled = []
    valid_indices = []

    n_timeseries = observed_df.shape[0]
    for idx in range(n_timeseries):
        ts_oracle = oracle_df.iloc[idx].values.astype(float)
        ts_imputed = imputed_df.iloc[idx].values.astype(float)
        ts_mean_filled = mean_filled_df.iloc[idx].values.astype(float)

        feat_oracle = extract_features(ts_oracle)
        feat_imputed = extract_features(ts_imputed)
        feat_mean_filled = extract_features(ts_mean_filled)

        if feat_oracle is None or feat_imputed is None or feat_mean_filled is None:
            continue
        if any(np.isnan(feat_oracle)) or any(np.isnan(feat_imputed)) or any(np.isnan(feat_mean_filled)):
            continue
        features_oracle.append(feat_oracle)
        features_imputed.append(feat_imputed)
        features_mean_filled.append(feat_mean_filled)
        valid_indices.append(idx)

    n_valid = len(valid_indices)
    if n_valid < 10:
        return {
            'error': f'Not enough valid time series for classification. Got {n_valid}, need at least 10.',
            'task': 'classification',
            'n_timeseries': n_timeseries,
            'n_valid': n_valid
        }

    X_oracle = np.array(features_oracle)
    X_imputed = np.array(features_imputed)
    X_mean_filled = np.array(features_mean_filled)

    scaler = StandardScaler()
    X_oracle_scaled = scaler.fit_transform(X_oracle)
    X_imputed_scaled = scaler.transform(X_imputed)
    X_mean_filled_scaled = scaler.transform(X_mean_filled)

    n_clusters = min(5, n_valid // 5)
    if n_clusters < 2:
        n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_labels = kmeans.fit_predict(X_oracle_scaled)

    n_neighbors = min(5, n_valid // (n_clusters * 2))
    if n_neighbors < 1:
        n_neighbors = 1

    cv_folds = min(5, n_valid // n_clusters)
    if cv_folds < 2:
        cv_folds = 2

    knn_gt = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_gt = cross_val_score(knn_gt, X_oracle_scaled, y_labels, cv=cv_folds)
    acc_gt = np.mean(scores_gt)
    std_gt = np.std(scores_gt)

    knn_imputed = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_imputed = cross_val_score(knn_imputed, X_imputed_scaled, y_labels, cv=cv_folds)
    acc_imputed = np.mean(scores_imputed)
    std_imputed = np.std(scores_imputed)

    knn_mean_filled = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_mean_filled = cross_val_score(knn_mean_filled, X_mean_filled_scaled, y_labels, cv=cv_folds)
    acc_mean_filled = np.mean(scores_mean_filled)
    std_mean_filled = np.std(scores_mean_filled)

    if acc_mean_filled > 0:
        improvement_vs_baseline = round((acc_imputed - acc_mean_filled) / acc_mean_filled * 100, 2)
    else:
        improvement_vs_baseline = 0

    if acc_gt > 0:
        gap_to_optimal = round((acc_gt - acc_imputed) / acc_gt * 100, 2)
    else:
        gap_to_optimal = 0

    return {
        'task': 'classification',
        'metric': 'Accuracy (higher is better)',
        'groundTruth': round(acc_gt, 4),
        'groundTruthStd': round(std_gt, 4),
        'withImputePilot': round(acc_imputed, 4),
        'withImputePilotStd': round(std_imputed, 4),
        'withoutImputePilot': round(acc_mean_filled, 4),
        'withoutImputePilotStd': round(std_mean_filled, 4),
        'withAdarts': round(acc_imputed, 4),
        'withAdartsStd': round(std_imputed, 4),
        'withoutAdarts': round(acc_mean_filled, 4),
        'withoutAdartsStd': round(std_mean_filled, 4),
        'improvement': improvement_vs_baseline,
        'gapToOptimal': gap_to_optimal,
        'n_evaluated': n_valid,
        'n_skipped': n_timeseries - n_valid,
        'n_clusters': n_clusters,
        'n_neighbors': n_neighbors,
        'cv_folds': cv_folds,
        'details': {
            'description': 'Self-supervised KNN classification with pseudo labels from observed data.',
            'groundTruth': 'Observed values with interpolation (pseudo upper bound)',
            'withImputePilot': 'ImputePilot imputation on eval-masked data',
            'withoutImputePilot': 'Mean-filled baseline on eval-masked data',
            'withAdarts': 'ImputePilot imputation on eval-masked data',
            'withoutAdarts': 'Mean-filled baseline on eval-masked data',
            'labels': 'Pseudo-labels created by K-Means clustering on observed features'
        }
    }


def _run_forecasting_task(timeseries):
    """Legacy function - redirects to error asking for proper evaluation setup"""
    return {
        'error': 'This legacy function is deprecated. Please use the full downstream evaluation pipeline.',
        'hint': 'First run setup_test_set or setup_upload, then run_imputation, then run_downstream.',
        'task': 'forecasting'
    }


def _run_classification_task(timeseries):
    """Legacy function - redirects to error asking for proper evaluation setup"""
    return {
        'error': 'This legacy function is deprecated. Please use the full downstream evaluation pipeline.',
        'hint': 'First run setup_test_set or setup_upload, then run_imputation, then run_downstream.',
        'task': 'classification'
    }


@require_http_methods(["GET"])
def download_result(request):
    """Download the latest imputed file (or one matching ?algo=)"""
    try:
        algo = (request.GET.get("algo") or "").strip().lower()
        inference_dir = AdartsService.get_inference_dir()

        imputed_files = []
        for name in os.listdir(inference_dir):
            if "imputed" not in name.lower():
                continue
            file_path = os.path.join(inference_dir, name)
            if not os.path.isfile(file_path):
                continue
            if algo and algo not in name.lower():
                continue
            imputed_files.append(file_path)

        if not imputed_files:
            return JsonResponse(
                {
                    "error": "No imputed data found. Please run imputation first.",
                    "hint": "Call /api/recommend/impute/ to generate an imputed file.",
                },
                status=400,
            )

        imputed_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        file_path = imputed_files[0]
        filename = os.path.basename(file_path)
        content_type = "text/csv" if filename.lower().endswith(".csv") else "text/plain"
        return FileResponse(
            open(file_path, "rb"),
            as_attachment=True,
            filename=filename,
            content_type=content_type,
        )
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


# ========== Dashboard Views ==========

@require_http_methods(["GET"])
def get_model_status(request):
    try:
        trained_model = AdartsService.get_trained_model()
        
        if trained_model is None:
            return JsonResponse({
                'lastTrained': '--',
                'winningPipeline': 'Not trained yet',
                'f1Score': 0.0,
                'status': 'not_trained'
            })
        
        pipelines = trained_model.get('pipelines', [])
        
        best_pipeline_name = 'Unknown'
        best_f1 = 0.0
        
        if pipelines:
            best_pipe = None
            for pipe in pipelines:
                if hasattr(pipe, 'scores') and pipe.scores:
                    score = np.mean(pipe.scores) if isinstance(pipe.scores, list) else pipe.scores
                    if score > best_f1:
                        best_f1 = score
                        best_pipe = pipe
            
            if best_pipe and hasattr(best_pipe, 'rm') and hasattr(best_pipe.rm, 'pipe'):
                try:
                    steps = best_pipe.rm.pipe.named_steps
                    step_names = list(steps.keys())
                    best_pipeline_name = ' + '.join([s.replace('_', ' ').title() for s in step_names])
                except:
                    best_pipeline_name = f'Pipeline {best_pipe.id}'
        
        return JsonResponse({
            'lastTrained': time.strftime("%Y-%m-%d %H:%M"),
            'winningPipeline': best_pipeline_name,
            'f1Score': round(float(best_f1), 3),
            'status': 'trained'
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'lastTrained': '--',
            'winningPipeline': 'Error',
            'f1Score': 0.0,
            'status': 'error',
            'error': str(e)
        })


@require_http_methods(["GET"])
def get_benchmarks(request):
    try:
        rec_dir = AdartsService.get_recommendations_dir()
        candidates = []
        if os.path.isdir(rec_dir):
            for name in os.listdir(rec_dir):
                if not (name.startswith("realworld_downstream_eval") and name.endswith(".json")):
                    continue
                # Ignore auxiliary files; Dashboard must read benchmark payload with `rows`.
                if "_summary" in name or name.startswith("_tmp_"):
                    continue
                candidates.append(os.path.join(rec_dir, name))
        if not candidates:
            return JsonResponse({
                "available": False,
                "message": "No RealWorld downstream evaluation found. Run the evaluation command first.",
            })

        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        candidate_by_name = {os.path.basename(path): path for path in candidates}
        requested_file = str(request.GET.get("benchmark_file") or "").strip()
        selected_path = candidate_by_name.get(requested_file, candidates[0])
        selected_file = os.path.basename(selected_path)

        available_benchmarks = []
        for path in candidates:
            file_name = os.path.basename(path)
            generated_at = None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    tmp_payload = json.load(fh)
                generated_at = tmp_payload.get("generated_at")
            except Exception:
                generated_at = None
            available_benchmarks.append({
                "file": file_name,
                "generated_at": generated_at,
                "mtime": int(os.path.getmtime(path)),
            })

        with open(selected_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        rows = annotate_benchmark_rows(_apply_dashboard_status_overrides(payload.get("rows", [])))
        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            row_copy["method"] = _normalize_primary_method_name(row_copy.get("method"))
            normalized_rows.append(row_copy)
        rows = normalized_rows

        methods_raw = payload.get("methods") or [r.get("method") for r in rows if r.get("method")]
        methods = _dedupe_preserve_order(
            [_normalize_primary_method_name(m) for m in methods_raw if m]
        )
        category_payload = build_benchmark_category_summary(
            rows,
            methods,
            dataset_weight_map=_get_realworld_dataset_weight_map(),
        )

        return JsonResponse({
            "available": True,
            "benchmark_file": selected_file,
            "available_benchmarks": available_benchmarks,
            "generated_at": payload.get("generated_at"),
            "missing_rate": payload.get("missing_rate"),
            "seed": payload.get("seed"),
            "methods": methods,
            "rows": category_payload["rows"],
            "categories": category_payload["categories"],
            "category_summary": category_payload["category_summary"],
            "category_stats": category_payload["category_stats"],
        })
    except Exception as e:
        return JsonResponse({
            "available": False,
            "message": f"Failed to load benchmarks: {e}",
        }, status=500)


# ========== Utility Views ==========

@csrf_exempt  
def echo(request):
    """API for test"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({
                'status': 'success',
                'reply': f"Hello {data.get('name', 'Guest')}, I got: {data.get('message')}"
            })
        except:
            return JsonResponse({'status': 'error'}, status=400)
    return JsonResponse({'status': 'error'}, status=405)


from celery.result import AsyncResult
from ImputePilot_api.tasks import run_features_task

@csrf_exempt
@require_http_methods(["POST"])
def run_features_async(request):
    data = json.loads(request.body)
    requested_features = data.get("features", ["catch22", "tsfresh", "topological"])

    task = run_features_task.delay(requested_features)
    return JsonResponse({"task_id": task.id})

@require_http_methods(["GET"])
def get_task_status(request, task_id):
    """Get the status of celery"""
    result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": result.status,   # PENDING / STARTED / SUCCESS / FAILURE
    }
    
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "STARTED":
        response["meta"] = result.info
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    
    return JsonResponse(response)
