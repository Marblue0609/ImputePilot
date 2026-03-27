# ImputePilot/demo/ImputePilot_api/tasks.py
import traceback

import numpy as np
import pandas as pd
from celery import shared_task
from django.conf import settings

from ImputePilot_api.views import AdartsService  # Reuse the existing service layer.


def _build_eval_mask(values, seed, missing_rate=0.2):
    rng = np.random.default_rng(seed)
    eval_mask = rng.random(values.shape) < missing_rate
    if not np.any(eval_mask):
        eval_mask.reshape(-1)[0] = True
    return eval_mask


@shared_task(bind=True)
def run_external_dl_eval_task(self, dataset_name, cluster_id, external_algo, mask_seed, max_rows=64, missing_rate=0.2):
    """
    Evaluate one external DL algorithm on a single cluster.
    Returns a payload similar to the external runner output, plus dataset/cluster info.
    """
    try:
        from ImputePilot_api.views import _run_external_labeling_runner

        clusterer = AdartsService.get_clusterer()
        datasets = AdartsService.load_datasets()
        dataset = next((d for d in datasets if d.name == dataset_name), None)
        if dataset is None:
            return {
                "status": "failed",
                "algorithm": external_algo,
                "dataset": dataset_name,
                "clusterId": int(cluster_id),
                "error": f"Dataset not found: {dataset_name}",
            }

        cassignment = dataset.load_cassignment(clusterer)
        full_timeseries = dataset.load_timeseries(transpose=True)
        cluster_ts = dataset.get_cluster_by_id(full_timeseries, int(cluster_id), cassignment)
        cluster_ts = cluster_ts.apply(pd.to_numeric, errors="coerce")
        values = cluster_ts.to_numpy(dtype="float32")

        if values.size == 0:
            return {
                "status": "skipped",
                "algorithm": external_algo,
                "dataset": dataset_name,
                "clusterId": int(cluster_id),
                "message": "Cluster is empty.",
            }

        values = values[: min(max_rows, values.shape[0])]
        if values.shape[0] == 0 or values.shape[1] == 0:
            return {
                "status": "skipped",
                "algorithm": external_algo,
                "dataset": dataset_name,
                "clusterId": int(cluster_id),
                "message": "Cluster has invalid shape for DL evaluation.",
            }

        eval_mask = _build_eval_mask(values, mask_seed, missing_rate=missing_rate)
        values_with_missing = values.copy()
        values_with_missing[eval_mask] = np.nan

        algo_key = str(external_algo).strip().lower()
        meta_dict = {
            "algorithm": algo_key,
            "device": getattr(settings, "DL_LABEL_DEVICE", "cpu"),
        }
        if algo_key == "deepmvi":
            meta_dict.update({"epochs": 20, "patience": 5, "tr_ratio": 0.7})
        elif algo_key == "mrnn":
            meta_dict.update({"iterations": 200, "hidden_dim": 10, "seq_length": 7, "tr_ratio": 0.7})
        elif algo_key == "mpin":
            meta_dict.update({"epochs": 50, "num_of_iteration": 3, "window": 2, "k": 10, "tr_ratio": 0.7})
        elif algo_key == "iim":
            meta_dict.update({"neighbors": 10, "adaptive": False})

        result = _run_external_labeling_runner(
            mode="benchmark",
            arrays_dict={
                "X_input": values_with_missing,
                "X_true": values,
                "eval_mask": eval_mask,
            },
            meta_dict=meta_dict,
            timeout_sec=int(getattr(settings, "DL_LABEL_TIMEOUT_SEC", 1800)),
        )
        result["algorithm"] = algo_key
        result["dataset"] = dataset_name
        result["clusterId"] = int(cluster_id)
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "failed",
            "algorithm": str(external_algo).strip().lower(),
            "dataset": dataset_name,
            "clusterId": int(cluster_id),
            "error": str(e),
        }

@shared_task(bind=True)
def run_features_task(self, requested_features=None):
    try:
        allowed_features = {"catch22", "tsfresh", "topological"}
        if requested_features is None:
            requested_features = ["catch22", "tsfresh", "topological"]
        else:
            requested_features = [
                str(feat).strip().lower()
                for feat in requested_features
                if str(feat).strip().lower() in allowed_features
            ]
            if not requested_features:
                requested_features = ["catch22", "tsfresh", "topological"]

        self.update_state(state="STARTED", meta={"step": "loading_datasets"})

        datasets = AdartsService.load_datasets()
        if not datasets:
            return {"error": "No datasets found.", "featureImportance": []}

        extracted_summary = {}

        # Serial operation
        for feat_name in requested_features:
            self.update_state(state="STARTED", meta={"step": f"loading_extractor:{feat_name}"})
            extractor_cls = AdartsService.get_feature_extractor(feat_name)
            if not extractor_cls:
                extracted_summary[feat_name] = 0
                continue

            extractor = extractor_cls.get_instance()

            for i, dataset in enumerate(datasets):
                self.update_state(
                    state="STARTED",
                    meta={"step": f"extracting:{feat_name}", "dataset_index": i, "dataset_total": len(datasets)}
                )

                try:
                    extractor.extract(dataset)
                    df = extractor.load_features(dataset)
                    feat_count = max(len(df.columns) - 1, 0)  # Exclude the label column.
                    extracted_summary[feat_name] = extracted_summary.get(feat_name, 0) + feat_count
                except Exception:
                    traceback.print_exc()
                    # Do not fail the whole task when one dataset fails.
                    extracted_summary[feat_name] = extracted_summary.get(feat_name, 0)

        response_data = [{"name": k, "value": v} for k, v in extracted_summary.items()]
        return {"featureImportance": response_data}

    except Exception as e:
        traceback.print_exc()
        # Re-raise so Celery marks the task as FAILURE.
        raise e
