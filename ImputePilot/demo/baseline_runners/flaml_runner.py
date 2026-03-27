#!/usr/bin/env python3
import json
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

from common import fail_payload, load_inputs, parse_args, state_file, write_output


MODEL_PATH = state_file("flaml_model.joblib")
META_PATH = state_file("flaml_meta.json")


def _ensure_2d(array):
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _save_meta(data):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def train(arrays, meta):
    try:
        from flaml import AutoML
    except Exception as e:
        return fail_payload("FLAML is not available in FLAML runner environment.", e)

    if "X_train" not in arrays or "y_train" not in arrays:
        return fail_payload("Missing required arrays: X_train and y_train are required.")

    X_train = _ensure_2d(arrays["X_train"])
    y_train = np.asarray(arrays["y_train"]).reshape(-1)

    automl = AutoML()
    available_estimators = []

    try:
        import lightgbm  # noqa: F401
        available_estimators.append("lgbm")
    except Exception:
        pass

    try:
        import xgboost  # noqa: F401
        available_estimators.append("xgboost")
    except Exception:
        pass

    available_estimators.extend(["rf", "extra_tree"])

    n_classes = len(np.unique(y_train))
    if n_classes <= 1:
        return fail_payload(
            "FLAML training aborted: only one class present in y_train."
        )

    metric = "f1" if n_classes == 2 else "macro_f1"
    fit_kwargs = {
        "X_train": X_train,
        "y_train": y_train,
        "task": "classification",
        "metric": metric,
        "estimator_list": available_estimators,
        "verbose": 0,
    }
    time_budget = meta.get("time_budget", None)
    if time_budget is not None:
        try:
            tb = int(time_budget)
            if tb > 0:
                fit_kwargs["time_budget"] = tb
            else:
                fit_kwargs["time_budget"] = 3600
                fit_kwargs["early_stop"] = True
        except Exception:
            fit_kwargs["time_budget"] = 3600
            fit_kwargs["early_stop"] = True
    else:
        fit_kwargs["time_budget"] = 3600
        fit_kwargs["early_stop"] = True

    start = time.time()
    automl.fit(**fit_kwargs)
    training_time = time.time() - start

    try:
        cv_scores = cross_val_score(
            automl.model,
            X_train,
            y_train,
            cv=3,
            scoring="f1_weighted",
        )
        f1_score_val = float(np.mean(cv_scores))
    except Exception:
        f1_score_val = 0.0

    model_data = {
        "model": automl,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(model_data, MODEL_PATH)
    _save_meta(
        {
            "best_estimator": str(automl.best_estimator),
            "best_config": automl.best_config,
            "f1_score": float(f1_score_val),
            "training_time": float(training_time),
            "trained_at": model_data["trained_at"],
        }
    )

    return {
        "status": "success",
        "training_time": float(training_time),
        "best_estimator": str(automl.best_estimator),
        "best_config": automl.best_config,
        "f1_score": float(f1_score_val),
    }


def predict(arrays, meta):
    del meta
    if not MODEL_PATH.exists():
        return fail_payload(f"Model file not found: {MODEL_PATH}")
    if "X_infer" not in arrays:
        return fail_payload("Missing required array: X_infer")

    model_data = joblib.load(MODEL_PATH)
    automl = model_data["model"]
    X_infer = _ensure_2d(arrays["X_infer"])

    start = time.time()
    preds = automl.predict(X_infer)
    inference_time_ms = (time.time() - start) * 1000.0

    pred_counts = Counter(preds)
    most_common_algo, most_common_count = pred_counts.most_common(1)[0]

    confidence = 0.0
    try:
        proba = automl.predict_proba(X_infer)
        if proba is not None and len(proba) > 0:
            confidence = float(np.mean(np.max(proba, axis=1)))
        else:
            confidence = float(most_common_count / max(1, len(preds)))
    except Exception:
        confidence = float(most_common_count / max(1, len(preds)))

    return {
        "status": "success",
        "algo": str(most_common_algo),
        "confidence": float(confidence),
        "inference_time_ms": float(inference_time_ms),
    }


def main():
    args = parse_args()
    try:
        arrays, meta = load_inputs(args.input, args.meta)
        if args.mode == "train":
            result = train(arrays, meta)
        else:
            result = predict(arrays, meta)
    except Exception as e:
        result = fail_payload("Unhandled exception in FLAML runner.", e)
    write_output(args.output, result)


if __name__ == "__main__":
    main()
