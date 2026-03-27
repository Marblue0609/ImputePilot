#!/usr/bin/env python3
import json
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from common import fail_payload, load_inputs, parse_args, state_file, write_output


MODEL_PATH = state_file("autofolio_model.joblib")
META_PATH = state_file("autofolio_meta.json")


def _ensure_2d(array):
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _save_meta(data):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _build_candidates(seed, n_classes):
    class_weight = "balanced" if n_classes > 1 else None
    return [
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=seed, class_weight=class_weight)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=300, random_state=seed, class_weight=class_weight)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=seed)),
        ("logistic_regression", LogisticRegression(max_iter=2000, multi_class="auto")),
    ]


def train(arrays, meta):
    if "X_train" not in arrays or "y_train" not in arrays:
        return fail_payload("Missing required arrays: X_train and y_train are required.")

    X_train = _ensure_2d(arrays["X_train"])
    y_train = np.asarray(arrays["y_train"]).reshape(-1)

    n_classes = len(np.unique(y_train))
    if n_classes <= 1:
        return fail_payload("AutoFolio training aborted: only one class present in y_train.")

    seed = int(meta.get("seed", 23))
    scoring = "f1" if n_classes == 2 else "f1_macro"

    candidates = _build_candidates(seed, n_classes)
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in candidates:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
            score = float(np.mean(scores))
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_model is None:
        return fail_payload("AutoFolio training failed: no candidate model could be evaluated.")

    start = time.time()
    best_model.fit(X_train, y_train)
    training_time = time.time() - start

    model_data = {
        "model": best_model,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_estimator": best_name,
    }
    joblib.dump(model_data, MODEL_PATH)
    _save_meta(
        {
            "best_estimator": best_name,
            "f1_score": float(best_score if np.isfinite(best_score) else 0.0),
            "training_time": float(training_time),
            "trained_at": model_data["trained_at"],
        }
    )

    return {
        "status": "success",
        "training_time": float(training_time),
        "best_estimator": best_name,
        "f1_score": float(best_score if np.isfinite(best_score) else 0.0),
    }


def predict(arrays, meta):
    del meta
    if not MODEL_PATH.exists():
        return fail_payload(f"Model file not found: {MODEL_PATH}")
    if "X_infer" not in arrays:
        return fail_payload("Missing required array: X_infer")

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    X_infer = _ensure_2d(arrays["X_infer"])

    start = time.time()
    preds = model.predict(X_infer)
    inference_time_ms = (time.time() - start) * 1000.0

    pred_counts = Counter(preds)
    most_common_algo, most_common_count = pred_counts.most_common(1)[0]

    confidence = float(most_common_count / max(1, len(preds)))
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_infer)
            if proba is not None and len(proba) > 0:
                confidence = float(np.mean(np.max(proba, axis=1)))
        except Exception:
            pass

    return {
        "status": "success",
        "algo": str(most_common_algo),
        "confidence": float(confidence),
        "inference_time_ms": float(inference_time_ms),
        "best_estimator": model_data.get("best_estimator", ""),
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
        result = fail_payload("Unhandled exception in AutoFolio runner.", e)
    write_output(args.output, result)


if __name__ == "__main__":
    main()
