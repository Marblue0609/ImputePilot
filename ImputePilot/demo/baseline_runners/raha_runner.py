#!/usr/bin/env python3
import json
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from common import fail_payload, load_inputs, parse_args, state_file, write_output


MODEL_PATH = state_file("raha_model.joblib")
META_PATH = state_file("raha_meta.json")


def _ensure_2d(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _cosine_predict(X, labels, centroids):
    eps = 1e-12
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    C_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + eps)
    sim = X_norm @ C_norm.T
    best_idx = np.argmax(sim, axis=1)
    preds = labels[best_idx]
    conf = (sim[np.arange(sim.shape[0]), best_idx] + 1.0) / 2.0
    conf = np.clip(conf, 0.0, 1.0)
    return preds, conf


def _build_centroids(X, y):
    unique_labels = np.array(sorted(set(y.tolist())))
    centroids = []
    for label in unique_labels:
        mask = y == label
        if np.any(mask):
            centroids.append(np.mean(X[mask], axis=0))
        else:
            centroids.append(np.zeros((X.shape[1],), dtype=np.float64))
    return unique_labels, np.asarray(centroids, dtype=np.float64)


def train(arrays, meta):
    del meta
    if "X_train" not in arrays or "y_train" not in arrays:
        return fail_payload("Missing required arrays: X_train and y_train are required.")

    X = _ensure_2d(arrays["X_train"])
    y = np.asarray(arrays["y_train"]).astype(str).reshape(-1)
    if X.shape[0] != y.shape[0]:
        return fail_payload("Mismatched row count between X_train and y_train.")
    if X.shape[0] < 2:
        return fail_payload("Need at least 2 training samples for RAHA baseline.")

    start = time.time()

    unique_count = len(set(y.tolist()))
    if unique_count >= 2 and X.shape[0] >= 10:
        try:
            X_fit, X_eval, y_fit, y_eval = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        except Exception:
            X_fit, X_eval, y_fit, y_eval = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None,
            )
    else:
        X_fit, y_fit = X, y
        X_eval, y_eval = X, y

    labels, centroids = _build_centroids(X_fit, y_fit)
    y_pred, _ = _cosine_predict(X_eval, labels, centroids)

    f1_val = float(f1_score(y_eval, y_pred, average="weighted", zero_division=0))
    acc_val = float(accuracy_score(y_eval, y_pred))
    precision_val = float(precision_score(y_eval, y_pred, average="weighted", zero_division=0))
    recall_val = float(recall_score(y_eval, y_pred, average="weighted", zero_division=0))

    training_time = time.time() - start

    model_data = {
        "labels": labels,
        "centroids": centroids,
    }
    joblib.dump(model_data, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_clusters": int(len(labels)),
                "test_size": int(len(y_eval)),
                "training_time": float(training_time),
            },
            f,
            ensure_ascii=False,
        )

    return {
        "status": "success",
        "f1_score": f1_val,
        "accuracy": acc_val,
        "precision": precision_val,
        "recall": recall_val,
        "num_clusters": int(len(labels)),
        "test_size": int(len(y_eval)),
        "training_time": float(training_time),
    }


def predict(arrays, meta):
    del meta
    if "X_infer" not in arrays:
        return fail_payload("Missing required array: X_infer")
    if not MODEL_PATH.exists():
        return fail_payload(f"Model file not found: {MODEL_PATH}")

    model_data = joblib.load(MODEL_PATH)
    labels = np.asarray(model_data["labels"]).astype(str)
    centroids = _ensure_2d(model_data["centroids"])

    X_infer = _ensure_2d(arrays["X_infer"])

    start = time.time()
    preds, conf = _cosine_predict(X_infer, labels, centroids)
    inference_time_ms = (time.time() - start) * 1000.0

    pred_counts = Counter(preds.tolist())
    most_common_algo, _ = pred_counts.most_common(1)[0]

    mask = preds == most_common_algo
    if np.any(mask):
        confidence = float(np.mean(conf[mask]))
    else:
        confidence = float(np.mean(conf)) if len(conf) > 0 else 0.0

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
        result = fail_payload("Unhandled exception in RAHA runner.", e)
    write_output(args.output, result)


if __name__ == "__main__":
    main()

