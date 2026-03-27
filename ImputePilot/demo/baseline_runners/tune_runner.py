#!/usr/bin/env python3
import json
import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from common import fail_payload, load_inputs, parse_args, state_file, to_jsonable, write_output


MODEL_PATH = state_file("tune_model.joblib")
META_PATH = state_file("tune_meta.json")


def _ensure_2d(array):
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _get_space(classifier_type):
    if classifier_type == "RandomForest":
        return {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    if classifier_type == "KNN":
        return {
            "n_neighbors": [1, 3, 5, 10, 15, 25, 50],
            "weights": ["uniform", "distance"],
        }
    if classifier_type == "MLP":
        return {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (200,)],
            "activation": ["relu", "tanh"],
            "alpha": ("loguniform", 1e-5, 1e-2),
            "learning_rate": ["constant", "adaptive"],
        }
    if classifier_type == "DecisionTree":
        return {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        }
    return {
        "n_estimators": [10, 50, 100],
        "max_depth": [5, 10, 15],
    }


def _sample_config(space, rng):
    config = {}
    for key, spec in space.items():
        if isinstance(spec, tuple) and len(spec) == 3 and spec[0] == "loguniform":
            low, high = float(spec[1]), float(spec[2])
            config[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            choices = list(spec)
            config[key] = choices[int(rng.integers(0, len(choices)))]
    return config


def _create_classifier(classifier_type, config):
    if classifier_type == "RandomForest":
        return RandomForestClassifier(**config, random_state=42, n_jobs=-1)
    if classifier_type == "KNN":
        return KNeighborsClassifier(**config)
    if classifier_type == "MLP":
        return MLPClassifier(**config, max_iter=500, random_state=42)
    if classifier_type == "DecisionTree":
        return DecisionTreeClassifier(**config, random_state=42)
    return RandomForestClassifier(**config, random_state=42, n_jobs=-1)


def _train_with_ray(classifier_type, X_train, y_train, num_samples, time_budget):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    search_space = _get_space(classifier_type)

    def _to_tune_space(space):
        tune_space = {}
        for key, spec in space.items():
            if isinstance(spec, tuple) and len(spec) == 3 and spec[0] == "loguniform":
                tune_space[key] = tune.loguniform(float(spec[1]), float(spec[2]))
            else:
                tune_space[key] = tune.choice(list(spec))
        return tune_space

    tune_space = _to_tune_space(search_space)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(1, min(4, os.cpu_count() or 1)), log_to_driver=False)

    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    def train_func(config):
        X_local = ray.get(X_ref)
        y_local = ray.get(y_ref)
        clf = _create_classifier(classifier_type, config)
        try:
            scores = cross_val_score(clf, X_local, y_local, cv=3, scoring="f1_weighted")
            tune.report(f1_score=float(np.mean(scores)))
        except Exception:
            tune.report(f1_score=0.0)

    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_func,
        config=tune_space,
        num_samples=max(1, int(num_samples)),
        scheduler=scheduler,
        resources_per_trial={"cpu": 1},
        verbose=0,
        raise_on_failed_trial=False,
        time_budget_s=max(1, int(time_budget)),
        local_dir=str(state_file("ray_results").parent / "ray_results"),
    )

    best_config = analysis.get_best_config(metric="f1_score", mode="max")
    best_trial = analysis.get_best_trial(metric="f1_score", mode="max")
    best_cv_f1 = best_trial.last_result.get("f1_score", 0.0) if best_trial else 0.0
    num_trials_done = len(analysis.trials)

    ray.shutdown()
    return best_config, float(best_cv_f1), int(num_trials_done)


def _train_with_manual_search(classifier_type, X_train, y_train, num_samples, time_budget):
    rng = np.random.default_rng(42)
    search_space = _get_space(classifier_type)
    max_trials = max(1, int(num_samples))
    deadline = time.time() + max(1, int(time_budget))

    best_config = None
    best_score = -1.0
    trials_done = 0

    for _ in range(max_trials):
        if time.time() > deadline:
            break
        config = _sample_config(search_space, rng)
        clf = _create_classifier(classifier_type, config)
        try:
            scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1_weighted")
            score = float(np.mean(scores))
        except Exception:
            score = 0.0
        trials_done += 1
        if score > best_score:
            best_score = score
            best_config = config

    if best_config is None:
        best_config = _sample_config(search_space, rng)
        best_score = 0.0

    return best_config, float(best_score), int(trials_done)


def train(arrays, meta):
    if "X_train" not in arrays or "y_train" not in arrays:
        return fail_payload("Missing required arrays: X_train and y_train are required.")

    X_train = _ensure_2d(arrays["X_train"])
    y_train = np.asarray(arrays["y_train"]).reshape(-1)
    X_test = _ensure_2d(arrays["X_test"]) if "X_test" in arrays else None
    y_test = np.asarray(arrays["y_test"]).reshape(-1) if "y_test" in arrays else None

    classifier_type = str(meta.get("classifier", "RandomForest"))
    num_samples = int(meta.get("num_samples", 50))
    time_budget = int(meta.get("time_budget", 300))

    start = time.time()

    best_config = None
    best_cv_f1 = 0.0
    num_trials_done = 0
    used_ray = False
    try:
        best_config, best_cv_f1, num_trials_done = _train_with_ray(
            classifier_type, X_train, y_train, num_samples, time_budget
        )
        used_ray = True
    except Exception:
        best_config, best_cv_f1, num_trials_done = _train_with_manual_search(
            classifier_type, X_train, y_train, num_samples, time_budget
        )

    final_clf = _create_classifier(classifier_type, best_config)
    final_clf.fit(X_train, y_train)

    if X_test is None or y_test is None or len(y_test) == 0:
        y_eval_true = y_train
        y_eval_pred = final_clf.predict(X_train)
    else:
        y_eval_true = y_test
        y_eval_pred = final_clf.predict(X_test)

    test_f1 = float(f1_score(y_eval_true, y_eval_pred, average="weighted"))
    test_acc = float(accuracy_score(y_eval_true, y_eval_pred))
    training_time = time.time() - start

    joblib.dump({"model": final_clf}, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classifier": classifier_type,
                "best_config": to_jsonable(best_config),
                "cv_f1_score": float(best_cv_f1),
                "num_trials": int(num_trials_done),
                "training_time": float(training_time),
                "used_ray": used_ray,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
        )

    return {
        "status": "success",
        "classifier": classifier_type,
        "best_config": best_config,
        "f1_score": test_f1,
        "accuracy": test_acc,
        "cv_f1_score": float(best_cv_f1),
        "num_trials": int(num_trials_done),
        "training_time": float(training_time),
    }


def predict(arrays, meta):
    del meta
    if "X_infer" not in arrays:
        return fail_payload("Missing required array: X_infer")
    if not MODEL_PATH.exists():
        return fail_payload(f"Model file not found: {MODEL_PATH}")

    model_data = joblib.load(MODEL_PATH)
    clf = model_data["model"]
    X_infer = _ensure_2d(arrays["X_infer"])

    start = time.time()
    preds = clf.predict(X_infer)
    inference_time_ms = (time.time() - start) * 1000.0

    pred_counts = Counter(preds)
    most_common_algo, most_common_count = pred_counts.most_common(1)[0]

    confidence = float(most_common_count / max(1, len(preds)))
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_infer)
            confidence = float(np.mean(np.max(proba, axis=1)))
        except Exception:
            pass

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
        result = fail_payload("Unhandled exception in Tune runner.", e)
    write_output(args.output, result)


if __name__ == "__main__":
    main()

