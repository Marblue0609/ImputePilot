#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["impute", "benchmark"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_inputs(input_npz_path, meta_json_path):
    arrays = {}
    with np.load(input_npz_path, allow_pickle=True) as data:
        for key in data.files:
            arrays[key] = data[key]

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return arrays, meta


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_output(path, payload):
    payload = to_jsonable(payload)
    payload.setdefault("status", "success")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def fail_payload(error, exc=None):
    payload = {"status": "failed", "error": str(error)}
    if exc is not None:
        payload["exception"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc(limit=20)[-4000:]
    return payload


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _ensure_3d(array):
    arr = np.asarray(array, dtype="float32")
    if arr.ndim == 1:
        return arr.reshape(1, arr.shape[0], 1)
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], arr.shape[1], 1)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported input shape: {arr.shape}")


def _get_input_array(arrays):
    for key in ("X_input", "X", "input"):
        if key in arrays:
            return arrays[key]
    raise ValueError("Missing required input array. Expected one of: X_input, X, input")


def _align_mask(mask, target_shape):
    arr = np.asarray(mask)
    if arr.shape == target_shape:
        return arr.astype(bool)

    if arr.ndim == 2 and len(target_shape) == 3 and target_shape[2] == 1:
        if arr.shape == target_shape[:2]:
            return arr.reshape(target_shape).astype(bool)

    return None


def _build_eval_mask(arrays, x_input_3d):
    target_shape = x_input_3d.shape

    if "eval_mask" in arrays:
        mask = _align_mask(arrays["eval_mask"], target_shape)
        if mask is not None:
            return mask

    if "missing_mask" in arrays:
        mask = _align_mask(arrays["missing_mask"], target_shape)
        if mask is not None:
            return mask

    return np.isnan(x_input_3d)


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_str(value, default):
    if value is None:
        return default
    return str(value)


def _prepare_2d(array_3d):
    x_input = np.asarray(array_3d, dtype="float32")
    transposed = False
    if x_input.ndim == 3:
        if x_input.shape[-1] == 1:
            x_input = x_input[..., 0]
        else:
            x_input = x_input.reshape(x_input.shape[0], -1)
    if x_input.ndim != 2:
        raise ValueError(f"Unsupported input shape: {x_input.shape}")
    if x_input.shape[0] < x_input.shape[1]:
        x_input = x_input.T
        transposed = True
    return x_input, transposed


def _restore_from_2d(x_imputed_2d, original_shape, transposed):
    x_imputed = np.asarray(x_imputed_2d, dtype="float32")
    if transposed:
        x_imputed = x_imputed.T
    if len(original_shape) == 3:
        if original_shape[-1] == 1:
            x_imputed = x_imputed.reshape(original_shape)
        else:
            x_imputed = x_imputed.reshape(
                original_shape[0],
                original_shape[1],
                original_shape[2],
            )
    return x_imputed


def _run_brits(x_input_3d, meta):
    try:
        from pypots.imputation.brits import BRITS
    except Exception as e:
        raise RuntimeError("pypots.BRITS is unavailable in this environment") from e

    if np.isnan(x_input_3d).all():
        raise ValueError("Input is fully NaN; BRITS cannot run on empty observations")

    n_samples, n_steps, n_features = x_input_3d.shape
    hidden_size = _safe_int(meta.get("rnn_hidden_size", 32), 32)
    batch_size = _safe_int(meta.get("batch_size", min(16, max(1, n_samples))), 8)
    epochs = _safe_int(meta.get("epochs", 3), 3)
    patience = _safe_int(meta.get("patience", 2), 2)
    num_workers = _safe_int(meta.get("num_workers", 0), 0)
    verbose = _parse_bool(meta.get("verbose", False), default=False)
    device = _safe_str(meta.get("device", "cpu"), "cpu").strip().lower()
    if device in {"auto", "cuda_if_available", "gpu"}:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    elif device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    model = BRITS(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=max(1, hidden_size),
        batch_size=max(1, batch_size),
        epochs=max(1, epochs),
        patience=max(1, patience),
        num_workers=max(0, num_workers),
        device=device,
        verbose=verbose,
    )

    model.fit({"X": x_input_3d})
    x_imputed = model.impute({"X": x_input_3d})
    return np.asarray(x_imputed, dtype="float32")


def _init_model(model_cls, kwargs):
    try:
        sig = inspect.signature(model_cls)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        filtered = kwargs
    return model_cls(**filtered)


def _run_deepmvi(x_input_3d, meta):
    device = _safe_str(meta.get("device", "auto"), "auto").strip().lower()
    if device == "cpu":
        # DeepMVI wrapper does not expose a device argument; force torch to see no CUDA devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        from imputegap.wrapper.AlgoPython.DeepMVI.recovery import deep_mvi_recovery
    except Exception as e:
        raise RuntimeError("imputegap.DeepMVI is unavailable in this environment") from e

    if np.isnan(x_input_3d).all():
        raise ValueError("Input is fully NaN; DeepMVI cannot run on empty observations")

    x_input, transposed = _prepare_2d(x_input_3d)

    max_epoch = _safe_int(meta.get("epochs", meta.get("max_epoch", 3)), 3)
    patience = _safe_int(meta.get("patience", 2), 2)
    lr = float(meta.get("lr", meta.get("learning_rate", 1e-3)) or 1e-3)
    tr_ratio = float(meta.get("tr_ratio", 0.9) or 0.9)
    missing_ratio = float(np.isnan(x_input).mean())
    if missing_ratio > 0:
        tr_ratio = min(tr_ratio, max(0.05, 1.0 - missing_ratio))
    seed = _safe_int(meta.get("seed", 42), 42)
    verbose = _parse_bool(meta.get("verbose", False), default=False)

    x_imputed_2d = deep_mvi_recovery(
        x_input,
        max_epoch=max(1, max_epoch),
        patience=max(1, patience),
        lr=lr,
        tr_ratio=tr_ratio,
        seed=seed,
        verbose=verbose,
    )
    return _restore_from_2d(x_imputed_2d, x_input_3d.shape, transposed)


def _run_mrnn(x_input_3d, meta):
    disable_xla = _parse_bool(meta.get("disable_xla", True), default=True)
    if disable_xla:
        # Disable XLA only for MRNN to avoid TF JIT compilation failures.
        os.environ.setdefault("TF_ENABLE_XLA", "0")
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    try:
        if disable_xla:
            try:
                import tensorflow as tf
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass
        from imputegap.wrapper.AlgoPython.MRNN.runnerMRNN import mrnn_recov
    except Exception as e:
        raise RuntimeError("imputegap.MRNN is unavailable in this environment") from e

    if np.isnan(x_input_3d).all():
        raise ValueError("Input is fully NaN; MRNN cannot run on empty observations")

    x_input, transposed = _prepare_2d(x_input_3d)

    hidden_dim = _safe_int(meta.get("hidden_dim", 10), 10)
    learning_rate = float(meta.get("lr", meta.get("learning_rate", 1e-2)) or 1e-2)
    iterations = _safe_int(meta.get("iterations", 200), 200)
    seq_length = _safe_int(meta.get("seq_length", 7), 7)
    tr_ratio = float(meta.get("tr_ratio", 0.7) or 0.7)
    seed = _safe_int(meta.get("seed", 42), 42)
    verbose = _parse_bool(meta.get("verbose", False), default=False)

    x_imputed_2d = mrnn_recov(
        x_input,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        iterations=max(1, iterations),
        seq_length=max(2, seq_length),
        tr_ratio=tr_ratio,
        seed=seed,
        verbose=verbose,
    )
    return _restore_from_2d(x_imputed_2d, x_input_3d.shape, transposed)


def _run_iim(x_input_3d, meta):
    try:
        from imputegap.wrapper.AlgoPython.IIM.runnerIIM import iim_recovery
    except Exception as e:
        raise RuntimeError("imputegap.IIM is unavailable in this environment") from e

    if np.isnan(x_input_3d).all():
        raise ValueError("Input is fully NaN; IIM cannot run on empty observations")

    x_input, transposed = _prepare_2d(x_input_3d)
    neighbors = _safe_int(meta.get("neighbors", meta.get("learning_neighbors", 10)), 10)
    adaptive_flag = _parse_bool(meta.get("adaptive", False), default=False)
    verbose = _parse_bool(meta.get("verbose", False), default=False)

    x_imputed_2d = iim_recovery(
        x_input,
        adaptive_flag=adaptive_flag,
        learning_neighbors=max(1, neighbors),
    )
    if verbose:
        print(f"(IMPUTATION) IIM\n\tMatrix: {x_input.shape[0]}, {x_input.shape[1]}\n\tneighbors: {neighbors}\n\tadaptive: {adaptive_flag}")
    return _restore_from_2d(x_imputed_2d, x_input_3d.shape, transposed)


def _run_mpin(x_input_3d, meta):
    try:
        from imputegap.wrapper.AlgoPython.MPIN.runnerMPIN import recoverMPIN
    except Exception as e:
        raise RuntimeError("imputegap.MPIN is unavailable in this environment") from e

    if np.isnan(x_input_3d).all():
        raise ValueError("Input is fully NaN; MPIN cannot run on empty observations")

    x_input, transposed = _prepare_2d(x_input_3d)

    window = _safe_int(meta.get("window", 2), 2)
    k = _safe_int(meta.get("k", 10), 10)
    lr = float(meta.get("lr", meta.get("learning_rate", 1e-2)) or 1e-2)
    weight_decay = float(meta.get("weight_decay", 0.1) or 0.1)
    epochs = _safe_int(meta.get("epochs", 50), 50)
    num_of_iteration = _safe_int(meta.get("num_of_iteration", 3), 3)
    thre = float(meta.get("thre", 0.25) or 0.25)
    base = _safe_str(meta.get("base", "SAGE"), "SAGE")
    eval_ratio = float(meta.get("eval_ratio", 0.05) or 0.05)
    dynamic = _safe_str(meta.get("dynamic", "true"), "true")
    tr_ratio = float(meta.get("tr_ratio", 0.7) or 0.7)
    seed = _safe_int(meta.get("seed", 42), 42)
    verbose = _parse_bool(meta.get("verbose", False), default=False)

    x_imputed_2d = recoverMPIN(
        x_input,
        mode="alone",
        window=max(1, window),
        k=max(1, k),
        lr=lr,
        weight_decay=weight_decay,
        epochs=max(1, epochs),
        num_of_iteration=max(1, num_of_iteration),
        thre=thre,
        base=base,
        out_channels=64,
        eval_ratio=eval_ratio,
        state=True,
        dynamic=dynamic,
        tr_ratio=tr_ratio,
        seed=seed,
        verbose=verbose,
    )
    return _restore_from_2d(x_imputed_2d, x_input_3d.shape, transposed)


def _mean_fallback(x_input_3d):
    x = np.asarray(x_input_3d, dtype="float32")
    out = x.copy()

    feature_means = np.nanmean(out, axis=(0, 1))
    global_mean = np.nanmean(out)
    if np.isnan(global_mean):
        global_mean = 0.0

    feature_means = np.where(np.isnan(feature_means), global_mean, feature_means)
    nan_idx = np.where(np.isnan(out))
    if len(nan_idx[0]) > 0:
        out[nan_idx] = feature_means[nan_idx[2]]
    return out


def _compute_metrics(x_true, x_imputed, eval_mask):
    diff = x_imputed[eval_mask] - x_true[eval_mask]
    if diff.size == 0:
        return {}

    return {
        "eval_count": int(diff.size),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mae": float(np.mean(np.abs(diff))),
    }


def _write_imputed_csv(x_imputed, output_path):
    arr = np.asarray(x_imputed, dtype="float32")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path), arr, delimiter=",")
    return str(output_path)


def run(mode, arrays, meta):
    start_time = time.time()

    algo = str(meta.get("algorithm", "brits")).strip().lower()
    x_input_raw = _get_input_array(arrays)
    x_input_3d = _ensure_3d(x_input_raw)

    if algo not in {"brits", "deepmvi", "mrnn", "mpin", "iim"}:
        return fail_payload(
            "Unsupported algorithm: "
            f"{algo}. Currently supported: brits, deepmvi, mrnn, mpin, iim"
        )

    nan_before = int(np.isnan(x_input_3d).sum())

    try:
        if algo == "deepmvi":
            x_imputed = _run_deepmvi(x_input_3d, meta)
        elif algo == "mrnn":
            x_imputed = _run_mrnn(x_input_3d, meta)
        elif algo == "mpin":
            x_imputed = _run_mpin(x_input_3d, meta)
        elif algo == "iim":
            x_imputed = _run_iim(x_input_3d, meta)
        else:
            x_imputed = _run_brits(x_input_3d, meta)
        used_fallback = False
    except Exception as e:
        if not _parse_bool(meta.get("allow_fallback", False), default=False):
            raise
        x_imputed = _mean_fallback(x_input_3d)
        used_fallback = True

    nan_after = int(np.isnan(x_imputed).sum())

    payload = {
        "status": "success",
        "algorithm": algo,
        "shape": list(x_imputed.shape),
        "nan_before": nan_before,
        "nan_after": nan_after,
        "used_fallback": used_fallback,
    }

    if "X_true" in arrays:
        try:
            x_true_3d = _ensure_3d(arrays["X_true"])
            if x_true_3d.shape == x_imputed.shape:
                eval_mask = _build_eval_mask(arrays, x_input_3d)
                payload.update(_compute_metrics(x_true_3d, x_imputed, eval_mask))
            else:
                payload["metrics_warning"] = (
                    "X_true shape does not match imputed output; skipped RMSE/MAE"
                )
        except Exception:
            payload["metrics_warning"] = "Failed to compute RMSE/MAE from X_true"

    imputed_output_path = meta.get("imputed_output_path")
    imputed_output_dir = meta.get("imputed_output_dir")
    if imputed_output_path or imputed_output_dir or mode == "impute":
        if not imputed_output_path and imputed_output_dir:
            filename = f"imputed_{algo}_{int(time.time())}.csv"
            imputed_output_path = str(Path(imputed_output_dir) / filename)
        if imputed_output_path:
            try:
                payload["imputed_file"] = _write_imputed_csv(x_imputed, imputed_output_path)
            except Exception as e:
                payload["imputed_error"] = f"Failed to write imputed CSV: {e}"

    if _parse_bool(meta.get("return_imputed", False), default=False):
        if x_input_raw.ndim <= 2 and x_imputed.shape[-1] == 1:
            payload["imputed"] = x_imputed[..., 0]
        else:
            payload["imputed"] = x_imputed

    payload["runtime"] = round(time.time() - start_time, 3)

    return payload


def main():
    args = parse_args()
    try:
        arrays, meta = load_inputs(args.input, args.meta)
        result = run(args.mode, arrays, meta)
    except Exception as e:
        result = fail_payload("Unhandled exception in DL labeling runner.", e)
    write_output(args.output, result)


if __name__ == "__main__":
    main()
