#!/usr/bin/env python3
import argparse
import json
import traceback
from pathlib import Path

import numpy as np


RUNNER_DIR = Path(__file__).resolve().parent
STATE_DIR = RUNNER_DIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
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


def state_file(filename):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / filename

