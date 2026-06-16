"""Dogfood test: SynthefyNoriClient LOCAL mode on a Modal T4 GPU.

This builds a Modal image with the published ``synthefy[local]==4.0.0`` and runs
in-process inference on a Tesla T4, confirming CUDA is actually used.

Prerequisites
-------------
- A Modal account: https://modal.com  (running a T4 incurs a small compute charge).
- Auth, either:
    uvx modal setup                                  # browser login, writes ~/.modal.toml
  or export both env vars:
    export MODAL_TOKEN_ID="ak-..."
    export MODAL_TOKEN_SECRET="as-..."

Run
---
    uvx modal run dogfood/modal_t4.py
    # or, if modal is already installed: modal run dogfood/modal_t4.py

Pass criteria: prints a RESULT block with ``cuda_available: True``,
``device: Tesla T4`` and ``ok: True``. The function returns a JSON *string* so the
result deserializes even in a minimal local environment (no numpy/torch needed).
"""

import json

import modal

app = modal.App("synthefy-nori-t4-dogfood")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "synthefy[local]==4.0.0"
)

TOL = 1.0


@app.function(gpu="T4", image=image, timeout=900)
def run_local_predict() -> str:
    import traceback

    import numpy as np
    import torch

    out = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
    }
    print("ENV:", out)

    try:
        from synthefy import SynthefyNoriClient

        def truth(X):
            return 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0

        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1, 1, size=(50, 2))
        y_train = truth(X_train)
        X_test = np.array([[0.5, 0.5], [-1.0, 1.0], [1.0, -1.0]])

        client = SynthefyNoriClient(mode="local")
        preds = [float(p) for p in client.predict(X_train, y_train, X_test)]
        expected = [float(v) for v in truth(X_test)]
        errs = [abs(p - e) for p, e in zip(preds, expected)]

        out["preds"] = preds
        out["expected"] = expected
        out["max_err"] = max(errs)
        out["ok"] = bool(max(errs) < TOL and out["cuda_available"])
        print("preds:", preds, "| expected:", expected, "| max_err:", round(max(errs), 3))
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        print("REMOTE ERROR:\n" + out["traceback"])

    return json.dumps(out)


@app.local_entrypoint()
def main():
    result = json.loads(run_local_predict.remote())
    print("=== RESULT FROM MODAL T4 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if not result.get("ok"):
        raise SystemExit("Modal T4 dogfood FAILED (see error/traceback above)")
    print("MODAL T4 OK")
