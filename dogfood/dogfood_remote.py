"""Dogfood test: SynthefyNoriClient REMOTE mode (hosted Baseten endpoint).

Prerequisites
-------------
    pip install "synthefy==4.1.0"             # lightweight; no torch
    export BASETEN_API_KEY="<your baseten key>"   # sent as `Authorization: Bearer <key>`

Run
---
    python dogfood/dogfood_remote.py

Hits the Baseten inference gateway (https://inference.baseten.co/predict), which
routes by the model slug in the body. This is the only supported remote path for a
Frontier/gateway key.

Pass criteria: exits 0 and prints "REMOTE OK". 401/403 -> bad/missing key;
400 -> bad request; 5xx -> server issue (the client retries first).
"""

import os

import numpy as np

from synthefy import SynthefyNoriClient

TOL = 1.0


def make_dataset():
    def truth(X):
        return 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0

    rng = np.random.default_rng(42)
    X_train = rng.uniform(-1, 1, size=(50, 2))
    y_train = truth(X_train)
    X_test = np.array([[0.5, 0.5], [-1.0, 1.0], [1.0, -1.0]])
    return X_train, y_train, X_test, [float(v) for v in truth(X_test)]


def check(preds, expected, tol=TOL):
    assert isinstance(preds, list), f"expected list, got {type(preds).__name__}"
    assert len(preds) == len(expected), f"expected {len(expected)} preds, got {len(preds)}"
    assert all(isinstance(p, float) for p in preds), "predictions must be floats"
    errs = [abs(p - e) for p, e in zip(preds, expected)]
    assert max(errs) < tol, (
        f"prediction off by {max(errs):.3f} (tol {tol}): {preds} vs {expected}"
    )
    return errs


def main():
    if not os.getenv("BASETEN_API_KEY"):
        raise SystemExit(
            "BASETEN_API_KEY is not set. Export your Baseten key and re-run."
        )

    X_train, y_train, X_test, expected = make_dataset()

    client = SynthefyNoriClient()  # gateway default (model=synthefy/nori)
    print("url:", client.base_url + client.endpoint, "| mode:", client.mode)
    preds = client.predict(X_train, y_train, X_test)

    errs = check(preds, expected)
    print("REMOTE OK")
    print("  preds   :", [round(p, 3) for p in preds])
    print("  expected:", [round(e, 3) for e in expected])
    print("  max err :", round(max(errs), 3), f"(tol {TOL})")


if __name__ == "__main__":
    main()
