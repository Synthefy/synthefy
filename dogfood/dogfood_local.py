"""Dogfood test: SynthefyNoriClient LOCAL mode (in-process, no network, no key).

Prerequisites
-------------
    pip install "synthefy[local]==4.0.0"      # Python >=3.9; pulls synthefy-nori (torch)

Run
---
    python dogfood/dogfood_local.py

Pass criteria: exits 0 and prints "LOCAL OK". A non-zero exit / assertion means
local in-context regression is broken. Uses GPU automatically if one is present,
otherwise CPU.
"""

import numpy as np

from synthefy import SynthefyNoriClient

# Known-answer dataset shared by every dogfood script: y = 3*x0 - 2*x1 + 1.
TOL = 1.0  # generous; catches gross breakage without flaking on float/precision drift


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
    X_train, y_train, X_test, expected = make_dataset()

    client = SynthefyNoriClient(mode="local")
    preds = client.predict(X_train, y_train, X_test)

    errs = check(preds, expected)
    print("LOCAL OK")
    print("  preds   :", [round(p, 3) for p in preds])
    print("  expected:", [round(e, 3) for e in expected])
    print("  max err :", round(max(errs), 3), f"(tol {TOL})")


if __name__ == "__main__":
    main()
