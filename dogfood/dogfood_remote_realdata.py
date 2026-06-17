"""Non-trivial dogfood test: SynthefyNoriClient REMOTE mode on a real dataset.

Unlike dogfood_remote.py (a 2-feature known-answer linear toy), this exercises the
hosted Baseten model on a genuinely hard, real regression problem and asks: is it
*competitive*, not just "does it return numbers".

Dataset: California Housing (sklearn) — 20,640 rows, 8 numeric features
(income, house age, rooms, bedrooms, population, occupancy, lat, long), target =
median house value in $100k units. Strongly nonlinear (geography + interactions).
Falls back to the bundled diabetes dataset if the download is blocked.

Method: in-context regression as a drop-in regressor. For each context size we
sample N labeled context rows, standardize features leak-free (scaler fit on the
context only), predict a FIXED held-out query set in one forward pass, and score
R2 / MAE / RMSE. We compare against sklearn baselines trained on the *same*
standardized context: mean (Dummy), LinearRegression, HistGradientBoosting.

Prereqs
-------
    pip install "synthefy==4.0.1" scikit-learn
    export BASETEN_API_KEY="<your baseten key>"

Run
---
    python dogfood/dogfood_remote_realdata.py
"""

import os
import sys
import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from synthefy import SynthefyNoriClient

SEED = 0
N_TEST = 256
CONTEXT_SIZES = [128, 512, 1024]
TIMEOUT = 1800.0  # generous: large contexts can take many minutes per forward pass


def load_data():
    """Real, non-trivial regression data; diabetes fallback if the fetch is blocked."""
    try:
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing()
        return "California Housing", d.data, d.target
    except Exception as exc:  # offline / mirror down
        from sklearn.datasets import load_diabetes

        print(f"(california fetch failed: {type(exc).__name__}: {exc}; using diabetes)")
        d = load_diabetes()
        return "Diabetes", d.data, d.target


def score(y_true, y_pred):
    rmse = float(mean_squared_error(y_true, y_pred)) ** 0.5
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), rmse


def row(n_ctx, name, r2, mae, rmse, sec=""):
    sec = f"{sec:6.1f}" if isinstance(sec, float) else f"{sec:>6}"
    print(f"{n_ctx:>6} | {name:<26} | {r2:7.3f} | {mae:7.3f} | {rmse:7.3f} | {sec}")


def main():
    if not os.getenv("BASETEN_API_KEY"):
        raise SystemExit("BASETEN_API_KEY is not set. Export your Baseten key and re-run.")

    name, X, y = load_data()
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED
    )
    requested = [int(a) for a in sys.argv[1:] if a.lstrip("-").isdigit()] or CONTEXT_SIZES
    sizes = [n for n in requested if n <= len(X_pool)]
    dropped = [n for n in requested if n > len(X_pool)]
    print(
        f"dataset: {name}  features={X.shape[1]}  pool={len(X_pool)}  test={len(X_test)}"
    )
    if dropped:
        print(f"(skipping {dropped}: exceed available context pool of {len(X_pool)})")

    # max_retries=0: a multi-minute call shouldn't silently 3x its wall-clock on a timeout.
    client = SynthefyNoriClient(timeout=TIMEOUT, max_retries=0)  # gateway, BASETEN_API_KEY
    print(f"endpoint: {client.base_url + client.endpoint} | model: {client.model}")
    print(f"\n{'n_ctx':>6} | {'model':<26} | {'R2':>7} | {'MAE':>7} | {'RMSE':>7} | {'sec':>6}")
    print("-" * 72)

    rng = np.random.default_rng(SEED)
    for n_ctx in sizes:
        idx = rng.choice(len(X_pool), size=n_ctx, replace=False)
        Xc, yc = X_pool[idx], y_pool[idx]
        scaler = StandardScaler().fit(Xc)  # leak-free: fit on context only
        Xc_s, Xt_s = scaler.transform(Xc), scaler.transform(X_test)

        t0 = time.time()
        try:
            preds = client.predict(Xc_s, yc, Xt_s)
        except Exception as exc:
            print(f"{n_ctx:>6} | synthefy-nori FAILED: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        assert len(preds) == len(y_test), f"got {len(preds)} preds for {len(y_test)} rows"
        r2, mae, rmse = score(y_test, preds)
        row(n_ctx, "synthefy-nori (remote)", r2, mae, rmse, dt)

        for bname, est in [
            ("mean (DummyRegressor)", DummyRegressor()),
            ("LinearRegression", LinearRegression()),
            ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=SEED)),
        ]:
            est.fit(Xc_s, yc)
            r2, mae, rmse = score(y_test, est.predict(Xt_s))
            row(n_ctx, "  " + bname, r2, mae, rmse)
        print("-" * 72)

    print("\nHigher R2 is better (1.0 = perfect, 0.0 = no better than the mean).")
    print("MAE/RMSE in target units ($100k for California Housing).")


if __name__ == "__main__":
    main()
