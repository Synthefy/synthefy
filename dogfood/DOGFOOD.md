# Synthefy Nori Client — Dogfood Runbook

A self-contained kit for verifying that `SynthefyNoriClient` (shipped in
`synthefy` **4.0.1**) works across all three execution paths. Clone the repo,
follow the section for each path, and tick the checklist at the bottom.

## What we're testing

`SynthefyNoriClient` is an in-context regressor: you pass labeled context rows
(`X_train`, `y_train`) and query rows (`X_test`), and get one prediction per query
row in a single forward pass. It runs in three modes:

| Path | Mode | Mechanism | Needs |
|------|------|-----------|-------|
| **Local (CPU/GPU)** | `mode="local"` | in-process via `synthefy-nori` | `synthefy[local]`, no key |
| **Remote** | `mode="remote"` (default) | HTTPS to hosted Baseten endpoint | `synthefy`, `BASETEN_API_KEY` |
| **Modal T4** | `mode="local"` on a GPU worker | same as Local, on a Modal T4 | Modal account |

Every script uses the **same known-answer dataset**: the model learns
`y = 3·x0 − 2·x1 + 1` in-context from 50 rows, then predicts on three points whose
true values are `[1.5, -4.0, 6.0]`. Each script asserts the predictions land within
a generous tolerance (`max abs error < 1.0`) — enough to catch real breakage while
tolerating normal float/precision drift between CPU, GPU, and the hosted model.
Expected predictions hover around `[1.45, -4.1, 6.06]`.

> Use a **fresh virtualenv** for each install so you're testing the published
> wheel, not a local checkout. (`python -m venv .dogfood && source .dogfood/bin/activate`)

---

## 1. Local mode (no network, no key)

```bash
pip install "synthefy[local]==4.0.1"     # Python >=3.10 recommended (>=3.9 supported)
python dogfood/dogfood_local.py
```

**Expected output:**

```
LOCAL OK
  preds   : [1.454, -4.126, 6.075]
  expected: [1.5, -4.0, 6.0]
  max err : 0.126 (tol 1.0)
```

✅ **Pass:** prints `LOCAL OK` and exits 0. (Uses a GPU automatically if present;
otherwise CPU — on CPU you'll see a one-line "mixed precision disabled" notice,
which is expected.)

---

## 2. Remote mode (hosted Baseten endpoint)

```bash
pip install "synthefy==4.0.1"            # lightweight; does NOT install torch
export BASETEN_API_KEY="<your baseten key>"
python dogfood/dogfood_remote.py
```

This hits the **gateway** (`https://inference.baseten.co/predict`), which routes by
the `synthefy/nori` model slug in the request body. The gateway is the only remote
path a Frontier/gateway key can reach — dedicated deployment URLs reject these keys
with a 403 by design, so the kit no longer targets them.

**Expected output:**

```
url: https://inference.baseten.co/predict | mode: remote
REMOTE OK
  preds   : [1.452, -4.155, 6.074]
  expected: [1.5, -4.0, 6.0]
  max err : 0.155 (tol 1.0)
```

✅ **Pass:** prints `REMOTE OK` and exits 0.
- `AuthenticationError` (401) / `PermissionDeniedError` (403) → bad/missing `BASETEN_API_KEY`.
- `BadRequestError` (400) → carries the server's error string.
- `InternalServerError` (5xx) → server-side; the client already retried with backoff.
- Hangs / read timeout → the hosted `synthefy/nori` deployment is asleep or unhealthy; confirm a `curl` to the gateway returns 200.

---

## 3. Modal T4 (local mode on a GPU)

Confirms `synthefy[local]` installs into a clean image and runs CUDA-accelerated.

```bash
# Auth once (writes ~/.modal.toml):
uvx modal setup
#   ...or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET

uvx modal run dogfood/modal_t4.py
```

**Expected output:**

```
ENV: {'torch': '2.12.0+cu130', 'cuda_available': True, 'device': 'Tesla T4'}
=== RESULT FROM MODAL T4 ===
  torch: 2.12.0+cu130
  cuda_available: True
  device: Tesla T4
  preds: [1.450..., -4.108..., 6.059...]
  expected: [1.5, -4.0, 6.0]
  max_err: 0.108...
  ok: True
MODAL T4 OK
```

✅ **Pass:** `cuda_available: True`, `device: Tesla T4`, `ok: True`, prints `MODAL T4 OK`.

> **Note:** running this incurs a small T4 compute charge on your Modal account.
> The first run builds the image (a few minutes; cached afterward).

---

## Troubleshooting

- **`ImportError: ... pip install "synthefy[local]"`** (local/modal) — the
  `synthefy-nori` extra isn't installed. Install it; needs Python ≥3.10 to be
  effective (the dep is gated to 3.10+).
- **`DeserializationError: 'torch' module not available`** when running Modal from
  a minimal env — that's why `modal_t4.py` returns a JSON *string*; don't return
  numpy/torch objects from a Modal function if your local runner lacks them.
- **`File already exists` on a release** — unrelated to dogfooding; means a PyPI
  version wasn't bumped.
- **Predictions off by > 1.0** — genuine regression; capture the full output and
  the install (`pip show synthefy synthefy-nori`) and flag it.

## Sign-off checklist

Copy into your dogfood ticket and tick what you ran:

- [ ] **Local** — `python dogfood/dogfood_local.py` → `LOCAL OK`
- [ ] **Remote (gateway)** — `python dogfood/dogfood_remote.py` → `REMOTE OK`
- [ ] **Modal T4** — `uvx modal run dogfood/modal_t4.py` → `MODAL T4 OK`
- [ ] OS / Python version noted: `__________`
- [ ] Anything surprising noted in the ticket
