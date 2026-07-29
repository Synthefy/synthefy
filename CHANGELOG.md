# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0]

### Removed (breaking)

- **`DEDICATED_BASE_URL` and `DEDICATED_ENDPOINT` are gone.** `from synthefy.nori_client import
  DEDICATED_BASE_URL, DEDICATED_ENDPOINT` now raises `ImportError`. Nori is addressed by gateway
  slug: `SynthefyNoriClient(api_key=..., model="nori-6m")` sends `{"model": "synthefy/nori-6m"}`
  to `https://inference.baseten.co/predict`, and the gateway resolves the slug to a deployment.

  A hardcoded `model-<id>.api.baseten.co` host cannot stay correct. Each Nori variant is its own
  Baseten model with its own id, so one constant can address at most one of them, and an id does
  not survive a model being deleted and re-created. The gateway slug is stable across both.

  The gateway is also the only path Synthefy meters, rate-limits and grants per key, so it is the
  one supported way to reach hosted Nori.

### Migration

```python
# before
from synthefy.nori_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT
client = SynthefyNoriClient(
    api_key=key, base_url=DEDICATED_BASE_URL, endpoint=DEDICATED_ENDPOINT,
    model=None, auth_scheme="Api-Key",
)

# after
client = SynthefyNoriClient(api_key=key, model="nori-6m")
```

`base_url`, `endpoint`, `auth_scheme` and `model=None` all still exist as generic overrides for
pointing the client at a host of your own. Only the two Synthefy-specific constants are removed.

## [5.0.0]

### Changed (breaking)

- **`model=` is now REQUIRED** on `SynthefyNoriClient` — there is no default. Every request
  names a size: `model="nori-6m"` (~6M base) or `model="nori-30m"` (~29.2M). Omitting `model`
  raises `ValueError`. (`model=None` still targets a dedicated deployment endpoint.) This keeps a
  model identifier from ever silently changing which model it serves.
- **Removed the bare `nori` / `synthefy/nori` selectors.** Only size-explicit names/slugs resolve:
  `"nori-6m"` / `"synthefy/nori-6m"` and `"nori-30m"` / `"synthefy/nori-30m"` (plus the hosted-only
  `nori-30m-thinking*`). The `GATEWAY_MODEL` constant is removed.

### Migration

- Add an explicit size: `SynthefyNoriClient(api_key=..., model="nori-30m")` (was the ~6M default;
  pass `model="nori-6m"` to keep the smaller base).

## [4.3.0]

### Changed

- **Default categorical encoding is now ordinal** (was one-hot).
  `SynthefyNoriClient.predict` maps each non-numeric DataFrame column to a
  single column of integer codes in sorted-category order — the same
  convention as the Nori model's server-side `OrdinalEncoder` path (unseen
  test value → `-1`, missing → NaN, imputed server-side). A 35-dataset
  benchmark across three model sizes found ordinal at least as accurate as
  one-hot on average, with one-hot substantially worse on wide,
  categorical-heavy tables (it never widens the feature matrix).
  Pass `categorical_encoding="onehot"` to restore the previous behavior.

### Added

- `categorical_encoding` parameter on `predict` (`"ordinal"` (default) or
  `"onehot"`).

## [4.2.2]

### Fixed

- `SynthefyNoriClient` remote retries no longer surface a stale earlier-attempt
  exception. When an earlier attempt raised a transient error (e.g. a connection
  error or timeout) but the final attempt returned a retryable response (e.g. a
  5xx), the client raised the stale `APIConnectionError`/`APITimeoutError`
  instead of the true final error. `_post_with_retries` now resets its per-attempt
  state each iteration, so the final attempt's outcome is what's raised (e.g. a
  5xx maps to `InternalServerError`).

## [4.2.1]

### Added

- `SynthefyNoriClient.predict` now **one-hot encodes non-numeric columns** when
  both `X_train` and `X_test` are DataFrames, instead of raising. The encoding is
  fit on `X_train` and applied to `X_test` (a category seen only in `X_test`
  becomes an all-zeros indicator group), producing a fully numeric, model-ready
  matrix client-side — no server change and no reliance on server-side category
  detection. Numeric columns (including `bool`) pass through unchanged. A missing
  value (NaN) in a categorical column gets its own one-hot indicator
  (`dummy_na`), kept only when missingness actually occurs.
- New `max_categorical_cardinality` argument to `predict` (default `100`):
  non-numeric columns with more than this many distinct training values — plus
  datetime columns and columns with no non-missing values — are dropped with a
  `UserWarning` rather than exploding the feature matrix.
- `category` columns whose categories are numeric are kept as a numeric
  magnitude (not one-hot exploded), NaN-safe for integer categories.

### Changed

- A non-numeric column in a DataFrame `X_train`/`X_test` pair no longer raises;
  it is one-hot encoded (see above). Passing a non-numeric column with a
  non-DataFrame `X_test` still raises, since one-hot alignment needs column
  names on both sides. A column that is numeric in one of `X_train`/`X_test` but
  not the other raises a clear type-mismatch `ValueError`; duplicate column
  names, name/value one-hot collisions, and `timedelta` columns also raise with
  actionable messages (convert timedeltas to a number or string first).

## [4.2.0]

### Added

- `SynthefyNoriClient.predict` now accepts **pandas** inputs in addition to
  Python lists and numpy arrays: a `DataFrame` for `X_train`/`X_test`, and a
  `Series` or single-column `DataFrame` for `y_train`. All feature columns must
  be numeric — non-numeric (categorical/text/datetime) columns raise a clear
  `ValueError` directing the caller to encode them first. Missing values (NaN)
  are allowed and imputed server-side; no need to fill them in beforehand.
- When both `X_train` and `X_test` are DataFrames, `X_test` is aligned to
  `X_train`'s columns **by name**, so column order no longer has to match.
  Mismatched column *sets* raise `ValueError`.
- `predict(..., as_pandas=True)` returns a pandas `Series` instead of the default
  `list[float]`: one value per `X_test` row, named after `y_train` (its `Series`
  name or single-column `DataFrame` label, else `"prediction"`) and indexed by
  `X_test`'s index when it is a pandas object. Mirrors AutoGluon's `as_pandas`;
  defaults to `False` so existing callers are unaffected.

## [4.1.3]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.9.0` (was `>=0.8.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.9.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.2]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.8.0` (was `>=0.6.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.8.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.1]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.6.0` (was `>=0.5.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.6.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.1.0]

### Fixed

- **Remote gateway authentication.** Requests to the default Baseten inference
  gateway (`https://inference.baseten.co/predict`) now send
  `Authorization: Bearer <key>` instead of `Authorization: Api-Key <key>`. The
  gateway accepts only the `Bearer` scheme, so every default-configured remote
  `predict(...)` call previously failed with HTTP `403` ("please check the
  api-key you provided") even when the key was valid. Dedicated deployments
  continue to use `Api-Key` (see below).

### Added

- New `auth_scheme` constructor argument on `SynthefyNoriClient`
  (`{"Bearer", "Api-Key"}`, default `"Bearer"`). The default fixes gateway
  auth out of the box; pass `auth_scheme="Api-Key"` when targeting a dedicated
  deployment. Invalid values raise `ValueError`.

## [4.0.1]

### Changed

- Bumped the `local` extra's floor to `synthefy-nori>=0.5.0` (was `>=0.1.0`) so
  `pip install "synthefy[local]"` pulls in the latest local-inference package.
  `synthefy-nori>=0.5.0` still supports Python >=3.9, matching the base
  package's floor, so no environment marker is needed. No code or public API
  changes.

## [4.0.0]

The tabular in-context regression product is now **Synthefy Nori**. This is a
breaking release: the client class, models, module, and optional local-inference
package were all renamed from `tabular` to `nori`. The forecasting client
(`SynthefyAPIClient` / `SynthefyAsyncAPIClient`) is unchanged, and the
`predict(...)` signature is identical — only names changed.

### Changed (BREAKING)

- Renamed `SynthefyTabularClient` → `SynthefyNoriClient`. There is no
  backward-compatible alias; the old name no longer imports.
- Renamed the request/response models `TabularPredictRequest` →
  `NoriPredictRequest` and `TabularPredictResponse` → `NoriPredictResponse`.
- Renamed the module `synthefy.tabular_client` → `synthefy.nori_client`. Imports
  such as `from synthefy.nori_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT`
  must be updated.
- The `local` extra now installs `synthefy-nori>=0.1.0` (was
  `synthefy-tabular>=0.2.3`). Local inference now imports from the `synthefy_nori`
  package. `pip install "synthefy[local]"` is unchanged.
- The default hosted gateway model identifier is now `synthefy/nori` (was
  `synthefy/synthefy-tabular`). The dedicated deployment `base_url`/`endpoint` are
  unchanged.

## [3.1.2]

### Changed

- Bumped the `local` extra's floor to `synthefy-tabular>=0.2.3` (was `>=0.2.2`)
  so `pip install "synthefy[local]"` pulls in the latest local-inference
  package. Still supports Python >=3.9, so no environment marker is needed.

## [3.1.1]

### Changed

- Documentation only (PyPI long description): the README now reflects the tabular
  client. The intro, feature list, and installation instructions cover
  `SynthefyTabularClient` (hosted and local modes) alongside forecasting. No code
  or API changes — released solely to refresh the immutable PyPI project page.

## [3.1.0]

### Added

- `SynthefyTabularClient`: a standalone, synchronous client for Synthefy Tabular
  in-context regression. Supply labeled context rows (`X_train`, `y_train`) and
  query rows (`X_test`) and receive one prediction per query row in a single
  forward pass — no training step. Accepts Python lists or numpy arrays,
  validates shapes, and is exported from the top-level `synthefy` package.
  - A single `mode` argument selects how predictions run: `"remote"` (default,
    hosted Baseten endpoint), `"local"` (in-process, no network, no API key), or
    `"auto"` (local if the optional package is installed, else remote).
  - Remote mode authenticates with a Baseten API key sent as
    `Authorization: Api-Key <key>`, taken from the `api_key` argument or the
    `BASETEN_API_KEY` environment variable, and defaults to the Baseten inference
    gateway (`https://inference.baseten.co/predict`, model
    `synthefy/synthefy-tabular`). To target a dedicated deployment, pass
    `base_url`/`endpoint` and `model=None`.
  - Reuses the package's existing error types: HTTP 400 maps to `BadRequestError`
    (carrying the server's `error` string) and 401 to `AuthenticationError`, with
    the same retry/backoff behavior as the forecasting client.
  - Local mode runs via the optional `synthefy-tabular` package, exposed through
    the new `local` extra: `pip install "synthefy[local]"` (supports Python >=3.9
    via `synthefy-tabular>=0.2.2`, matching the base package's floor).
- `TabularPredictRequest` and `TabularPredictResponse` pydantic models, exported
  from the top-level `synthefy` package.

## [3.0.0]

- Baseline release of the Synthefy forecasting client (`SynthefyAPIClient`,
  `SynthefyAsyncAPIClient`).
