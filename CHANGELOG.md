# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
