# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
