# Synthefy 7 migration

The `synthefy` PyPI project continues, but its authoritative source moves from
this standalone repository to
[`libs/synthefy`](https://github.com/Synthefy/synthefy-nori/tree/main/libs/synthefy).
The consolidated repository builds the lightweight `synthefy` client and the
heavyweight `synthefy-nori` runtime from one reviewed checkout.

This retirement change must merge only after `synthefy` 7.0.0 and
`synthefy-nori` 0.17.0 are published from the public consolidated repository,
the hosted and SageMaker release checks pass, and the release owner completes
the observation window.

## What stays the same

- The package name is still `synthefy`.
- `SynthefyNoriClient` and `NoriPredictRequest` keep their existing names.
- Hosted Nori regression remains the lightweight installation path.

## What changes in 7.0

- Execution is deliberate: use `mode="remote"`, `mode="sagemaker"`, or
  `mode="local"`. The old `mode="auto"` fallback is removed.
- A Nori model must be specified explicitly on the client or forecaster; there
  is no default model.
- Local inference is installed with `pip install synthefy-nori`. The heavy
  package depends on the lightweight client, so the old `synthefy[local]` extra
  is no longer needed.
- Nori forecasting lives at `synthefy.nori_ts.NoriTSForecaster`. The legacy
  `SynthefyAPIClient`, `SynthefyAsyncAPIClient`, `ForecastV2Request`,
  `ForecastV2Response`, and `/v2/forecast` transport are retired.
- There is no drop-in async Nori client in the initial 7.0 release. A future
  async API can be added without preserving the retired ForecastV2 surface.

For hosted regression:

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(mode="remote", model="nori-30m")
predictions = client.predict(X_train, y_train, X_test)
client.close()
```

For local regression:

```bash
pip install synthefy-nori
```

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(mode="local", model="nori-30m")
predictions = client.predict(X_train, y_train, X_test)
client.close()
```

## Cutover checklist

1. Merge and validate the internal, staging, then public consolidation PRs.
2. Publish `synthefy` 7.0.0 first, then `synthefy-nori` 0.17.0.
3. Complete the hosted and SageMaker release checks and observation window.
4. Merge this migration notice and freeze this repository.
5. Revoke the old PyPI trusted-publisher identity for
   `Synthefy/synthefy/.github/workflows/publish.yaml`.
6. Archive this repository without deleting its tags or release history.

The publisher revocation and repository archival are manual release-owner
steps; this pull request deliberately does not perform them early.
