# Synthefy Python Client

| Branch | Status |
|--------|--------|
| `dev` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Adev) |
| `main` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Amain) |

A Python client for the Synthefy API. It provides time series **forecasting** (synchronous and asynchronous) and **tabular in-context regression** via `SynthefyNoriClient` — which runs against the hosted endpoint or fully locally — with an easy-to-use interface, full type hints, and pydantic validation.

## Features

- **Tabular In-Context Regression**: `SynthefyNoriClient` predicts from labeled context rows in a single forward pass — hosted on Baseten or fully local, no training step
- **Sync & Async Support**: Separate clients for synchronous and asynchronous operations
- **Professional Error Handling**: Comprehensive exception hierarchy with detailed error messages
- **Retry Logic**: Built-in exponential backoff for transient errors (rate limits, server errors)
- **Context Managers**: Automatic resource cleanup with `with` and `async with` statements
- **Pandas Integration**: Built-in support for pandas DataFrames
- **Type Safety**: Full type hints and Pydantic validation

## Installation

```bash
pip install synthefy
```

For optional fully-local tabular inference (no API key, runs in-process), install the extra:

```bash
pip install "synthefy[local]"
```

## Quick Start

### Basic Usage

```python
from synthefy import SynthefyAPIClient, SynthefyAsyncAPIClient
import pandas as pd

# Synchronous client
with SynthefyAPIClient(api_key="your_api_key_here") as client:
    # Make requests...
    pass

# Asynchronous client
async with SynthefyAsyncAPIClient() as client:  # Uses SYNTHEFY_API_KEY env var
    # Make async requests...
    pass
```

### Making a Forecast Request

```python
from synthefy import SynthefyAPIClient
import pandas as pd
import numpy as np

# Create sample data with numeric metadata
history_data = {
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.normal(100, 10, 100),
    'store_id': 1,
    'category_id': 101,
    'promotion_active': 0
}

target_data = {
    'date': pd.date_range('2024-04-11', periods=30, freq='D'),
    'sales': np.nan,  # Values to forecast
    'store_id': 1,
    'category_id': 101,
    'promotion_active': 1  # Promotion active in forecast period
}

history_df = pd.DataFrame(history_data)
target_df = pd.DataFrame(target_data)

# Synchronous forecast
with SynthefyAPIClient() as client:
    forecast_dfs = client.forecast_dfs(
        history_dfs=[history_df],
        target_dfs=[target_df],
        target_col='sales',
        timestamp_col='date',
        metadata_cols=['store_id', 'category_id', 'promotion_active'],
        leak_cols=[],
        model='sfm-moe-v1'
    )

# Result is a list of DataFrames with forecasts
forecast_df = forecast_dfs[0]
print(forecast_df[['timestamps', 'sales']].head())
```

### Asynchronous Usage

```python
import asyncio
from synthefy.api_client import SynthefyAsyncAPIClient

async def main():
    async with SynthefyAsyncAPIClient() as client:
        # Single async forecast
        forecast_dfs = await client.forecast_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col='sales',
            timestamp_col='date',
            metadata_cols=['store_id', 'category_id', 'promotion_active'],
            leak_cols=[],
            model='sfm-moe-v1'
        )

        # Concurrent forecasts for multiple datasets
        tasks = []
        for i in range(3):
            # Create variations of your data
            modified_history = history_df.copy()
            modified_target = target_df.copy()
            modified_history['store_id'] = i + 1
            modified_target['store_id'] = i + 1

            task = client.forecast_dfs(
                history_dfs=[modified_history],
                target_dfs=[modified_target],
                target_col='sales',
                timestamp_col='date',
                metadata_cols=['store_id', 'category_id', 'promotion_active'],
                leak_cols=[],
                model='sfm-moe-v1'
            )
            tasks.append(task)

        # Execute all forecasts concurrently
        results = await asyncio.gather(*tasks)

        for i, forecast_dfs in enumerate(results):
            print(f"Forecast for store {i+1}: {len(forecast_dfs[0])} predictions")

# Run the async function
asyncio.run(main())
```

### Backtesting

```python
import asyncio
import pandas as pd
import numpy as np
from synthefy.data_models import ForecastV2Request
from synthefy.api_client import SynthefyAsyncAPIClient

async def main():

    # Create sample time series data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'date': dates,
        'sales': np.random.normal(100, 10, len(dates)),
        'store_id': 1,
        'category_id': 101,
        'promotion_active': np.random.choice([0, 1], len(dates), p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

    print(f"Created dataset with {len(df)} rows from {df['date'].min()} to {df['date'].max()}")

    # Use from_dfs_pre_split for backtesting with date-based windows
    request = ForecastV2Request.from_dfs_pre_split(
        dfs=[df],
        timestamp_col='date',
        target_cols=['sales'],
        model='sfm-moe-v1',
        cutoff_date='2023-06-01',  # Start backtesting from June 1st
        forecast_window='7D',      # 7-day forecast windows
        stride='14D',              # Move forward 14 days between windows
        metadata_cols=['store_id', 'category_id', 'promotion_active'],
        leak_cols=['promotion_active']  # Promotion data may leak into target
    )

    print(f"Created {len(request.samples)} forecast windows for backtesting")
    print("Window details:")
    for i, sample in enumerate(request.samples):
        history_start = sample[0].history_timestamps[0]
        history_end = sample[0].history_timestamps[-1]
        target_start = sample[0].target_timestamps[0]
        target_end = sample[0].target_timestamps[-1]
        print(f"  Window {i+1}: History {history_start} to {history_end}, Target {target_start} to {target_end}")

    # Make async forecast request
    async with SynthefyAsyncAPIClient() as client:
        response = await client.forecast(request)

        print(f"\nBacktesting completed with {len(response.samples)} forecast windows")

        # Process results for each window
        for i, sample in enumerate(response.samples):
            print(f"Window {i+1}: {len(sample.history_timestamps)} history points, "
                f"{len(sample.target_timestamps)} target points")

            # Access forecast values
            if hasattr(sample, 'forecast_values') and sample.forecast_values:
                print(f"  Forecast values: {sample.forecast_values[:3]}...")  # First 3 values
asyncio.run(main())
```

### Advanced Configuration

```python
from synthefy import SynthefyAPIClient
from synthefy.api_client import BadRequestError, RateLimitError

# Client with custom configuration
with SynthefyAPIClient(
    api_key="your_key",
    timeout=600.0,  # 10 minutes
    max_retries=3,
    organization="your_org_id",
    base_url="https://custom.synthefy.com"  # For enterprise customers
) as client:
    try:
        # Per-request configuration
        forecast_dfs = client.forecast_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col='sales',
            timestamp_col='date',
            metadata_cols=['store_id'],
            leak_cols=[],
            model='sfm-moe-v1',
            timeout=120.0,  # Override client timeout for this request
            idempotency_key="unique-request-id",  # Prevent duplicate processing
            extra_headers={"X-Custom-Header": "value"}
        )
    except BadRequestError as e:
        print(f"Invalid request: {e}")
        print(f"Status code: {e.status_code}")
        print(f"Request ID: {e.request_id}")
    except RateLimitError as e:
        print(f"Rate limited: {e}")
        # Client automatically retries with exponential backoff
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Nori — Tabular In-Context Regression

`SynthefyNoriClient` is a standalone client for **Synthefy Nori**, an
in-context learning regressor. Each call supplies labeled context rows
(`X_train`, `y_train`) and query rows (`X_test`); the model returns one predicted
value per query row in a single forward pass — there is no training step.

It is independent of the forecasting client above (different model, different
endpoint, different credential) but is exported from the same package. A single
`SynthefyNoriClient` runs predictions either against the hosted endpoint or
locally, selected with the `mode` argument (`"remote"` (default), `"local"`, or
`"auto"`).

### Use it from your AI coding assistant

Paste this into Claude Code, Cursor, or any AI coding assistant and it will wire
Nori into your own project:

````text
Look at my code/task/report here and figure out where Nori would best fit — it's
Synthefy's tabular foundation model for regression, used through the `synthefy`
client with no training loop and no hyperparameters. It runs fully on your own
machine (local mode, uses your GPU when one's available), or against the hosted
Synthefy API if you'd rather not run it locally.

1. Install it with this project's package manager, with the local extra
   (e.g. `uv add "synthefy[local]"`, or `pip install -U "synthefy[local]"`).

2. Use it wherever a tabular regression / prediction step fits:

   ```python
   from synthefy import SynthefyNoriClient

   # model is required -- name a size: "nori-30m" (~29.2M) or "nori-6m" (~6M base).
   client = SynthefyNoriClient(mode="local", model="nori-30m")   # runs on this machine, no API key

   y_pred = client.predict(
       X_train=X_train,   # lists, numpy arrays, or pandas — NaNs OK, imputed for you
       y_train=y_train,   # continuous target
       X_test=X_test,     # rows to score
   )                      # -> list of floats, one per X_test row (as_pandas=True for a Series)
   ```

X is a numeric feature matrix (or a pandas DataFrame — non-numeric columns are
encoded for you); y is a continuous target. If I already have a model, wire Nori
up alongside it on the same train/test split and metric so I can compare them. If
the best place to plug Nori in isn't obvious, show me where you'd put it and
confirm with me before making changes.

Prefer not to run it locally? Use the hosted API instead — sign up for a Baseten
API key at https://docs.synthefy.com/setup/api_key, then:
`client = SynthefyNoriClient(api_key="YOUR_BASETEN_API_KEY")` (or set BASETEN_API_KEY).
````

### Hosted Usage (`mode="remote"`)

```python
from synthefy import SynthefyNoriClient

# Auth uses a Baseten API key, sent as `Authorization: Bearer <key>` (gateway default).
# Pass it explicitly or set the BASETEN_API_KEY environment variable.
client = SynthefyNoriClient(api_key="your_baseten_api_key")

predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],  # context features
    y_train=[1.0, 1.0, 2.0],                        # context targets
    X_test=[[2.0, 2.0], [0.5, 0.5]],                # query features
)
print(predictions)  # -> [<float>, <float>]  (one per X_test row)
```

`X_train`, `y_train`, and `X_test` accept Python lists, numpy arrays, or pandas
objects (a `DataFrame` for the feature matrices; a `Series` or single-column
`DataFrame` for `y_train`). When both `X_train` and `X_test` are DataFrames,
**non-numeric columns are encoded for you** — fit on `X_train` and applied
to `X_test` — so you can pass raw categorical columns directly:

```python
import pandas as pd

X_train = pd.DataFrame({"price": [9.99, 4.50, 7.25], "region": ["NW", "SE", "NW"]})
y_train = pd.Series([120.0, 305.0, 180.0])
X_test  = pd.DataFrame({"region": ["SE"], "price": [5.00]})  # order need not match

predictions = client.predict(X_train, y_train, X_test)  # 'region' is encoded
```

By default each categorical column becomes a single column of **ordinal codes**
(categories from `X_train` in sorted order — the model's own server-side
convention): a value seen only in `X_test` maps to `-1`, and a missing value
(NaN) stays NaN for server-side imputation. Pass
`categorical_encoding="onehot"` for the previous one-hot behavior (indicator
columns per category; missing values get their own indicator; unseen values map
to an all-zeros group). Datetime columns and categorical columns with more than
`max_categorical_cardinality` (default 100) distinct training values are dropped
with a warning; `timedelta` columns are unsupported and raise (convert them to a
number or string first). Numeric columns (including `bool`) pass through unchanged,
with NaN imputed server-side. Any **object-dtype** column is treated as categorical
(including numeric-looking strings such as IDs or zip codes, and object date
values) — cast genuine numeric columns to a numeric dtype if you want them kept as
magnitudes. (Plain lists/numpy arrays must already be numeric — encoding needs
column names.)

Shapes are validated client-side: `X_train` and `y_train` must have the same
number of rows, and `X_test` must have the same number of features as `X_train`.
When both `X_train` and `X_test` are DataFrames, `X_test` is aligned to
`X_train`'s columns **by name** (so column order is irrelevant), and a mismatch
in the column sets raises. **Missing values (NaN) are allowed** — you don't need
to fill them in beforehand; the model imputes them server-side.

`predict` returns a plain `list[float]` by default. Pass **`as_pandas=True`** to
get a pandas `Series` instead — one value per `X_test` row, named after `y_train`
and indexed by `X_test`'s index (when `X_test` is a DataFrame), so predictions
join straight back:

```python
preds = client.predict(X_train, y_train, X_test, as_pandas=True)
# preds is a pd.Series named after y_train, sharing X_test's index
```

By default the client targets the Baseten inference **gateway**
(`https://inference.baseten.co/predict`); `model=` is required and names a size —
`"nori-30m"` (→ `synthefy/nori-30m`) or `"nori-6m"` (→ `synthefy/nori-6m`). To
target a **dedicated** deployment instead, point `base_url`/`endpoint` at it and
set `model=None` (the dedicated endpoint takes the body verbatim, with no `model`
field):

```python
from synthefy.nori_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT

client = SynthefyNoriClient(
    api_key="your_baseten_api_key",
    base_url=DEDICATED_BASE_URL,    # https://model-3m5j7y9w.api.baseten.co
    endpoint=DEDICATED_ENDPOINT,    # /environments/production/predict
    model=None,
    auth_scheme="Api-Key",          # dedicated endpoints use Api-Key, not Bearer
)
```

`timeout` and `max_retries` are also configurable on the constructor.

#### Authentication

- The only credential is a **Baseten API key** (there is no separate Synthefy
  key for this endpoint).
- Provide it via the `api_key` argument or the `BASETEN_API_KEY` environment
  variable. It is sent as the header `Authorization: <auth_scheme> <key>`. The
  scheme defaults to `Bearer` (required by the gateway); pass
  `auth_scheme="Api-Key"` for a dedicated deployment.

#### Errors

The Nori client reuses the package's
[exception hierarchy](#exception-hierarchy):

- HTTP `400` → `BadRequestError`, carrying the server's `error` string as the
  message (e.g. a missing field or unsupported task).
- HTTP `401` → `AuthenticationError` (bad or missing key).
- Transient errors (timeouts, connection errors, `429`, `5xx`) are retried with
  exponential backoff, then surface as `RateLimitError` / `InternalServerError` /
  `APITimeoutError` / `APIConnectionError`.

### Local Usage (`mode="local"`, Optional, No Network)

The same prediction can run locally — no network call and no API key — via the
optional [`synthefy-nori`](https://pypi.org/project/synthefy-nori/)
package. Install the extra:

```bash
pip install "synthefy[local]"
```

The `local` extra (via `synthefy-nori>=0.10.0`, which provides the `model=` size selector)
supports Python >= 3.9, the same floor as the base package.

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(mode="local")  # no API key needed
predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    y_train=[1.0, 1.0, 2.0],
    X_test=[[2.0, 2.0]],
)
```

`predict` has the same signature in every mode. The `synthefy-nori` dependency
is imported lazily on first use; if it is not installed, a clear `ImportError` is
raised telling you to `pip install "synthefy[local]"`.

Use `mode="auto"` to prefer local when `synthefy-nori` is installed and
transparently fall back to the hosted endpoint (which then requires an API key)
otherwise:

```python
client = SynthefyNoriClient(api_key="your_baseten_api_key", mode="auto")
print(client.mode)  # "local" if synthefy-nori is installed, else "remote"
```

### Categorical / Ordinal Targets (`discretize=` / `categorical_levels=`)

When the target only takes a small set of discrete values (a 1–5 rating, a
count, a quality score), pass `discretize=` and every returned prediction is
one of the target's own levels instead of a continuous estimate:

```python
labels = client.predict(X_train, y_train, X_test, discretize="snap-mean")
labels = client.predict(
    X_train, y_train, X_test,
    discretize="snap-mean",
    categorical_levels=[1, 2, 3, 4, 5],   # the full scale, if the context may under-cover it
)
```

Discretization is **strictly opt-in** — nothing is snapped unless you ask.
`categorical_levels` is the set of values the target can take (numeric; order
and duplicates don't matter); it defaults to the distinct values of `y_train`, which is
leak-safe. A `NaN` prediction stays `NaN` rather than becoming a confident
label.

Capability differs by mode:

- **Remote**: the hosted endpoint returns point predictions (the distribution
  mean), so the supported strategy is `discretize="snap-mean"` — the nearest
  level to the point prediction, computed client-side and identical to local
  `"snap-mean"`. Other strategies raise a `ValueError` pointing here.
- **Local** (`pip install "synthefy[local]"`, with a `synthefy-nori` recent
  enough to ship `synthefy_nori.discretize`): the full strategy set is
  forwarded — `"map-cell"` (accuracy-optimal), `"median-cell"` (MAE-optimal),
  `"snap-mean"` (QWK), `"snap-median"`, `"expected-level"`, `"prior-match"`.
  Choose by the metric you are scored on; see the `synthefy-nori` docs. An
  older `synthefy-nori` raises an `ImportError` with an upgrade hint.

If your task is scored by squared error / R², don't discretize — the
continuous mean is already optimal for those metrics.

## API Reference

### SynthefyNoriClient (Tabular Regression)

- `SynthefyNoriClient(api_key=None, *, mode="remote", timeout=300.0, max_retries=2, base_url=..., endpoint=..., model, user_agent=None)` — `model` is **required** (`"nori-6m"` / `"nori-30m"`; `None` for a dedicated endpoint)
  - `mode`: `"remote"` (hosted, default), `"local"` (in-process via
    `synthefy-nori`), or `"auto"` (local if installed, else remote).
  - `api_key` (remote mode) falls back to the `BASETEN_API_KEY` environment
    variable. Not required in local mode.
  - For a dedicated deployment, pass `base_url`/`endpoint` and `model=None`.
- `predict(X_train, y_train, X_test, task="regression", *, timeout=None, extra_headers=None) -> List[float]`
  - Returns one predicted value per row of `X_test`. `timeout`/`extra_headers`
    apply to remote mode only.
  - Inputs accept Python lists, numpy arrays, or pandas DataFrames/Series.
    Feature columns must be numeric; DataFrame `X_test` is aligned to `X_train`
    by column name; non-numeric columns are encoded (fit on `X_train`;
    `categorical_encoding="ordinal"` by default, `"onehot"` available);
    missing values (NaN) are imputed server-side.
  - `max_categorical_cardinality` (default 100): non-numeric columns with more
    distinct training values than this — and datetime columns — are dropped with
    a warning instead of encoded.
  - `as_pandas=True` returns a pandas `Series` (named after `y_train`, indexed by
    `X_test`) instead of the default `list[float]`.
  - `discretize=` / `categorical_levels=` map predictions onto a discrete
    target's levels (see
    [Categorical / Ordinal Targets](#categorical--ordinal-targets-discretize--categorical_levels));
    remote mode supports `discretize="snap-mean"`, local mode the full
    strategy set of the installed `synthefy-nori`.
- `mode`: the resolved mode (`"auto"` becomes `"local"`/`"remote"` at
  construction).
- `close()` / context manager support (`with SynthefyNoriClient(...) as client:`).

### SynthefyAPIClient (Synchronous)

The synchronous client class for interacting with the Synthefy API.

#### Constructor Parameters

- `api_key`: Your Synthefy API key (can also be set via `SYNTHEFY_API_KEY` environment variable)
- `timeout`: Request timeout in seconds (default: 300.0 / 5 minutes)
- `max_retries`: Number of retries for transient errors (default: 2)
- `base_url`: API base URL (default: "https://forecast.synthefy.com")
- `organization`: Optional organization ID for multi-tenant setups
- `user_agent`: Custom user agent string

#### Methods

- `forecast(request, *, timeout=None, idempotency_key=None, extra_headers=None) -> ForecastV2Response`
  - Make a direct forecast request with a `ForecastV2Request` object
- `forecast_dfs(history_dfs, target_dfs, target_col, timestamp_col, metadata_cols, leak_cols, model) -> List[pd.DataFrame]`
  - Convenience method for working directly with pandas DataFrames
- `close()`: Manually close the HTTP client
- Context manager support: Use with `with SynthefyAPIClient() as client:`

### SynthefyAsyncAPIClient (Asynchronous)

The asynchronous client class for non-blocking operations and concurrent requests.

#### Constructor Parameters

Same as `SynthefyAPIClient`.

#### Methods

- `async forecast(request, *, timeout=None, idempotency_key=None, extra_headers=None) -> ForecastV2Response`
  - Async version of forecast method
- `async forecast_dfs(history_dfs, target_dfs, target_col, timestamp_col, metadata_cols, leak_cols, model) -> List[pd.DataFrame]`
  - Async version of forecast_dfs method
- `async aclose()`: Manually close the async HTTP client
- Async context manager support: Use with `async with SynthefyAsyncAPIClient() as client:`

### Exception Hierarchy

All exceptions inherit from `SynthefyError`:

- `APITimeoutError`: Request timed out
- `APIConnectionError`: Network/connection issues
- `APIStatusError`: Base class for HTTP status errors
  - `BadRequestError` (400, 422): Invalid request data
  - `AuthenticationError` (401): Invalid API key
  - `PermissionDeniedError` (403): Access denied
  - `NotFoundError` (404): Resource not found
  - `RateLimitError` (429): Rate limit exceeded
  - `InternalServerError` (5xx): Server errors

Each status error includes:
- `status_code`: HTTP status code
- `request_id`: Request ID for debugging (if available)
- `error_code`: API-specific error code (if available)
- `response_body`: Raw response body

## Configuration

### Environment Variables

- `SYNTHEFY_API_KEY`: Your Synthefy API key (forecasting client)
- `BASETEN_API_KEY`: Your Baseten API key (`SynthefyNoriClient`)

## Support

For support and questions:
- Email: contact@synthefy.com

## License

MIT License - see LICENSE file for details.