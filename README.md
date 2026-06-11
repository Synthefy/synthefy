# Synthefy Python Client

| Branch | Status |
|--------|--------|
| `dev` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Adev) |
| `main` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Amain) |

A Python client for the Synthefy API. It provides time series **forecasting** (synchronous and asynchronous) and **tabular in-context regression** via `SynthefyTabularClient` — which runs against the hosted endpoint or fully locally — with an easy-to-use interface, full type hints, and pydantic validation.

## Features

- **Tabular In-Context Regression**: `SynthefyTabularClient` predicts from labeled context rows in a single forward pass — hosted on Baseten or fully local, no training step
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

## Tabular Regression (In-Context)

`SynthefyTabularClient` is a standalone client for **Synthefy Tabular**, an
in-context learning regressor. Each call supplies labeled context rows
(`X_train`, `y_train`) and query rows (`X_test`); the model returns one predicted
value per query row in a single forward pass — there is no training step.

It is independent of the forecasting client above (different model, different
endpoint, different credential) but is exported from the same package. A single
`SynthefyTabularClient` runs predictions either against the hosted endpoint or
locally, selected with the `mode` argument (`"remote"` (default), `"local"`, or
`"auto"`).

### Hosted Usage (`mode="remote"`)

```python
from synthefy import SynthefyTabularClient

# Auth uses a Baseten API key, sent as `Authorization: Api-Key <key>`.
# Pass it explicitly or set the BASETEN_API_KEY environment variable.
client = SynthefyTabularClient(api_key="your_baseten_api_key")

predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],  # context features
    y_train=[1.0, 1.0, 2.0],                        # context targets
    X_test=[[2.0, 2.0], [0.5, 0.5]],                # query features
)
print(predictions)  # -> [<float>, <float>]  (one per X_test row)
```

`X_train`, `y_train`, and `X_test` accept either Python lists or numpy arrays.
Shapes are validated client-side: `X_train` and `y_train` must have the same
number of rows, and `X_test` must have the same number of features as `X_train`.

By default the client targets the Baseten inference **gateway**
(`https://inference.baseten.co/predict`, model `synthefy/synthefy-tabular`). To
target a **dedicated** deployment instead, point `base_url`/`endpoint` at it and
set `model=None` (the dedicated endpoint takes the body verbatim, with no `model`
field):

```python
from synthefy.tabular_client import DEDICATED_BASE_URL, DEDICATED_ENDPOINT

client = SynthefyTabularClient(
    api_key="your_baseten_api_key",
    base_url=DEDICATED_BASE_URL,    # https://model-3m5j7y9w.api.baseten.co
    endpoint=DEDICATED_ENDPOINT,    # /environments/production/predict
    model=None,
)
```

`timeout` and `max_retries` are also configurable on the constructor.

#### Authentication

- The only credential is a **Baseten API key** (there is no separate Synthefy
  key for this endpoint).
- Provide it via the `api_key` argument or the `BASETEN_API_KEY` environment
  variable. It is sent as the header `Authorization: Api-Key <key>`.

#### Errors

The tabular client reuses the package's
[exception hierarchy](#exception-hierarchy):

- HTTP `400` → `BadRequestError`, carrying the server's `error` string as the
  message (e.g. a missing field or unsupported task).
- HTTP `401` → `AuthenticationError` (bad or missing key).
- Transient errors (timeouts, connection errors, `429`, `5xx`) are retried with
  exponential backoff, then surface as `RateLimitError` / `InternalServerError` /
  `APITimeoutError` / `APIConnectionError`.

### Local Usage (`mode="local"`, Optional, No Network)

The same prediction can run locally — no network call and no API key — via the
optional [`synthefy-tabular`](https://pypi.org/project/synthefy-tabular/)
package. Install the extra:

```bash
pip install "synthefy[local]"
```

The `local` extra (via `synthefy-tabular>=0.2.2`) supports Python >= 3.9, the
same floor as the base package.

```python
from synthefy import SynthefyTabularClient

client = SynthefyTabularClient(mode="local")  # no API key needed
predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    y_train=[1.0, 1.0, 2.0],
    X_test=[[2.0, 2.0]],
)
```

`predict` has the same signature in every mode. The `synthefy-tabular` dependency
is imported lazily on first use; if it is not installed, a clear `ImportError` is
raised telling you to `pip install "synthefy[local]"`.

Use `mode="auto"` to prefer local when `synthefy-tabular` is installed and
transparently fall back to the hosted endpoint (which then requires an API key)
otherwise:

```python
client = SynthefyTabularClient(api_key="your_baseten_api_key", mode="auto")
print(client.mode)  # "local" if synthefy-tabular is installed, else "remote"
```

## API Reference

### SynthefyTabularClient (Tabular Regression)

- `SynthefyTabularClient(api_key=None, *, mode="remote", timeout=300.0, max_retries=2, base_url=..., endpoint=..., model="synthefy/synthefy-tabular", user_agent=None)`
  - `mode`: `"remote"` (hosted, default), `"local"` (in-process via
    `synthefy-tabular`), or `"auto"` (local if installed, else remote).
  - `api_key` (remote mode) falls back to the `BASETEN_API_KEY` environment
    variable. Not required in local mode.
  - For a dedicated deployment, pass `base_url`/`endpoint` and `model=None`.
- `predict(X_train, y_train, X_test, task="regression", *, timeout=None, extra_headers=None) -> List[float]`
  - Returns one predicted value per row of `X_test`. `timeout`/`extra_headers`
    apply to remote mode only.
- `mode`: the resolved mode (`"auto"` becomes `"local"`/`"remote"` at
  construction).
- `close()` / context manager support (`with SynthefyTabularClient(...) as client:`).

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
- `BASETEN_API_KEY`: Your Baseten API key (`SynthefyTabularClient`)

## Support

For support and questions:
- Email: contact@synthefy.com

## License

MIT License - see LICENSE file for details.