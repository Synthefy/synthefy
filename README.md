# Synthefy Python Client

A Python client for the Synthefy API forecasting service. This package provides an easy-to-use interface for making time series forecasting requests with both synchronous and asynchronous support.

## Features

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

## API Reference

### SynthefyAPIClient (Synchronous)

The synchronous client class for interacting with the Synthefy API.

#### Constructor Parameters

- `api_key`: Your Synthefy API key (can also be set via `SYNTHEFY_API_KEY` environment variable)
- `timeout`: Request timeout in seconds (default: 300.0 / 5 minutes)
- `max_retries`: Number of retries for transient errors (default: 2)
- `base_url`: API base URL (default: "https://prod.synthefy.com")
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

- `SYNTHEFY_API_KEY`: Your Synthefy API key

## Support

For support and questions:
- Email: contact@synthefy.com

## License

MIT License - see LICENSE file for details.