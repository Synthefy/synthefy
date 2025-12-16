# Synthefy Python Client

| Branch | Status |
|--------|--------|
| `dev` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Adev) |
| `main` | [![Tests](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/Synthefy/synthefy/actions/workflows/tests.yaml?query=branch%3Amain) |

A Python client for the Synthefy forecasting API with sync and async support, pandas integration, and built-in retry logic.

## Documentation

For complete documentation, tutorials, and examples, visit:

- **[SDK Quickstart Guide](https://docs.synthefy.com/sdk/quickstart)** - Get started with forecasting in minutes
- **[API Reference](https://docs.synthefy.com/api-reference/python-sdk)** - Complete SDK reference documentation

## Installation

```bash
pip install synthefy
```

## Authentication

Get your API key from the [Synthefy Console](https://console.synthefy.com).

Set it as an environment variable:

```bash
export SYNTHEFY_API_KEY="your-api-key-here"
```

Or pass it directly to the client:

```python
from synthefy import SynthefyAPIClient

client = SynthefyAPIClient(api_key="your-api-key-here")
```

## Quick Example

```python
from synthefy import SynthefyAPIClient
import pandas as pd
import numpy as np

# Create historical data
history_df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.normal(100, 10, 100),
    'promotion_active': 0
})

# Create target data (what we want to forecast)
target_df = pd.DataFrame({
    'date': pd.date_range('2024-04-11', periods=30, freq='D'),
    'sales': np.nan,  # Values to forecast
    'promotion_active': 1
})

# Make forecast
with SynthefyAPIClient() as client:
    forecast_dfs = client.forecast_dfs(
        history_dfs=[history_df],
        target_dfs=[target_df],
        target_col='sales',
        timestamp_col='date',
        metadata_cols=['promotion_active'],
        leak_cols=[],
        model='Migas-1.0'
    )

# Get predictions
print(forecast_dfs[0].head())
```

## Support

- **Documentation**: [docs.synthefy.com](https://docs.synthefy.com)
- **Email**: contact@synthefy.com

## License

MIT License - see LICENSE file for details.
