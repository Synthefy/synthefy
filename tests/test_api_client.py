import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from synthefy.api_client import SynthefyAPIClient, SynthefyAsyncAPIClient
from synthefy.data_models import ForecastV2Request


@pytest.fixture
def sample_forecast_data():
    """Returns sample data for end-to-end API testing with realistic time series and metadata."""

    # Generate date ranges
    start_date = datetime(2023, 1, 1)
    history_dates = pd.date_range(start_date, periods=100, freq="D")
    target_dates = pd.date_range(
        start_date + timedelta(days=100), periods=30, freq="D"
    )

    # Electronics store in North region
    np.random.seed(42)  # For reproducible tests
    trend = np.linspace(100, 150, 100)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(100) / 30)  # 30-day cycle
    noise = np.random.normal(0, 5, 100)
    sales = trend + seasonal + noise

    history_df = pd.DataFrame(
        {
            "date": history_dates,
            "sales": sales,
            "store_id": 1,  # Numeric store identifier
            "category_id": 101,  # Numeric category (electronics)
            "promotion_active": 0,  # Binary feature (0/1)
        }
    )

    target_df = pd.DataFrame(
        {
            "date": target_dates,
            "sales": np.nan,  # Required column, all NaN for forecasting
            "store_id": 1,  # Same numeric store identifier
            "category_id": 101,  # Same numeric category
            "promotion_active": 1,  # Promotion active in forecast period
        }
    )

    return {
        "history_dfs": [history_df],
        "target_dfs": [target_df],
        "target_col": "sales",
        "timestamp_col": "date",
        "metadata_cols": ["store_id", "category_id", "promotion_active"],
        "leak_cols": ["store_id", "category_id", "promotion_active"],
    }


def test_sync_forecast_raw_request(sample_forecast_data):
    """Test synchronous forecast using raw ForecastV2Request object."""
    data = sample_forecast_data

    # Create request object from DataFrames
    request = ForecastV2Request.from_dfs(
        history_dfs=data["history_dfs"],
        target_dfs=data["target_dfs"],
        target_col=data["target_col"],
        timestamp_col=data["timestamp_col"],
        metadata_cols=data["metadata_cols"],
        leak_cols=data["leak_cols"],
        model="sfm-moe-v1",
    )

    # Make API call
    with SynthefyAPIClient() as client:
        response = client.forecast(request)

    # Validate response
    assert response is not None
    assert hasattr(response, "to_dfs")

    # Convert to DataFrames and validate structure
    result_dfs = response.to_dfs()
    assert isinstance(result_dfs, list)
    assert len(result_dfs) == 1

    result_df = result_dfs[0]
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 30  # 30 days forecast
    assert "timestamps" in result_df.columns
    assert "sales" in result_df.columns


def test_sync_forecast_from_dfs(sample_forecast_data):
    """Test synchronous forecast using convenience forecast_dfs method."""
    data = sample_forecast_data

    # Make API call directly with DataFrames
    with SynthefyAPIClient() as client:
        result_dfs = client.forecast_dfs(
            history_dfs=data["history_dfs"],
            target_dfs=data["target_dfs"],
            target_col=data["target_col"],
            timestamp_col=data["timestamp_col"],
            metadata_cols=data["metadata_cols"],
            leak_cols=data["leak_cols"],
            model="sfm-moe-v1",
        )

    # Validate response
    assert isinstance(result_dfs, list)
    assert len(result_dfs) == 1

    result_df = result_dfs[0]
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 30  # 30 days forecast
    assert "timestamps" in result_df.columns
    assert "sales" in result_df.columns


@pytest.mark.asyncio
async def test_async_forecast_raw_request(sample_forecast_data):
    """Test asynchronous forecast using raw ForecastV2Request object."""
    data = sample_forecast_data

    # Create request object from DataFrames
    request = ForecastV2Request.from_dfs(
        history_dfs=data["history_dfs"],
        target_dfs=data["target_dfs"],
        target_col=data["target_col"],
        timestamp_col=data["timestamp_col"],
        metadata_cols=data["metadata_cols"],
        leak_cols=data["leak_cols"],
        model="sfm-moe-v1",
    )

    # Make async API call
    async with SynthefyAsyncAPIClient() as client:
        response = await client.forecast(request)

    # Validate response
    assert response is not None
    assert hasattr(response, "to_dfs")

    # Convert to DataFrames and validate structure
    result_dfs = response.to_dfs()
    assert isinstance(result_dfs, list)
    assert len(result_dfs) == 1

    result_df = result_dfs[0]
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 30  # 30 days forecast
    assert "timestamps" in result_df.columns
    assert "sales" in result_df.columns


@pytest.mark.asyncio
async def test_async_forecast_from_dfs(sample_forecast_data):
    """Test asynchronous forecast using convenience forecast_dfs method."""
    data = sample_forecast_data

    # Make async API call directly with DataFrames
    async with SynthefyAsyncAPIClient() as client:
        result_dfs = await client.forecast_dfs(
            history_dfs=data["history_dfs"],
            target_dfs=data["target_dfs"],
            target_col=data["target_col"],
            timestamp_col=data["timestamp_col"],
            metadata_cols=data["metadata_cols"],
            leak_cols=data["leak_cols"],
            model="sfm-moe-v1",
        )

    # Validate response
    assert isinstance(result_dfs, list)
    assert len(result_dfs) == 1

    result_df = result_dfs[0]
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 30  # 30 days forecast
    assert "timestamps" in result_df.columns
    assert "sales" in result_df.columns


@pytest.mark.asyncio
async def test_async_concurrent_forecasts(sample_forecast_data):
    """Test multiple concurrent async forecasts using asyncio.gather()."""
    data = sample_forecast_data

    # Create 3 copies of the data with slight variations
    forecast_tasks = []

    async with SynthefyAsyncAPIClient() as client:
        for i in range(3):
            # Create slightly different data for each request
            history_df = data["history_dfs"][0].copy()
            target_df = data["target_dfs"][0].copy()

            # Vary the store_id to make requests distinct
            history_df["store_id"] = i + 1
            target_df["store_id"] = i + 1

            # Create the forecast task
            task = client.forecast_dfs(
                history_dfs=[history_df],
                target_dfs=[target_df],
                target_col=data["target_col"],
                timestamp_col=data["timestamp_col"],
                metadata_cols=data["metadata_cols"],
                leak_cols=data["leak_cols"],
                model="sfm-moe-v1",
            )
            forecast_tasks.append(task)

        # Execute all forecasts concurrently
        results = await asyncio.gather(*forecast_tasks)

    # Validate all results
    assert len(results) == 3

    for i, result_dfs in enumerate(results):
        assert isinstance(result_dfs, list)
        assert len(result_dfs) == 1

        result_df = result_dfs[0]
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 30  # 30 days forecast
        assert "timestamps" in result_df.columns
        assert "sales" in result_df.columns
