import asyncio
import os
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


@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
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


@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
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
@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
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
@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
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
@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
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


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
async def test_backtesting_example():
    """Test the backtesting workflow from README.md example."""
    # Create sample time series data matching README.md example
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible results
    data = {
        'date': dates,
        'sales': np.random.normal(100, 10, len(dates)),
        'store_id': 1,
        'category_id': 101,
        'promotion_active': np.random.choice([0, 1], len(dates), p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

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

    # Validate request structure
    assert len(request.samples) > 1, "Should have multiple forecast windows for backtesting"
    assert request.model == 'sfm-moe-v1'
    
    # Validate each sample has proper structure
    for i, sample in enumerate(request.samples):
        assert len(sample) >= 1, f"Sample {i} should have at least one time series"
        # Check that we have both history and target timestamps
        first_sample = sample[0]
        assert len(first_sample.history_timestamps) > 0, f"Sample {i} should have history timestamps"
        assert len(first_sample.target_timestamps) > 0, f"Sample {i} should have target timestamps"

    # Make async forecast request
    async with SynthefyAsyncAPIClient() as client:
        response = await client.forecast(request)

        # Validate response structure
        assert response is not None
        assert hasattr(response, 'forecasts')
        assert len(response.forecasts) == len(request.samples), "Response should have same number of forecast scenarios as request samples"
        
        # Validate each forecast scenario
        for i, forecast_scenario in enumerate(response.forecasts):
            assert isinstance(forecast_scenario, list), f"Forecast scenario {i} should be a list"
            assert len(forecast_scenario) > 0, f"Forecast scenario {i} should have at least one forecast"
            
            # Check each forecast in the scenario
            for j, forecast in enumerate(forecast_scenario):
                assert hasattr(forecast, 'sample_id'), f"Forecast {i},{j} should have sample_id"
                assert hasattr(forecast, 'timestamps'), f"Forecast {i},{j} should have timestamps"
                assert hasattr(forecast, 'values'), f"Forecast {i},{j} should have values"
                assert hasattr(forecast, 'model_name'), f"Forecast {i},{j} should have model_name"
                
                # Only validate timestamps for forecast columns (not metadata columns)
                if forecast.sample_id == 'sales':  # Only the target column should have timestamps
                    assert len(forecast.timestamps) > 0, f"Sales forecast {i},{j} should have timestamps"
                    assert len(forecast.values) == len(forecast.timestamps), f"Sales forecast {i},{j} values should match timestamps length"
                # Metadata columns (store_id, category_id, promotion_active) may have empty timestamps - that's expected

        # Test conversion to DataFrames
        result_dfs = response.to_dfs()
        assert isinstance(result_dfs, list)
        assert len(result_dfs) == len(request.samples), "Should have one DataFrame per sample"
        
        for i, result_df in enumerate(result_dfs):
            assert isinstance(result_df, pd.DataFrame), f"Result {i} should be a DataFrame"
            assert 'timestamps' in result_df.columns, f"DataFrame {i} should have timestamps column"
            assert 'sales' in result_df.columns, f"DataFrame {i} should have sales column"


@pytest.mark.skipif(
    not os.getenv("SYNTHEFY_API_KEY"), 
    reason="SYNTHEFY_API_KEY environment variable not set"
)
def test_per_request_configuration():
    """Test per-request configuration features from README.md example."""
    # Create sample data for testing
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
        'promotion_active': 1
    }
    
    history_df = pd.DataFrame(history_data)
    target_df = pd.DataFrame(target_data)

    # Test with custom configuration
    with SynthefyAPIClient(
        timeout=600.0,  # 10 minutes
        max_retries=3,
        organization="test_org_id"
    ) as client:
        # Create request object for per-request configuration testing
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col='sales',
            timestamp_col='date',
            metadata_cols=['store_id', 'category_id', 'promotion_active'],
            leak_cols=[],
            model='sfm-moe-v1'
        )
        
        # Test per-request configuration overrides
        response = client.forecast(
            request,
            timeout=120.0,  # Override client timeout for this request
            idempotency_key="test-unique-request-id",  # Prevent duplicate processing
            extra_headers={"X-Custom-Header": "test-value", "X-Test-Source": "unit-test"}
        )

        # Validate response structure
        assert response is not None
        assert hasattr(response, 'forecasts')
        assert len(response.forecasts) == 1, "Should have one forecast scenario"
        
        # Validate forecast scenario
        forecast_scenario = response.forecasts[0]
        assert isinstance(forecast_scenario, list)
        assert len(forecast_scenario) > 0, "Should have at least one forecast"
        
        # Check forecast structure
        for forecast in forecast_scenario:
            assert hasattr(forecast, 'sample_id')
            assert hasattr(forecast, 'timestamps')
            assert hasattr(forecast, 'values')
            assert hasattr(forecast, 'model_name')
            
            # Only validate timestamps for forecast columns (not metadata columns)
            if forecast.sample_id == 'sales':  # Only the target column should have timestamps
                assert len(forecast.timestamps) > 0, "Sales forecast should have timestamps"
                assert len(forecast.values) == len(forecast.timestamps), "Sales forecast values should match timestamps length"
            # Metadata columns (store_id, category_id, promotion_active) may have empty timestamps - that's expected

        # Test conversion to DataFrames
        result_dfs = response.to_dfs()
        assert isinstance(result_dfs, list)
        assert len(result_dfs) == 1
        
        result_df = result_dfs[0]
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 30  # 30 days forecast
        assert "timestamps" in result_df.columns
        assert "sales" in result_df.columns
        
        # Validate that the forecast completed successfully with custom configuration
        assert not result_df['sales'].isna().all(), "Should have forecast values, not all NaN"
