"""
Unit test to verify Migas-1.0 handles NaN values in history correctly.

This test reproduces a bug where np.mean() and np.std() are used instead of
np.nanmean() and np.nanstd(), causing forecasts to return all NaN when
history contains any NaN values.

Usage:
    # Test against prod API (default)
    pytest tests/test_migas_nan_history.py -v -s

    # Test against local API (for testing the fix)
    FORECASTING_API_PORT=9619 pytest tests/test_migas_nan_history.py -v -s --synthefy-api-target=local
"""

import json

import numpy as np
import pandas as pd
import pytest
from synthefy import SynthefyAPIClient
from synthefy.data_models import ForecastV2Request


class TestMigasNanHistory:
    """Test suite for Migas forecasting with NaN values in history."""

    @pytest.fixture
    def simple_history_with_nan(self):
        """Create a simple time series history with some NaN values at the start."""
        # Quarterly data from 2020 Q1 to 2024 Q4 (20 quarters)
        dates = pd.date_range("2020-01-01", periods=20, freq="QS")

        # First 2 values are NaN, rest are valid values
        values = [None, None] + list(np.random.normal(0.02, 0.01, 18))

        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    @pytest.fixture
    def simple_target(self):
        """Create a simple target period (4 quarters to forecast)."""
        dates = pd.date_range("2025-01-01", periods=4, freq="QS")

        # Target values - these are the ground truth we want to forecast
        values = [0.015, 0.018, 0.012, 0.020]

        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    @pytest.fixture
    def gdp_history_from_json(self):
        """Load actual GDP history data from the JSON file."""
        with open(
            "/home/synthefy/data/aditya-dump/history_df_parquet-2025-12-15.json",
            "r",
        ) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df[["timestamp", "GDP_qoq"]].rename(
            columns={"GDP_qoq": "target"}
        )

    @pytest.fixture
    def gdp_target_from_json(self):
        """Load actual GDP target data from the JSON file."""
        with open(
            "/home/synthefy/data/aditya-dump/target_df_parquet-2025-12-15.json",
            "r",
        ) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df[["timestamp", "GDP_qoq"]].rename(
            columns={"GDP_qoq": "target"}
        )

    def test_migas_with_nan_in_history_simple(
        self,
        simple_history_with_nan,
        simple_target,
        synthefy_client: SynthefyAPIClient,
    ):
        """
        Test that Migas-1.0 can handle NaN values in history and return valid forecasts.

        This test uses a simple synthetic time series with 2 NaN values at the start.
        The forecast should not return all NaN values.
        """
        history_df = simple_history_with_nan
        target_df = simple_target

        # Verify we have NaN in history
        null_count = history_df["target"].isnull().sum()
        assert null_count > 0, (
            f"Expected NaN in history, got {null_count} nulls"
        )
        print(
            f"\nHistory has {null_count} NaN values out of {len(history_df)} rows"
        )

        # Create the request
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],
            leak_cols=[],
            model="Migas-1.0",
        )

        # Make the API call
        response = synthefy_client.forecast(request)

        # Convert to DataFrame
        result_dfs = response.to_dfs()
        assert len(result_dfs) == 1, (
            f"Expected 1 result DataFrame, got {len(result_dfs)}"
        )

        result_df = result_dfs[0]
        print(f"\nForecast result:\n{result_df}")

        # Check that the forecast values are NOT all NaN
        forecast_values = np.asarray(result_df["target"].values)
        nan_count = int(np.isnan(forecast_values).sum())

        assert nan_count < len(forecast_values), (
            f"Expected at least some non-NaN forecast values, but got {nan_count} NaN out of {len(forecast_values)}"
        )

        print(
            f"\nForecast has {nan_count} NaN values out of {len(forecast_values)} values"
        )
        print(f"Forecast values: {forecast_values}")

    def test_migas_with_real_gdp_data(
        self,
        gdp_history_from_json,
        gdp_target_from_json,
        synthefy_client: SynthefyAPIClient,
    ):
        """
        Test Migas-1.0 with actual GDP_qoq data that has NaN values.

        This reproduces the exact scenario from the failing forecast.
        """
        history_df = gdp_history_from_json
        target_df = gdp_target_from_json

        # Verify we have NaN in history
        null_count = history_df["target"].isnull().sum()
        print(
            f"\nGDP History has {null_count} NaN values out of {len(history_df)} rows"
        )
        print(
            f"Date range: {history_df['timestamp'].iloc[0]} to {history_df['timestamp'].iloc[-1]}"
        )

        # Create the request
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],
            leak_cols=[],
            model="Migas-1.0",
        )

        # Make the API call
        response = synthefy_client.forecast(request)

        # Convert to DataFrame
        result_dfs = response.to_dfs()
        assert len(result_dfs) == 1, (
            f"Expected 1 result DataFrame, got {len(result_dfs)}"
        )

        result_df = result_dfs[0]
        print(f"\nGDP Forecast result:\n{result_df}")

        # Check that the forecast values are NOT all NaN
        forecast_values = np.asarray(result_df["target"].values)
        nan_count = int(np.isnan(forecast_values).sum())

        assert nan_count < len(forecast_values), (
            f"BUG REPRODUCED: All forecast values are NaN! "
            f"Got {nan_count} NaN out of {len(forecast_values)}. "
            f"This is caused by np.mean()/np.std() being used instead of np.nanmean()/np.nanstd() in MigasEngine."
        )

        print(
            f"\nForecast has {nan_count} NaN values out of {len(forecast_values)} values"
        )

    def test_migas_without_nan_in_history(
        self, synthefy_client: SynthefyAPIClient
    ):
        """
        Control test: Migas-1.0 should work fine when history has no NaN values.
        """
        # Quarterly data from 2020 Q1 to 2024 Q4 (20 quarters) - NO NaN
        dates = pd.date_range("2020-01-01", periods=20, freq="QS")
        values = list(np.random.normal(0.02, 0.01, 20))

        history_df = pd.DataFrame(
            {"timestamp": dates.astype(str), "target": values}
        )

        # Target: 4 quarters
        target_dates = pd.date_range("2025-01-01", periods=4, freq="QS")
        target_df = pd.DataFrame(
            {
                "timestamp": target_dates.astype(str),
                "target": [0.015, 0.018, 0.012, 0.020],
            }
        )

        # Verify NO NaN in history
        null_count = history_df["target"].isnull().sum()
        assert null_count == 0, f"Expected no NaN in history, got {null_count}"

        # Create the request
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],
            leak_cols=[],
            model="Migas-1.0",
        )

        # Make the API call
        response = synthefy_client.forecast(request)

        # Convert to DataFrame
        result_dfs = response.to_dfs()
        result_df = result_dfs[0]
        print(f"\nControl test forecast result:\n{result_df}")

        # Check that the forecast values are NOT all NaN
        forecast_values = np.asarray(result_df["target"].values)
        nan_count = int(np.isnan(forecast_values).sum())

        assert nan_count == 0, (
            f"Expected no NaN forecast values when history has no NaN, "
            f"but got {nan_count} NaN out of {len(forecast_values)}"
        )

        print(
            f"\nControl test passed: All {len(forecast_values)} forecast values are valid"
        )


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
