"""
Unit test to verify Migas-1.0 handles NaN values in history correctly.

This test reproduces a bug where np.mean() and np.std() are used instead of
np.nanmean() and np.nanstd(), causing forecasts to return all NaN when
history contains any NaN values.

The bug was discovered when forecasting GDP_qoq which had NaN values at the
start of the history (missing data from 1946-1947).

Usage:
    # Test against prod API (default)
    pytest tests/online_tests/test_migas_nan_history.py -v -s

    # Test against dev API
    pytest tests/online_tests/test_migas_nan_history.py -v -s --synthefy-api-target=dev

    # Test against local API
    FORECASTING_API_PORT=9619 pytest tests/online_tests/test_migas_nan_history.py -v -s --synthefy-api-target=local
"""

import numpy as np
import pandas as pd
import pytest
from synthefy import SynthefyAPIClient
from synthefy.data_models import ForecastV2Request


class TestMigasNanHistory:
    """Test suite for Migas forecasting with NaN values in history."""

    @pytest.fixture
    def short_history_with_nan(self) -> pd.DataFrame:
        """Create a short time series history with NaN values at the start."""
        # Quarterly data - 20 quarters (5 years)
        dates = pd.date_range("2020-01-01", periods=20, freq="QS")

        # First 2 values are NaN (simulating missing early data)
        values = [None, None] + list(np.random.normal(0.02, 0.01, 18))

        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    @pytest.fixture
    def long_history_with_nan(self) -> pd.DataFrame:
        """
        Create a longer time series history with NaN values at the start.

        This simulates the GDP_qoq scenario where historical data from 1946
        had missing values at the beginning of the series.
        """
        # Quarterly data - 300+ quarters (~75 years, similar to GDP data from 1946)
        dates = pd.date_range("1950-01-01", periods=300, freq="QS")

        # First 5 values are NaN (simulating missing early data)
        values = [None] * 5 + list(np.random.normal(0.015, 0.02, 295))

        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    @pytest.fixture
    def short_target(self) -> pd.DataFrame:
        """Create a short target period (4 quarters to forecast)."""
        dates = pd.date_range("2025-01-01", periods=4, freq="QS")
        # Target values (ground truth)
        values = [0.015, 0.018, 0.012, 0.020]
        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    @pytest.fixture
    def long_target(self) -> pd.DataFrame:
        """Create a longer target period (12 quarters to forecast)."""
        dates = pd.date_range("2025-01-01", periods=12, freq="QS")
        # Target values (ground truth)
        values = list(np.random.normal(0.015, 0.005, 12))
        return pd.DataFrame({"timestamp": dates.astype(str), "target": values})

    def test_migas_short_history_with_nan(
        self,
        short_history_with_nan: pd.DataFrame,
        short_target: pd.DataFrame,
        synthefy_client: SynthefyAPIClient,
    ) -> None:
        """
        Test that Migas-1.0 handles NaN values in a short history.

        This test uses a 20-quarter history with 2 NaN values at the start.
        The forecast should return valid (non-NaN) values.
        """
        history_df = short_history_with_nan
        target_df = short_target

        # Verify we have NaN in history
        null_count = history_df["target"].isnull().sum()
        assert null_count > 0, (
            f"Expected NaN in history, got {null_count} nulls"
        )
        print(
            f"\nHistory has {null_count} NaN values out of {len(history_df)} rows"
        )

        # Create the request for univariate forecasting
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],  # Univariate - no metadata
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
            f"Expected at least some non-NaN forecast values, "
            f"but got {nan_count} NaN out of {len(forecast_values)}. "
            f"This indicates the NaN handling bug in MigasEngine."
        )

        print(
            f"\nForecast has {nan_count} NaN values out of {len(forecast_values)}"
        )
        print(f"Forecast values: {forecast_values}")

    def test_migas_long_history_with_nan(
        self,
        long_history_with_nan: pd.DataFrame,
        long_target: pd.DataFrame,
        synthefy_client: SynthefyAPIClient,
    ) -> None:
        """
        Test that Migas-1.0 handles NaN values in a long history.

        This test simulates the GDP_qoq scenario with ~300 quarters of data
        and 5 NaN values at the start (missing early historical data).
        """
        history_df = long_history_with_nan
        target_df = long_target

        # Verify we have NaN in history
        null_count = history_df["target"].isnull().sum()
        print(
            f"\nLong history has {null_count} NaN values out of {len(history_df)} rows"
        )
        print(
            f"Date range: {history_df['timestamp'].iloc[0]} to {history_df['timestamp'].iloc[-1]}"
        )

        # Create the request for univariate forecasting
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],  # Univariate - no metadata
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
            f"BUG: All forecast values are NaN! "
            f"Got {nan_count} NaN out of {len(forecast_values)}. "
            f"This is caused by np.mean()/np.std() not handling NaN values in MigasEngine."
        )

        print(
            f"\nForecast has {nan_count} NaN values out of {len(forecast_values)}"
        )

    def test_migas_without_nan_in_history(
        self,
        synthefy_client: SynthefyAPIClient,
    ) -> None:
        """
        Control test: Migas-1.0 should work fine when history has no NaN values.

        This ensures the fix for NaN handling doesn't break normal operation.
        """
        # Quarterly data - 20 quarters with NO NaN
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

        # Create the request for univariate forecasting
        request = ForecastV2Request.from_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col="target",
            timestamp_col="timestamp",
            metadata_cols=[],  # Univariate - no metadata
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
    pytest.main([__file__, "-v", "-s"])
