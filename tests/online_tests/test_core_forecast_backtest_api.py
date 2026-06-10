"""Integration tests for the "core forecast" concept (backtesting).

These tests intentionally hit a real Forecasting API.

- Default: prod (`https://forecast.synthefy.com`) using `SYNTHEFY_API_KEY`
- Dev override: `pytest --synthefy-api-target=dev` (`https://dev.forecast.synthefy.com`)
- Local override: `pytest --synthefy-api-target=local` (`http://localhost:{FORECASTING_API_PORT}`)

They mirror the patterns from the previous `raghav_test.py` debugging script:
- univariate backtesting by rows
- multivariate backtesting by rows
"""

import pandas as pd
import pytest
from synthefy.data_models import ForecastV2Request


class TestCoreForecastBacktestingAPI:
    @pytest.mark.asyncio
    async def test_univariate_backtest_by_rows_forecast_api(
        self, synthefy_async_client
    ):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=100, freq="D"
                ),
                "target": list(range(100, 200)),
            }
        )

        forecast_window = 7
        stride = 1
        num_target_rows = 10

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["target"],
            model="Migas-latest",
            num_target_rows=num_target_rows,
            forecast_window=forecast_window,
            stride=stride,
            metadata_cols=[],
            leak_cols=[],
        )

        response = await synthefy_async_client.forecast(request)

        assert len(response.forecasts) == len(request.samples)

        for scenario in response.forecasts:
            # At least the target should be present.
            sample_ids = [f.sample_id for f in scenario]
            assert "target" in sample_ids

            target_forecast = [f for f in scenario if f.sample_id == "target"][
                0
            ]
            assert len(target_forecast.timestamps) == forecast_window
            assert len(target_forecast.values) == forecast_window

    @pytest.mark.asyncio
    async def test_multivariate_backtest_by_rows_forecast_api(
        self, synthefy_async_client
    ):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=100, freq="D"
                ),
                "y1": list(range(100, 200)),
                "y2": list(range(1000, 1100)),
                "metadata": list(range(500, 600)),
            }
        )

        forecast_window = 7
        stride = 1
        num_target_rows = 10

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["y1", "y2"],
            model="Migas-latest",
            num_target_rows=num_target_rows,
            forecast_window=forecast_window,
            stride=stride,
            metadata_cols=["metadata"],
            leak_cols=[],
        )

        response = await synthefy_async_client.forecast(request)

        assert len(response.forecasts) == len(request.samples)

        for scenario in response.forecasts:
            sample_ids = [f.sample_id for f in scenario]
            assert "y1" in sample_ids
            assert "y2" in sample_ids

            y1_forecast = [f for f in scenario if f.sample_id == "y1"][0]
            y2_forecast = [f for f in scenario if f.sample_id == "y2"][0]

            assert len(y1_forecast.timestamps) == forecast_window
            assert len(y1_forecast.values) == forecast_window

            assert len(y2_forecast.timestamps) == forecast_window
            assert len(y2_forecast.values) == forecast_window
