"""
Inventory forecasting use case test.

This test demonstrates comparing univariate vs multivariate forecasting
for retail inventory items using weather data as covariates.
"""

import numpy as np
import pandas as pd
import pytest


class TestInventoryForecasting:
    """Test class for inventory forecasting use case."""

    DATA_URL = "https://drive.google.com/uc?export=download&id=1YcAFpFzcZgX0elekB_vdWrzTzEMbGzuD"
    TIMESTAMP_COL = "date"
    GROUP_COL = "item_name"
    TARGET_COL = "sales"
    METADATA_COLS = [
        "temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
        "precipitation",
    ]
    HISTORY_RATIO = 0.8

    @pytest.fixture
    def data_df(self) -> pd.DataFrame:
        """Load and prepare the Chicago store sales dataset."""
        df = pd.read_csv(self.DATA_URL)
        df[self.TIMESTAMP_COL] = pd.to_datetime(df[self.TIMESTAMP_COL])
        df = df.sort_values(self.TIMESTAMP_COL).reset_index(drop=True)
        return df

    @pytest.fixture
    def available_items(self, data_df: pd.DataFrame) -> list[str]:
        """Get list of available items in the dataset."""
        return data_df[self.GROUP_COL].unique().tolist()

    def _split_data(
        self, data_df: pd.DataFrame, item_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split item data into history and target periods."""
        item_data = data_df[data_df[self.GROUP_COL] == item_name].sort_values(
            self.TIMESTAMP_COL
        )
        split_idx = int(len(item_data) * self.HISTORY_RATIO)
        history_df = item_data.iloc[:split_idx].copy()
        target_df = item_data.iloc[split_idx:].copy()
        return history_df, target_df

    def _run_univariate_forecast(
        self,
        client,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run univariate forecast (time series only)."""
        forecast_dfs = client.forecast_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col=self.TARGET_COL,
            timestamp_col=self.TIMESTAMP_COL,
            metadata_cols=[],
            leak_cols=[],
            model="Migas-1.0",
        )
        return forecast_dfs[0]

    def _run_multivariate_forecast(
        self,
        client,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run multivariate forecast (time series + weather covariates)."""
        forecast_dfs = client.forecast_dfs(
            history_dfs=[history_df],
            target_dfs=[target_df],
            target_col=self.TARGET_COL,
            timestamp_col=self.TIMESTAMP_COL,
            metadata_cols=self.METADATA_COLS,
            leak_cols=self.METADATA_COLS,
            model="Migas-1.0",
        )
        return forecast_dfs[0]

    @pytest.mark.parametrize("item_name", ["soup", "tea", "bread", "ice_cream"])
    def test_univariate_forecast_for_item(
        self, data_df: pd.DataFrame, item_name: str, synthefy_client
    ):
        """Test univariate forecasting for each item."""
        history_df, target_df = self._split_data(data_df, item_name)

        forecast_df = self._run_univariate_forecast(
            synthefy_client, history_df, target_df
        )

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(target_df)
        assert "timestamps" in forecast_df.columns
        assert self.TARGET_COL in forecast_df.columns
        assert not forecast_df[self.TARGET_COL].isna().all()

    @pytest.mark.parametrize("item_name", ["soup", "tea", "bread", "ice_cream"])
    def test_multivariate_forecast_for_item(
        self, data_df: pd.DataFrame, item_name: str, synthefy_client
    ):
        """Test multivariate forecasting with weather covariates for each item."""
        history_df, target_df = self._split_data(data_df, item_name)

        forecast_df = self._run_multivariate_forecast(
            synthefy_client, history_df, target_df
        )

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(target_df)
        assert "timestamps" in forecast_df.columns
        assert self.TARGET_COL in forecast_df.columns
        assert not forecast_df[self.TARGET_COL].isna().all()

    def test_data_has_required_columns(self, data_df: pd.DataFrame):
        """Test that the dataset has all required columns."""
        required_columns = [
            self.TIMESTAMP_COL,
            self.GROUP_COL,
            self.TARGET_COL,
        ] + self.METADATA_COLS

        for col in required_columns:
            assert col in data_df.columns, f"Missing column: {col}"

    def test_data_has_expected_items(
        self, data_df: pd.DataFrame, available_items: list[str]
    ):
        """Test that the dataset contains the expected items."""
        expected_items = ["soup", "tea", "bread", "ice_cream"]

        for item in expected_items:
            assert item in available_items, f"Missing item: {item}"

    def test_data_split_produces_valid_sets(self, data_df: pd.DataFrame):
        """Test that data splitting produces non-empty history and target sets."""
        for item_name in ["soup", "tea", "bread", "ice_cream"]:
            history_df, target_df = self._split_data(data_df, item_name)

            assert len(history_df) > 0, f"Empty history for {item_name}"
            assert len(target_df) > 0, f"Empty target for {item_name}"
            assert (
                history_df[self.TIMESTAMP_COL].max()
                < target_df[self.TIMESTAMP_COL].min()
            )
