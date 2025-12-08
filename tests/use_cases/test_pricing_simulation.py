"""
Pricing simulation use case test.

This test demonstrates using the Synthefy API to simulate different pricing
scenarios and forecast their impact on sales.
"""

import os

import numpy as np
import pandas as pd
import pytest
from synthefy.api_client import SynthefyAPIClient

# Skip all tests in this module if SYNTHEFY_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("SYNTHEFY_API_KEY"),
    reason="SYNTHEFY_API_KEY environment variable not set",
)


class TestPricingSimulation:
    """Test class for pricing simulation use case."""

    HISTORY_URL = "https://drive.google.com/uc?export=download&id=1FtLW17XE1NHcV1bF8mLW_2WHVKzrHts_"
    FUTURE_URL = "https://drive.google.com/uc?export=download&id=1l2zG7GNTcdDm_HzKhhwii3GTI0Z99a9h"

    @pytest.fixture
    def history_df(self) -> pd.DataFrame:
        """Load historical sales data."""
        return pd.read_csv(self.HISTORY_URL)

    @pytest.fixture
    def future_df(self) -> pd.DataFrame:
        """Load future period data template."""
        return pd.read_csv(self.FUTURE_URL)

    @pytest.fixture
    def price_simulation_range(self, history_df: pd.DataFrame) -> np.ndarray:
        """Create price range from 85% to 115% of base price (11 price points)."""
        base_price = history_df["unit_price"].mean()
        return np.linspace(base_price * 0.85, base_price * 1.15, 11)

    @pytest.fixture
    def target_dfs(
        self, future_df: pd.DataFrame, price_simulation_range: np.ndarray
    ) -> list[pd.DataFrame]:
        """Create target DataFrames with different price scenarios."""
        target_dfs = []
        for price in price_simulation_range:
            modified_future = future_df.copy()
            modified_future["unit_price"] = price
            target_dfs.append(modified_future)
        return target_dfs

    def test_pricing_simulation_forecasts(
        self,
        history_df: pd.DataFrame,
        target_dfs: list[pd.DataFrame],
        price_simulation_range: np.ndarray,
    ):
        """Test that pricing simulation returns forecasts for all price scenarios."""
        with SynthefyAPIClient() as api_client:
            results = api_client.forecast_dfs(
                history_dfs=[history_df] * len(price_simulation_range),
                target_dfs=target_dfs,
                target_col="sales",
                timestamp_col="date",
                metadata_cols=["unit_price"],
                leak_cols=["unit_price"],
                model="Migas-1.0",
            )

        # Verify we got results for all price scenarios
        assert len(results) == len(price_simulation_range)

        # Verify each result is a valid DataFrame with expected columns
        for i, result_df in enumerate(results):
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) > 0, f"Result {i} is empty"
            assert "timestamps" in result_df.columns
            assert "sales" in result_df.columns

    def test_price_range_calculation(
        self, history_df: pd.DataFrame, price_simulation_range: np.ndarray
    ):
        """Test that price range is calculated correctly."""
        base_price = history_df["unit_price"].mean()

        assert len(price_simulation_range) == 11
        assert price_simulation_range[0] == pytest.approx(base_price * 0.85)
        assert price_simulation_range[-1] == pytest.approx(base_price * 1.15)

    def test_target_dfs_have_correct_prices(
        self, target_dfs: list[pd.DataFrame], price_simulation_range: np.ndarray
    ):
        """Test that each target DataFrame has the correct simulated price."""
        assert len(target_dfs) == len(price_simulation_range)

        for target_df, expected_price in zip(
            target_dfs, price_simulation_range
        ):
            actual_prices = target_df["unit_price"].unique()
            assert len(actual_prices) == 1
            assert actual_prices[0] == pytest.approx(expected_price)
