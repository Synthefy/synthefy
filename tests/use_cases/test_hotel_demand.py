"""
Hotel demand forecasting use case test.

This test demonstrates using the Synthefy API to forecast hotel room bookings
with both univariate and multivariate approaches.
"""

import os

import numpy as np
import pandas as pd
import pytest
from synthefy import SynthefyAsyncAPIClient

# Skip all tests in this module if SYNTHEFY_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("SYNTHEFY_API_KEY"),
    reason="SYNTHEFY_API_KEY environment variable not set",
)


class TestHotelDemandForecasting:
    """Test class for hotel demand forecasting use case."""

    PRICING_URL = "https://drive.google.com/uc?export=download&id=1DsYn2tTR0W0bmGoz5vczUqJEcImAcyjG"
    EVENTS_URL = "https://drive.google.com/uc?export=download&id=1wcfQagVUz8PWYQeVPURuXxsnrOizLG0Q"
    CUTOFF_DATE = "2024-08-31"

    @pytest.fixture
    def raw_data_df(self) -> pd.DataFrame:
        """Load and prepare the raw pricing data."""
        df = pd.read_csv(self.PRICING_URL)
        df["timestamp"] = pd.to_datetime(df["date"])
        return df

    @pytest.fixture
    def events_df(self) -> pd.DataFrame:
        """Load and prepare the events data."""
        df = pd.read_json(self.EVENTS_URL)
        df["timestamp"] = pd.to_datetime(df["date"])
        df.drop(columns=["date"], inplace=True)
        return df

    @pytest.fixture
    def data_df(
        self, raw_data_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge pricing and events data."""
        df = raw_data_df.merge(events_df, on="timestamp", how="left")
        df["has_event"] = df["events_around_hotel"].notna().astype(int)
        return df

    @pytest.fixture
    def history_df(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Get historical data up to cutoff date."""
        cutoff_ts = pd.Timestamp(self.CUTOFF_DATE)
        return data_df[data_df["timestamp"] <= cutoff_ts].copy()

    @pytest.fixture
    def target_period_df(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Get target period data after cutoff date."""
        cutoff_ts = pd.Timestamp(self.CUTOFF_DATE)
        return data_df[data_df["timestamp"] > cutoff_ts].copy()

    @staticmethod
    def _prepare_univariate_data(
        history_df: pd.DataFrame, target_period_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        """Prepare data for univariate forecasting."""
        history_forecast_df = history_df[
            ["timestamp", "num_rooms_booked"]
        ].copy()
        target_df = pd.DataFrame(
            {
                "timestamp": target_period_df["timestamp"],
                "num_rooms_booked": np.nan,
            }
        )
        return history_forecast_df, target_df, [], []

    @staticmethod
    def _prepare_multivariate_data(
        history_df: pd.DataFrame, target_period_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        """Prepare data for multivariate forecasting with events and pricing."""
        metadata_cols = ["has_event", "price_per_room", "avg_competitor_price"]
        leak_cols = ["has_event"]

        history_forecast_df = history_df[
            ["timestamp", "num_rooms_booked"] + metadata_cols
        ].copy()

        target_df = pd.DataFrame(
            {
                "timestamp": target_period_df["timestamp"],
                "num_rooms_booked": np.nan,
                "has_event": target_period_df["has_event"],
                "price_per_room": np.nan,
                "avg_competitor_price": np.nan,
            }
        )

        return history_forecast_df, target_df, metadata_cols, leak_cols

    @pytest.mark.asyncio
    async def test_univariate_forecast(
        self, history_df: pd.DataFrame, target_period_df: pd.DataFrame
    ):
        """Test univariate hotel demand forecasting (rooms only)."""
        history_forecast_df, target_df, metadata_cols, leak_cols = (
            self._prepare_univariate_data(history_df, target_period_df)
        )

        async with SynthefyAsyncAPIClient() as client:
            forecast_dfs = await client.forecast_dfs(
                history_dfs=[history_forecast_df],
                target_dfs=[target_df],
                target_col="num_rooms_booked",
                timestamp_col="timestamp",
                metadata_cols=metadata_cols,
                leak_cols=leak_cols,
                model="Migas-1.0",
            )

        assert len(forecast_dfs) == 1
        forecast_df = forecast_dfs[0]

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(target_period_df)
        assert "timestamps" in forecast_df.columns
        assert "num_rooms_booked" in forecast_df.columns

    @pytest.mark.asyncio
    async def test_multivariate_forecast(
        self, history_df: pd.DataFrame, target_period_df: pd.DataFrame
    ):
        """Test multivariate hotel demand forecasting (with events and pricing)."""
        history_forecast_df, target_df, metadata_cols, leak_cols = (
            self._prepare_multivariate_data(history_df, target_period_df)
        )

        async with SynthefyAsyncAPIClient() as client:
            forecast_dfs = await client.forecast_dfs(
                history_dfs=[history_forecast_df],
                target_dfs=[target_df],
                target_col="num_rooms_booked",
                timestamp_col="timestamp",
                metadata_cols=metadata_cols,
                leak_cols=leak_cols,
                model="chronos2",
            )

        assert len(forecast_dfs) == 1
        forecast_df = forecast_dfs[0]

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(target_period_df)
        assert "timestamps" in forecast_df.columns
        assert "num_rooms_booked" in forecast_df.columns

    def test_data_split_is_valid(
        self, history_df: pd.DataFrame, target_period_df: pd.DataFrame
    ):
        """Test that data is correctly split at the cutoff date."""
        cutoff_ts = pd.Timestamp(self.CUTOFF_DATE)

        assert history_df["timestamp"].max() <= cutoff_ts
        assert target_period_df["timestamp"].min() > cutoff_ts
        assert len(history_df) > 0
        assert len(target_period_df) > 0

    def test_events_data_merged_correctly(self, data_df: pd.DataFrame):
        """Test that events data is properly merged with pricing data."""
        assert "has_event" in data_df.columns
        assert "events_around_hotel" in data_df.columns
        assert data_df["has_event"].isin([0, 1]).all()
