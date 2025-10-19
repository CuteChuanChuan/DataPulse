"""Tests for FreshnessCheck with pytest best practices."""

from datetime import datetime, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from src.checks.constants import CheckConfigField, CheckConfigKey
from src.checks.freshness import FreshnessCheck
from src.models.result import CheckRule


class TestFreshnessCheck:
    """Test suite for FreshnessCheck."""

    # Mock "now" time for consistent testing
    FAKE_NOW = datetime(2025, 10, 18, 23, 0, 0)

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pl.LazyFrame({"updated_at": [self.FAKE_NOW - timedelta(hours=12)]})

    @staticmethod
    def _assert_basic_check_result(result, expected_column: str):
        """Helper to assert basic check result properties."""
        assert result.check_rule == CheckRule.FRESHNESS_CHECK
        assert result.check_name == f"freshness_{expected_column}"
        assert result.detail.column == expected_column
        assert result.detail.max_age_hours is not None

    @pytest.mark.parametrize(
        "hours_ago,max_age_hours,expected_pass",
        [
            # Fresh data scenarios
            (1, 24, True),  # 1 hour old, max 24 hours → pass
            (12, 24, True),  # 12 hours old, max 24 hours → pass
            # Boundary conditions
            (24, 24, True),  # Exactly 24 hours old (boundary) → pass
            (24.001, 24, False),  # Just over 24 hours → fail
            # Stale data scenarios
            (25, 24, False),  # 25 hours old, max 24 hours → fail
            (48, 24, False),  # 48 hours old, max 24 hours → fail
        ],
    )
    def test_freshness_various_ages(self, hours_ago, max_age_hours, expected_pass):
        """Test freshness check with various data ages and thresholds."""
        timestamp = self.FAKE_NOW - timedelta(hours=hours_ago)
        df = pl.LazyFrame({"updated_at": [timestamp]})

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: max_age_hours,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed == expected_pass
        assert result[0].detail.age_hours is not None
        assert result[0].detail.latest_timestamp is not None
        self._assert_basic_check_result(result[0], "updated_at")

    def test_uses_most_recent_timestamp(self):
        """Test that check uses the MOST RECENT timestamp, not oldest."""
        # Multiple timestamps: oldest is 48h ago, newest is 6h ago
        recent = self.FAKE_NOW - timedelta(hours=6)
        old = self.FAKE_NOW - timedelta(hours=48)

        df = pl.LazyFrame(
            {"updated_at": [old, recent, old - timedelta(hours=10), recent]}
        )

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        # Should pass because the MOST RECENT is only 6h old
        assert result[0].is_passed is True
        assert result[0].detail.age_hours == pytest.approx(6, rel=0.1)
        assert result[0].detail.max_age_hours == 24


class TestFreshnessCheckEdgeCases:

    # Mock "now" time for consistent testing
    FAKE_NOW = datetime(2025, 10, 18, 23, 0, 0)

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pl.LazyFrame({"updated_at": [self.FAKE_NOW - timedelta(hours=12)]})

    def test_empty_dataframe_fails(self):
        """Test with empty DataFrame (should fail)."""
        df = pl.LazyFrame({"updated_at": []}, schema={"updated_at": pl.Datetime})

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is False
        assert result[0].detail.latest_timestamp is None
        assert result[0].detail.age_hours is None
        assert result[0].detail.max_age_hours == 24

    def test_all_null_values_fails(self):
        """Test with all NULL timestamps (should fail)."""
        df = pl.LazyFrame({"updated_at": [None, None, None]})

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is False
        assert result[0].detail.latest_timestamp is None
        assert result[0].detail.age_hours is None

    @pytest.mark.parametrize(
        "config,expected_result_count",
        [
            ({}, 0),  # Empty config
            ({CheckConfigKey.FRESHNESS_CHECK: {}}, 0),  # Missing column
            (
                {CheckConfigKey.FRESHNESS_CHECK: {CheckConfigField.COLUMN: ""}},
                0,
            ),  # Empty column name
            (
                {
                    CheckConfigKey.FRESHNESS_CHECK: {
                        CheckConfigField.COLUMN: "updated_at"
                    }
                },
                0,
            ),  # Missing max_age_hours
            (
                {
                    CheckConfigKey.FRESHNESS_CHECK: {
                        CheckConfigField.COLUMN: "updated_at",
                        CheckConfigField.MAX_AGE_HOURS: -1,
                    }
                },
                0,
            ),  # Negative max_age_hours
            (
                {
                    CheckConfigKey.FRESHNESS_CHECK: {
                        CheckConfigField.COLUMN: "updated_at",
                        CheckConfigField.MAX_AGE_HOURS: -100,
                    }
                },
                0,
            ),  # Large negative max_age_hours
        ],
    )
    def test_invalid_configs(self, sample_df, config, expected_result_count):
        """Test handling of invalid configurations."""
        check = FreshnessCheck(config)
        result = check.execute(sample_df)

        assert len(result) == expected_result_count

    def test_mixed_null_and_valid_timestamps(self):
        """Test with a mix of NULL and valid timestamps (should ignore NULLs)."""
        valid_timestamp = self.FAKE_NOW - timedelta(hours=5)
        df = pl.LazyFrame(
            {"updated_at": [None, valid_timestamp, None, valid_timestamp]}
        )

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        # Should pass using the valid timestamp (5 hours old)
        assert result[0].is_passed is True
        assert result[0].detail.age_hours == pytest.approx(5, rel=0.1)

    def test_future_timestamp_treated_as_fresh(self):
        """Test with timestamp in the future (clock skew scenario).

        Timestamps in the future should be treated as age=0 (very fresh).
        This handles clock skew between servers.
        """
        # Timestamp 2 hours in the future
        future_timestamp = self.FAKE_NOW + timedelta(hours=2)
        df = pl.LazyFrame({"updated_at": [future_timestamp]})

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "updated_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)

        # Future timestamps should be treated as age=0 (very fresh)
        assert result[0].is_passed is True
        assert result[0].detail.age_hours == 0

    def test_detail_structure(self):
        """Test that detail contains all expected fields and correct values."""
        timestamp = self.FAKE_NOW - timedelta(hours=12)
        df = pl.LazyFrame({"created_at": [timestamp]})

        config = {
            CheckConfigKey.FRESHNESS_CHECK: {
                CheckConfigField.COLUMN: "created_at",
                CheckConfigField.MAX_AGE_HOURS: 24,
            }
        }

        with patch("src.checks.freshness.datetime") as mock_datetime:
            mock_datetime.now.return_value = self.FAKE_NOW
            check = FreshnessCheck(config)
            result = check.execute(df)[0]

        detail = result.detail

        # Check freshness-specific fields
        assert detail.column == "created_at"
        assert detail.latest_timestamp is not None
        assert isinstance(detail.latest_timestamp, str)  # ISO format string
        assert detail.age_hours == pytest.approx(12, rel=0.1)
        assert detail.max_age_hours == 24

        # Verify other check fields are None (not applicable to freshness)
        assert detail.total_rows is None
        assert detail.null_count is None
        assert detail.duplicated_count is None
