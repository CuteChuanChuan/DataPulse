"""Tests for NonNullCheck with pytest best practices."""

import polars as pl
import pytest

from src.checks.constants import CheckConfigField, CheckConfigKey
from src.checks.non_null import NonNullCheck
from src.models.result import CheckResult, CheckRule


class TestNonNullCheck:
    """Test suite for NonNullCheck."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["a@test.com", None, "c@test.com"],
                "age": [None, None, None],
            }
        )

    @staticmethod
    def _assert_basic_check_result(result: CheckResult, expected_name: str):
        """Helper to assert basic check result properties."""
        assert result.check_rule == CheckRule.NOT_NULL_CHECK
        assert result.check_name == expected_name
        assert result.detail.column in expected_name

    @pytest.mark.parametrize(
        "column,expected_pass",
        [
            ("id", True),  # All non-null
            ("name", True),  # All non-null
            ("email", False),  # Has one null
            ("age", False),  # All nulls
        ],
    )
    def test_non_null_various_columns(self, sample_df, column, expected_pass):
        """Test non-null check on various columns."""
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: [column]}}
        )
        result = check.execute(sample_df)

        assert len(result) == 1
        assert result[0].is_passed == expected_pass
        self._assert_basic_check_result(result[0], f"non_null_{column}")

    def test_non_null_multiple_columns(self, sample_df):
        """Test checking multiple columns at once."""
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["id", "email"]}}
        )
        results = check.execute(sample_df)

        assert len(results) == 2
        assert results[0].is_passed is True  # id: all non-null
        assert results[1].is_passed is False  # email: has null

    @pytest.mark.parametrize(
        "data,expected_null_count,expected_non_null_count",
        [
            ([1, 2, 3], 0, 3),  # All non-null
            ([1, None, 3], 1, 2),  # One null
            ([None, None, None], 3, 0),  # All nulls
            ([], 0, 0),  # Empty
        ],
    )
    def test_detail_counts(self, data, expected_null_count, expected_non_null_count):
        """Test that detail contains correct null/non-null counts."""
        df = pl.LazyFrame({"col": data})
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        assert result.detail.null_count == expected_null_count
        assert result.detail.non_null_count == expected_non_null_count
        assert result.detail.total_rows == len(data)

    def test_empty_dataframe(self):
        """Test with empty DataFrame (should pass)."""
        df = pl.LazyFrame({"col": []})
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 0

    @pytest.mark.parametrize(
        "config,expected_result_count",
        [
            ({}, 0),  # Empty config
            ({CheckConfigKey.NOT_NULL_CHECK: {}}, 0),  # Missing columns
            (
                {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: []}},
                0,
            ),  # Empty columns list
        ],
    )
    def test_invalid_configs(self, sample_df, config, expected_result_count):
        """Test handling of invalid configurations."""
        check = NonNullCheck(config)
        result = check.execute(sample_df)

        assert len(result) == expected_result_count

    def test_nonexistent_column(self, sample_df):
        """Test checking a column that doesn't exist."""
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["nonexistent"]}}
        )
        result = check.execute(sample_df)

        assert result == []

    def test_mixed_valid_invalid_columns(self, sample_df):
        """Test with mix of valid and invalid column names."""
        check = NonNullCheck(
            {
                CheckConfigKey.NOT_NULL_CHECK: {
                    CheckConfigField.COLUMNS: ["id", "nonexistent", "name"]
                }
            }
        )
        result = check.execute(sample_df)

        # Should return empty due to ColumnNotFoundError
        assert result == []

    def test_detail_structure(self, sample_df):
        """Test that CheckDetail has all required fields."""
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["email"]}}
        )
        result = check.execute(sample_df)[0]

        detail = result.detail
        assert detail.column == "email"
        assert detail.total_rows == 3
        assert detail.null_count == 1
        assert detail.non_null_count == 2

        # Check that other fields are None (not used in non-null check)
        assert detail.duplicated_count is None
        assert detail.duplicated_samples is None
        assert detail.exclude_nulls is None


class TestNonNullCheckEdgeCases:
    """Edge case tests for NonNullCheck."""

    @pytest.mark.parametrize(
        "dtype,null_value",
        [
            (pl.Int64, None),
            (pl.Float64, None),
            (pl.Utf8, None),
            (pl.Boolean, None),
        ],
    )
    def test_nulls_in_different_types(self, dtype, null_value):
        """Test null detection works across different data types."""
        df = pl.LazyFrame({"col": [null_value, null_value]}).cast({"col": dtype})
        check = NonNullCheck(
            {CheckConfigKey.NOT_NULL_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)

        assert len(result) == 1
        assert not result[0].is_passed
        assert result[0].detail.null_count == 2
