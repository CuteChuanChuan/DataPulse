"""Tests for RowCountCheck with pytest best practices."""

import polars as pl
import pytest

from src.checks.constants import CheckConfigField, CheckConfigKey
from src.checks.row_count import RowCountCheck
from src.models.result import CheckResult, CheckRule


class TestRowCountCheck:
    """Test suite for RowCountCheck."""

    @staticmethod
    def _assert_basic_check_result(result: CheckResult):
        """Helper to assert basic check result properties."""
        assert result.check_rule == CheckRule.ROWS_COUNT_CHECK
        assert result.check_name == "row_count"
        assert result.detail.total_rows is not None
        # Row count is table-level, not column-specific
        assert result.detail.column is None

    @pytest.mark.parametrize(
        "row_count,min_rows_expected,max_rows_expected,expected_pass",
        [
            (100, 0, 200, True),  # Within ranges
            (100, 100, 200, True),  # Equals to min_rows_expected
            (200, 101, 200, True),  # Equals to max_rows_expected
            (100, 101, 200, False),  # Smaller than the min_rows_expected
            (100, 0, 99, False),  # Larger than the max_rows_expected
            (0, 0, 100, True),  # Empty df
        ],
    )
    def test_row_count_various_patterns(
        self, row_count, min_rows_expected, max_rows_expected, expected_pass
    ):
        """Test row count check with various patterns."""
        df = pl.LazyFrame({"col": range(row_count)})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MIN_ROWS_EXPECTED: min_rows_expected,
                CheckConfigField.MAX_ROWS_EXPECTED: max_rows_expected,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed == expected_pass
        assert result[0].detail.total_rows == row_count
        self._assert_basic_check_result(result[0])

    def test_row_count_only_min_rows(self):
        """Test row count check with only min_rows_expected."""
        min_rows_expected = 90

        df = pl.LazyFrame({"col": range(100)})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MIN_ROWS_EXPECTED: min_rows_expected,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 100  # Actual row count
        assert result[0].detail.min_rows_expected == min_rows_expected
        assert result[0].detail.max_rows_expected is None
        self._assert_basic_check_result(result[0])

    def test_row_count_only_max_rows(self):
        """Test row count check with only max_rows_expected."""
        max_rows_expected = 200

        df = pl.LazyFrame({"col": range(100)})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MAX_ROWS_EXPECTED: max_rows_expected,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 100  # Actual row count
        assert result[0].detail.min_rows_expected == 0  # Default
        assert result[0].detail.max_rows_expected == max_rows_expected
        self._assert_basic_check_result(result[0])

    def test_row_count_no_constraints(self):
        """Test with no constraints (should always pass)."""
        df = pl.LazyFrame({"col": range(100)})
        check = RowCountCheck({CheckConfigKey.ROW_COUNT_CHECK: {}})
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 100
        assert result[0].detail.min_rows_expected == 0
        assert result[0].detail.max_rows_expected is None

    def test_empty_config(self):
        """Test with completely empty config (should always pass)."""
        df = pl.LazyFrame({"col": range(50)})
        check = RowCountCheck({})
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 50

    def test_row_count_includes_rows_with_nulls(self):
        """Test that rows with NULL values are counted.

        This is crucial: row count is table-level, not affected by NULLs.
        """
        df = pl.LazyFrame(
            {
                "id": [1, None, 3],
                "name": [None, "Bob", None],
                "email": [None, None, None],
            }
        )

        check = RowCountCheck(
            {
                CheckConfigKey.ROW_COUNT_CHECK: {
                    CheckConfigField.MIN_ROWS_EXPECTED: 3,
                    CheckConfigField.MAX_ROWS_EXPECTED: 3,
                }
            }
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 3  # 3 rows, even with NULLs

    def test_detail_structure(self):
        """Test detail structure."""
        min_rows_expected = 90
        max_rows_expected = 200

        df = pl.LazyFrame({"col": range(100)})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MIN_ROWS_EXPECTED: min_rows_expected,
                CheckConfigField.MAX_ROWS_EXPECTED: max_rows_expected,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)[0]
        detail = result.detail

        assert detail.total_rows == 100
        assert detail.min_rows_expected == min_rows_expected
        assert detail.max_rows_expected == max_rows_expected

        # Check that column-specific fields are None
        assert detail.column is None
        assert detail.null_count is None
        assert detail.non_null_count is None
        assert detail.duplicated_count is None
        assert detail.duplicated_samples is None
        assert detail.exclude_nulls is None


class TestRowCountCheckEdgeCases:
    """Edge case tests for RowCountCheck."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.LazyFrame({"col": []})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MIN_ROWS_EXPECTED: 0,
                CheckConfigField.MAX_ROWS_EXPECTED: 100,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 0
        assert result[0].detail.min_rows_expected == 0
        assert result[0].detail.max_rows_expected == 100

    def test_empty_dataframe_fails_min(self):
        """Test with empty DataFrame fails min_rows_expected."""
        df = pl.LazyFrame({"col": []})
        config = {
            CheckConfigKey.ROW_COUNT_CHECK: {
                CheckConfigField.MIN_ROWS_EXPECTED: 1,
                CheckConfigField.MAX_ROWS_EXPECTED: 100,
            }
        }
        check = RowCountCheck(config)
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is False
        assert result[0].detail.total_rows == 0
        assert result[0].detail.min_rows_expected == 1
        assert result[0].detail.max_rows_expected == 100

    def test_exact_min_boundary(self):
        """Test exact minimum boundary (inclusive)."""
        df = pl.LazyFrame({"col": range(100)})
        check = RowCountCheck(
            {CheckConfigKey.ROW_COUNT_CHECK: {CheckConfigField.MIN_ROWS_EXPECTED: 100}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 100

    def test_exact_max_boundary(self):
        """Test exact maximum boundary (inclusive)."""
        df = pl.LazyFrame({"col": range(100)})
        check = RowCountCheck(
            {CheckConfigKey.ROW_COUNT_CHECK: {CheckConfigField.MAX_ROWS_EXPECTED: 100}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 100

    def test_one_below_min(self):
        """Test one row below minimum boundary."""
        df = pl.LazyFrame({"col": range(99)})
        check = RowCountCheck(
            {CheckConfigKey.ROW_COUNT_CHECK: {CheckConfigField.MIN_ROWS_EXPECTED: 100}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is False
        assert result.detail.total_rows == 99
        assert result.detail.min_rows_expected == 100

    def test_one_above_max(self):
        """Test one row above maximum boundary."""
        df = pl.LazyFrame({"col": range(101)})
        check = RowCountCheck(
            {CheckConfigKey.ROW_COUNT_CHECK: {CheckConfigField.MAX_ROWS_EXPECTED: 100}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is False
        assert result.detail.total_rows == 101
        assert result.detail.max_rows_expected == 100

    def test_multiple_columns_same_count(self):
        """Test that row count is same regardless of number of columns."""
        df = pl.LazyFrame({"col1": range(50), "col2": range(50), "col3": range(50)})
        check = RowCountCheck(
            {
                CheckConfigKey.ROW_COUNT_CHECK: {
                    CheckConfigField.MIN_ROWS_EXPECTED: 50,
                    CheckConfigField.MAX_ROWS_EXPECTED: 50,
                }
            }
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 50

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pl.LazyFrame({"col": [1]})
        check = RowCountCheck(
            {
                CheckConfigKey.ROW_COUNT_CHECK: {
                    CheckConfigField.MIN_ROWS_EXPECTED: 1,
                    CheckConfigField.MAX_ROWS_EXPECTED: 1,
                }
            }
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 1
