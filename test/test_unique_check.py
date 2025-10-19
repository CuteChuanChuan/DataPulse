"""Tests for UniqueCheck with pytest best practices."""

import polars as pl
import pytest

from src.checks.constants import CheckConfigField, CheckConfigKey
from src.checks.unique import UniqueCheck
from src.models.result import CheckResult, CheckRule


class TestUniqueCheck:
    """Test suite for UniqueCheck."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with various uniqueness patterns."""
        return pl.LazyFrame(
            {
                "unique_int": [1, 2, 3],
                "unique_str": ["a", "b", "c"],
                "dup_int": [1, 2, 1],
                "all_dup": [5, 5, 5],
                "with_null": [1, None, None],
            }
        )

    @staticmethod
    def _assert_basic_check_result(result: CheckResult, expected_name: str):
        """Helper to assert basic check result properties."""
        assert result.check_rule == CheckRule.UNIQUE_CHECK
        assert result.check_name == expected_name
        assert result.detail.column in expected_name

    @pytest.mark.parametrize(
        "column,exclude_nulls,expected_pass",
        [
            ("unique_int", True, True),  # All unique values
            ("unique_str", True, True),  # All unique strings
            ("dup_int", True, False),  # Has duplicates
            ("all_dup", True, False),  # All same value
            ("with_null", True, True),  # Multiple NULLs but excluded
            ("with_null", False, False),  # Multiple NULLs counted as duplicates
        ],
    )
    def test_unique_various_columns(
        self, sample_df, column, exclude_nulls, expected_pass
    ):
        """Test unique check on various columns with different null handling."""
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: [column],
                    CheckConfigField.EXCLUDE_NULLS: exclude_nulls,
                }
            }
        )
        result = check.execute(sample_df)

        assert len(result) == 1
        assert result[0].is_passed == expected_pass
        assert result[0].detail.exclude_nulls == exclude_nulls
        self._assert_basic_check_result(result[0], f"unique_{column}")

    def test_unique_multiple_columns(self, sample_df):
        """Test checking multiple columns at once."""
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["unique_int", "dup_int", "all_dup"]
                }
            }
        )
        results = check.execute(sample_df)

        assert len(results) == 3
        assert results[0].is_passed is True  # unique_int: all unique
        assert results[1].is_passed is False  # dup_int: has duplicates
        assert results[2].is_passed is False  # all_dup: all duplicates

    @pytest.mark.parametrize(
        "data,exclude_nulls,expected_pass,expected_dup_count",
        [
            ([1, 2, 3], True, True, 0),  # All unique
            ([1, 2, 1], True, False, 2),  # Has duplicates
            ([1, 1, 1], True, False, 3),  # All same
            ([1, None, None], True, True, 0),  # NULLs excluded
            ([1, None, None], False, False, 2),  # NULLs counted
            ([None, None, None], True, True, 0),  # All NULL, excluded
            ([None, None, None], False, False, 3),  # All NULL, counted
            ([], True, True, 0),  # Empty
        ],
    )
    def test_uniqueness_patterns(
        self, data, exclude_nulls, expected_pass, expected_dup_count
    ):
        """Test various uniqueness patterns with different null handling."""
        df = pl.LazyFrame({"col": data})
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["col"],
                    CheckConfigField.EXCLUDE_NULLS: exclude_nulls,
                }
            }
        )
        result = check.execute(df)[0]

        assert result.is_passed == expected_pass
        assert result.detail.duplicated_count == expected_dup_count
        assert result.detail.exclude_nulls == exclude_nulls

    def test_default_exclude_nulls_is_true(self):
        """Test that exclude_nulls defaults to True."""
        df = pl.LazyFrame({"col": [1, None, None]})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is True  # NULLs excluded by default
        assert result.detail.exclude_nulls is True
        assert result.detail.null_count == 2

    @pytest.mark.parametrize(
        "config,expected_error_substring",
        [
            ({}, "No columns provided"),  # Empty config
            (
                {CheckConfigKey.UNIQUE_CHECK: {}}, 
                "No columns provided"
            ),  # Missing columns
            (
                {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: []}},
                "No columns provided",
            ),  # Empty columns list
        ],
    )
    def test_invalid_configs(self, sample_df, config, expected_error_substring):
        """Test handling of invalid configurations.

        Invalid configs now return a failed CheckResult with error_message
        instead of an empty list.
        """
        check = UniqueCheck(config)
        result = check.execute(sample_df)

        assert len(result) == 1
        assert result[0].is_passed is False
        assert result[0].detail.error_message is not None
        assert expected_error_substring in result[0].detail.error_message

    def test_nonexistent_column(self, sample_df):
        """Test checking a column that doesn't exist.

        Now returns a failed CheckResult with error message instead of empty list.
        """
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["nonexistent"]}}
        )
        result = check.execute(sample_df)

        assert len(result) == 1
        assert result[0].is_passed is False
        assert result[0].detail.error_message is not None
        assert "not found" in result[0].detail.error_message

    def test_detail_structure_with_duplicates(self):
        """Test CheckDetail structure when duplicates exist."""
        df = pl.LazyFrame({"col": [1, 2, 2, 3, 3, 3]})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        detail = result.detail
        assert detail.column == "col"
        assert detail.total_rows == 6
        assert detail.null_count == 0
        assert detail.duplicated_count == 5  # [2, 2, 3, 3, 3]
        assert detail.duplicated_samples is not None
        assert "2" in detail.duplicated_samples
        assert "3" in detail.duplicated_samples
        assert detail.exclude_nulls is True

    def test_detail_structure_all_unique(self):
        """Test CheckDetail structure when all values are unique."""
        df = pl.LazyFrame({"col": [1, 2, 3]})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        detail = result.detail
        assert detail.column == "col"
        assert detail.total_rows == 3
        assert detail.null_count == 0
        assert detail.duplicated_count == 0
        assert detail.duplicated_samples is None  # No duplicates


class TestUniqueCheckNullHandling:
    """Focused tests on NULL handling behavior."""

    @pytest.mark.parametrize(
        "data,exclude_nulls,expected_total_rows,expected_null_count",
        [
            ([1, 2, None], True, 2, 1),  # Filter out NULL
            ([1, 2, None], False, 3, 0),  # Keep NULL (not in filtered)
            ([None, None], True, 0, 2),  # All NULL filtered out
            ([1, None, 2, None], True, 2, 2),  # Multiple NULLs filtered
        ],
    )
    def test_null_filtering_behavior(
        self, data, exclude_nulls, expected_total_rows, expected_null_count
    ):
        """Test that null filtering works correctly."""
        df = pl.LazyFrame({"col": data})
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["col"],
                    CheckConfigField.EXCLUDE_NULLS: exclude_nulls,
                }
            }
        )
        result = check.execute(df)[0]

        assert result.detail.total_rows == expected_total_rows
        assert result.detail.null_count == expected_null_count

    def test_nulls_with_duplicates_exclude_true(self):
        """Test duplicates exist even after excluding NULLs."""
        df = pl.LazyFrame({"col": [1, 2, 2, None, None]})
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["col"],
                    CheckConfigField.EXCLUDE_NULLS: True,
                }
            }
        )
        result = check.execute(df)[0]

        assert not result.is_passed
        assert result.detail.null_count == 2
        assert result.detail.total_rows == 3  # [1, 2, 2]
        assert result.detail.duplicated_count == 2  # [2, 2]

    def test_nulls_with_duplicates_exclude_false(self):
        """Test NULLs counted as duplicates when exclude_nulls=False."""
        df = pl.LazyFrame({"col": [1, 2, None, None]})
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["col"],
                    CheckConfigField.EXCLUDE_NULLS: False,
                }
            }
        )
        result = check.execute(df)[0]

        assert not result.is_passed
        assert result.detail.null_count == 0  # Not filtered
        assert result.detail.total_rows == 4
        assert result.detail.duplicated_count == 2  # Two NULLs


class TestUniqueCheckEdgeCases:
    """Edge case tests for UniqueCheck."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.LazyFrame({"col": []})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)

        assert len(result) == 1
        assert result[0].is_passed is True
        assert result[0].detail.total_rows == 0
        assert result[0].detail.duplicated_count == 0

    def test_single_value(self):
        """Test with single value (should be unique)."""
        df = pl.LazyFrame({"col": [1]})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        assert result.is_passed is True
        assert result.detail.total_rows == 1
        assert result.detail.duplicated_count == 0

    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 1]),
            (pl.Float64, [1.0, 2.0, 1.0]),
            (pl.Utf8, ["a", "b", "a"]),
            (pl.Boolean, [True, False, True]),
        ],
    )
    def test_duplicates_in_different_types(self, dtype, values):
        """Test duplicate detection works across different data types."""
        df = pl.LazyFrame({"col": values}).cast({"col": dtype})
        check = UniqueCheck(
            {CheckConfigKey.UNIQUE_CHECK: {CheckConfigField.COLUMNS: ["col"]}}
        )
        result = check.execute(df)[0]

        assert not result.is_passed
        assert result.detail.duplicated_count == 2

    def test_mixed_types_with_nulls(self):
        """Test with mixed numeric values and NULLs."""
        df = pl.LazyFrame({"col": [1.0, 2.0, None, 2.0, None]})
        check = UniqueCheck(
            {
                CheckConfigKey.UNIQUE_CHECK: {
                    CheckConfigField.COLUMNS: ["col"],
                    CheckConfigField.EXCLUDE_NULLS: True,
                }
            }
        )
        result = check.execute(df)[0]

        assert not result.is_passed  # [1.0, 2.0, 2.0] has duplicate
        assert result.detail.null_count == 2
        assert result.detail.duplicated_count == 2  # [2.0, 2.0]
