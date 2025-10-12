import polars as pl
import pytest

from src.checks.unique import UniqueCheck
from src.models.result import CheckRule


@pytest.fixture
def test_df() -> pl.LazyFrame:
    df = pl.LazyFrame(
        {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 7], "col4": [10, 10, 10]}
    )
    return df


def test_unique_check_passed(test_df):
    columns_to_check_unique: list[str] = ["col1", "col2"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(test_df)

    assert len(result) == len(columns_to_check_unique)
    assert all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[1].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert result[1].check_name == "unique_col2"
    assert result[0].is_passed
    assert result[1].is_passed
    assert result[0].detail.column == "col1"
    assert result[0].detail.total_rows == 3
    assert result[0].detail.duplicated_count == 0
    assert result[0].detail.duplicated_samples is None
    assert result[0].detail.exclude_nulls


def test_unique_check_failed(test_df):
    columns_to_check_unique: list[str] = ["col3", "col4"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(test_df)

    assert len(result) == len(columns_to_check_unique)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[1].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col3"
    assert result[1].check_name == "unique_col4"
    assert not result[0].is_passed
    assert not result[1].is_passed
    assert result[0].detail.column == "col3"
    assert result[0].detail.total_rows == 3
    assert result[0].detail.duplicated_count == 2
    assert result[0].detail.duplicated_samples == ["7"]
    assert result[1].detail.duplicated_count == 3


def test_unique_check_mix_result(test_df):
    columns_to_check_unique: list[str] = ["col1", "col2", "col3", "col4"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(test_df)

    assert len(result) == len(columns_to_check_unique)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[1].check_rule == CheckRule.UNIQUE_CHECK
    assert result[2].check_rule == CheckRule.UNIQUE_CHECK
    assert result[3].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert result[1].check_name == "unique_col2"
    assert result[2].check_name == "unique_col3"
    assert result[3].check_name == "unique_col4"
    assert result[0].is_passed
    assert result[1].is_passed
    assert not result[2].is_passed
    assert not result[3].is_passed


def test_unique_check_empty_df():
    df = pl.LazyFrame({"col1": []})
    columns_to_check_unique: list[str] = ["col1"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_unique)
    assert all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert result[0].is_passed


def test_unique_check_with_nulls_excluding_nulls():
    df = pl.LazyFrame({"col1": [1, 2, None, None]})
    columns_to_check_unique: list[str] = ["col1"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_unique)
    assert all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert result[0].is_passed

    assert result[0].detail.null_count == 2
    assert result[0].detail.total_rows == 2
    assert result[0].detail.exclude_nulls
    assert result[0].detail.duplicated_count == 0


def test_unique_check_with_nulls_not_excluding_nulls():
    df = pl.LazyFrame({"col1": [1, 2, None, None]})
    columns_to_check_unique: list[str] = ["col1"]
    check = UniqueCheck(
        {"unique_check": {"columns": columns_to_check_unique, "exclude_nulls": False}}
    )
    result = check.execute(df)

    assert len(result) == len(columns_to_check_unique)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert not result[0].is_passed
    assert result[0].detail.null_count == 0
    assert result[0].detail.total_rows == 4
    assert not result[0].detail.exclude_nulls
    assert result[0].detail.duplicated_count == 2


def test_unique_check_all_nulls():
    df = pl.LazyFrame({"col1": [None, None, None, None]})
    columns_to_check_unique: list[str] = ["col1"]

    check1 = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result1 = check1.execute(df)

    assert len(result1) == len(columns_to_check_unique)
    assert all(r.is_passed for r in result1)
    assert result1[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result1[0].check_name == "unique_col1"
    assert result1[0].is_passed
    assert result1[0].detail.null_count == 4
    assert result1[0].detail.total_rows == 0
    assert result1[0].detail.duplicated_count == 0

    check2 = UniqueCheck(
        {"unique_check": {"columns": columns_to_check_unique, "exclude_nulls": False}}
    )
    result2 = check2.execute(df)

    assert len(result2) == len(columns_to_check_unique)
    assert not all(r.is_passed for r in result2)
    assert result2[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result2[0].check_name == "unique_col1"
    assert not result2[0].is_passed
    assert result2[0].detail.total_rows == 4
    assert result2[0].detail.duplicated_count == 4


def test_unique_check_with_duplicate_and_null():
    df = pl.LazyFrame({"col1": [1, 2, 2, 3, None, None]})
    columns_to_check_unique: list[str] = ["col1"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_unique)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.UNIQUE_CHECK
    assert result[0].check_name == "unique_col1"
    assert not result[0].is_passed
    assert result[0].detail.null_count == 2
    assert result[0].detail.total_rows == 4
    assert result[0].detail.duplicated_count == 2
    assert result[0].detail.duplicated_samples == ["2"]


def test_unique_check_missing_config():
    df = pl.LazyFrame({"col1": [1, 2, None, None]})
    check = UniqueCheck({})
    result = check.execute(df)

    assert not result


def test_unique_check_invalid_column():
    df = pl.LazyFrame({"col1": [1, 2, 3], "col4": [4, 5, 6]})
    columns_to_check_unique: list[str] = ["col2", "col3", "col4"]
    check = UniqueCheck({"unique_check": {"columns": columns_to_check_unique}})
    result = check.execute(df)

    assert not result
