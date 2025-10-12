import polars as pl

from src.checks.non_null import NonNullCheck
from src.models.result import CheckRule


def test_non_null_check_passed():
    df = pl.LazyFrame({"col1": [1, 2, 3]})
    columns_to_check_non_null: list[str] = ["col1"]
    check = NonNullCheck({"non_null_check": {"columns": columns_to_check_non_null}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_non_null)
    assert all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.NOT_NULL_CHECK
    assert result[0].check_name == "non_null_col1"
    assert result[0].is_passed


def test_non_null_check_failed():
    df = pl.LazyFrame({"col1": [1, 2, None]})
    columns_to_check_non_null: list[str] = ["col1"]
    check = NonNullCheck({"non_null_check": {"columns": columns_to_check_non_null}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_non_null)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.NOT_NULL_CHECK
    assert result[0].check_name == "non_null_col1"
    assert not result[0].is_passed


def test_non_null_check_mix_result():
    df = pl.LazyFrame({"col1": [1, 2, None], "col2": [4, 5, 6]})
    columns_to_check_non_null: list[str] = ["col1", "col2"]
    check = NonNullCheck({"non_null_check": {"columns": columns_to_check_non_null}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_non_null)
    assert not all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.NOT_NULL_CHECK
    assert result[0].check_name == "non_null_col1"
    assert not result[0].is_passed
    assert result[1].check_rule == CheckRule.NOT_NULL_CHECK
    assert result[1].check_name == "non_null_col2"
    assert result[1].is_passed


def test_non_null_check_empty_df():
    df = pl.LazyFrame({"col1": []})
    columns_to_check_non_null: list[str] = ["col1"]
    check = NonNullCheck({"non_null_check": {"columns": columns_to_check_non_null}})
    result = check.execute(df)

    assert len(result) == len(columns_to_check_non_null)
    assert all(r.is_passed for r in result)
    assert result[0].check_rule == CheckRule.NOT_NULL_CHECK
    assert result[0].check_name == "non_null_col1"
    assert result[0].is_passed


def test_non_null_check_all_nulls():
    df = pl.LazyFrame({"col1": [None, None, None]})
    check = NonNullCheck({"non_null_check": {"columns": ["col1"]}})
    result = check.execute(df)

    assert len(result) == 1
    assert not result[0].is_passed
    assert result[0].detail.null_count == 3
    assert result[0].detail.non_null_count == 0
    assert result[0].detail.total_rows == 3


def test_non_null_check_missing_config():
    df = pl.LazyFrame({"col1": [1, 2, 3]})
    check = NonNullCheck({})
    result = check.execute(df)

    assert result == []


def test_non_null_check_invalid_column():
    df = pl.LazyFrame({"col1": [1, 2, 3]})
    check = NonNullCheck({"non_null_check": {"columns": ["col_not_exist"]}})
    result = check.execute(df)

    assert result == []


def test_non_null_check_detail():
    df = pl.LazyFrame({"col1": [1, None, 3, None, 5]})
    check = NonNullCheck({"non_null_check": {"columns": ["col1"]}})
    result = check.execute(df)

    assert result[0].detail.column == "col1"
    assert result[0].detail.total_rows == 5
    assert result[0].detail.null_count == 2
    assert result[0].detail.non_null_count == 3
    assert not result[0].is_passed
