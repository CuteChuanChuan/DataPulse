import polars as pl
from loguru import logger

from src.checks.base import QualityCheck
from src.checks.constants import CheckConfigField, CheckConfigKey
from src.models import CheckDetail, CheckResult, CheckRule


class NonNullCheck(QualityCheck):
    """Check if specified columns contain no NULL values.

    This check validates that all values in the specified columns are non-null.

    Configuration:
        The config dict must contain a 'non_null_check' key::


            {
                "non_null_check": {
                    "columns": ["col1", "col2"]
                }
            }

        Fields:
            columns (list[str]): Required. List of columns to check for non-null values

    Examples:
        >>> df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        >>> check = NonNullCheck({"non_null_check": {"columns": ["id", "name"]}})
        >>> results = check.execute(df)
        >>> assert all(r.is_passed for r in results)

    Note:
        - Empty DataFrames pass the check (no NULL values present)
        - Returns detailed information about null count and non-null count
    """

    def get_rule_type(self) -> CheckRule:
        return CheckRule.NOT_NULL_CHECK

    def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
        """Execute non-null check on the DataFrame.

        Args:
            df: Polars LazyFrame to check

        Returns:
            List of CheckResult objects, one per column checked,
            or list with single error CheckResult if configuration is invalid
        """
        check_config = self.config.get(CheckConfigKey.NOT_NULL_CHECK, {})
        columns_to_check: list[str] = check_config.get(CheckConfigField.COLUMNS, [])

        # Validate columns provided
        if not columns_to_check:
            error_msg = f"No columns provided for {self.get_rule_type().value} check"
            logger.warning(error_msg)
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="non_null_config_error",
                    is_passed=False,
                    detail=CheckDetail(error_message=error_msg),
                )
            ]

        df_collected = self._validate_and_collect_columns(df, columns_to_check)
        if df_collected is None:
            # Column validation failed (detailed error already logged by base class)
            error_msg = "One or more columns not found in DataFrame"
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="non_null_config_error",
                    is_passed=False,
                    detail=CheckDetail(error_message=error_msg),
                )
            ]

        return [
            self._check_column_non_null(df_collected, col) for col in columns_to_check
        ]

    def _check_column_non_null(self, df: pl.DataFrame, col: str) -> CheckResult:
        """Check non-null constraint for a single column.

        Args:
            df: Polars DataFrame (already collected)
            col: Column name to check

        Returns:
            CheckResult with detailed information about null counts
        """
        null_count = df.select(pl.col(col).is_null().sum()).item()
        total_rows = df.height
        is_non_null_all = null_count == 0

        detail = CheckDetail(
            column=col,
            total_rows=total_rows,
            null_count=null_count,
            non_null_count=total_rows - null_count,
        )

        return CheckResult(
            check_rule=self.get_rule_type(),
            check_name=f"non_null_{col}",
            is_passed=is_non_null_all,
            detail=detail,
        )
