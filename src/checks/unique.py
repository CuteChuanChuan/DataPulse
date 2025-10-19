import polars as pl
from loguru import logger

from src.checks.base import QualityCheck
from src.checks.constants import CheckConfigField, CheckConfigKey
from src.models import CheckDetail, CheckResult, CheckRule


class UniqueCheck(QualityCheck):
    """Check if the values in specified columns are unique.

    This check validates that all values in the specified columns are unique,
    with optional handling of NULL values.

    Configuration:
        The config dict must contain a 'unique_check' key with the following structure::

            {
                "unique_check": {
                    "columns": ["col1", "col2"],
                    "exclude_nulls": True
                }
            }

        Fields:
            columns (list[str]): Required. List of columns to check for uniqueness

            exclude_nulls (bool): Optional. Exclude NULL values (default: True)

    Examples:
        >>> df = pl.LazyFrame({"id": [1, 2], "email": ["a@test.com", "b@test.com"]})
        >>> check = UniqueCheck({"unique_check": {"columns": ["id", "email"]}})
        >>> results = check.execute(df)
        >>> assert all(r.is_passed for r in results)

    Note:
        - When exclude_nulls=True (default), NULL values are not considered duplicates
        - When exclude_nulls=False, multiple NULLs are treated as duplicates
    """

    def get_rule_type(self) -> CheckRule:
        return CheckRule.UNIQUE_CHECK

    def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
        """Execute uniqueness check on the DataFrame.

        Args:
            df: Polars LazyFrame to check

        Returns:
            List of CheckResult objects, one per column checked,
            or list with single error CheckResult if configuration is invalid
        """
        check_config = self.config.get(CheckConfigKey.UNIQUE_CHECK, {})
        columns_to_check: list[str] = check_config.get(CheckConfigField.COLUMNS, [])
        exclude_nulls: bool = check_config.get(CheckConfigField.EXCLUDE_NULLS, True)

        # Validate columns provided
        if not columns_to_check:
            error_msg = f"No columns provided for {self.get_rule_type().value} check"
            logger.warning(error_msg)
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="unique_config_error",
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
                    check_name="unique_config_error",
                    is_passed=False,
                    detail=CheckDetail(error_message=error_msg),
                )
            ]

        return [
            self._check_column_uniqueness(df_collected, col, exclude_nulls)
            for col in columns_to_check
        ]

    def _check_column_uniqueness(
        self, df: pl.DataFrame, col: str, exclude_nulls: bool
    ) -> CheckResult:
        """Check uniqueness for a single column.

        Args:
            df: Polars DataFrame (already collected)
            col: Column name to check
            exclude_nulls: Whether to exclude NULL values from the check

        Returns:
            CheckResult with detailed information about duplicates
        """
        null_count = 0

        if exclude_nulls:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            df = df.filter(pl.col(col).is_not_null())

        total_rows = df.height
        is_duplicated_mask = df.select(pl.col(col).is_duplicated()).to_series()
        duplicated_count = is_duplicated_mask.sum()
        is_unique_all = duplicated_count == 0

        duplicated_samples = None
        if not is_unique_all:
            duplicates_df = df.filter(is_duplicated_mask)
            duplicated_samples = [
                str(x)
                for x in duplicates_df.select(col).unique().head().to_series().to_list()
            ]

        detail = CheckDetail(
            column=col,
            total_rows=total_rows,
            null_count=null_count,
            duplicated_count=duplicated_count,
            duplicated_samples=duplicated_samples,
            exclude_nulls=exclude_nulls,
        )

        return CheckResult(
            check_rule=self.get_rule_type(),
            check_name=f"unique_{col}",
            is_passed=is_unique_all,
            detail=detail,
        )
