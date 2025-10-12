import polars as pl
from loguru import logger
from polars.exceptions import ColumnNotFoundError

from src.checks.base import QualityCheck
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
            List of CheckResult objects, one per column checked

        Raises:
            ColumnNotFoundError: If specified columns don't exist in the DataFrame
        """
        unique_check_config = self.config.get("unique_check", {})
        columns_to_check: list[str] = unique_check_config.get("columns", [])
        exclude_nulls: bool = unique_check_config.get("exclude_nulls", True)

        if not columns_to_check:
            logger.warning("No columns provided to examine uniqueness")
            return []

        try:
            df_filtered_columns: pl.DataFrame = df.select(columns_to_check).collect()
        except ColumnNotFoundError:
            columns_non_exist = [
                col
                for col in columns_to_check
                if col not in df.collect_schema().names()
            ]
            logger.error(f"Columns {columns_non_exist} not found")
            return []

        check_result: list[CheckResult] = [
            self._check_column_uniqueness(df_filtered_columns, col, exclude_nulls)
            for col in columns_to_check
        ]

        return check_result

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
