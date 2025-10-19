import polars as pl
from loguru import logger

from src.checks.base import QualityCheck
from src.checks.constants import CheckConfigField, CheckConfigKey
from src.models import CheckDetail, CheckResult, CheckRule


class RowCountCheck(QualityCheck):
    """Check if DataFrame row count is within expected range.

    This check validates that the total number of rows in the DataFrame
    falls within specified minimum and maximum bounds. This is a table-level
    check that counts all rows regardless of NULL values in any columns.

    Configuration:
        The config dict must contain a 'row_count_check' key with the
        following structure::

            {
                "row_count_check": {
                    "min": 1000,
                    "max": 1000000
                }
            }

        Fields:
            min (int): Optional. Minimum expected row count (default: 0)

            max (int): Optional. Maximum expected row count (default: None, no limit)

    Examples:
        >>> df = pl.LazyFrame({"id": range(5000), "name": ["User"] * 5000})
        >>> check = RowCountCheck({"row_count_check": {"min": 1000, "max": 10000}})
        >>> result = check.execute(df)
        >>> assert result[0].is_passed

        >>> # Only minimum
        >>> check = RowCountCheck({"row_count_check": {"min": 100}})
        >>> result = check.execute(df)
        >>> assert result[0].is_passed

        >>> # Only maximum
        >>> check = RowCountCheck({"row_count_check": {"max": 10000}})
        >>> result = check.execute(df)
        >>> assert result[0].is_passed

    Note:
        - Returns a single CheckResult (not per-column)
        - If both min and max are omitted, check will always pass
        - Rows with NULL values are counted
        - Setting max to None means no upper limit
        - This is a table-level check, not column-specific
    """

    def get_rule_type(self) -> CheckRule:
        return CheckRule.ROWS_COUNT_CHECK

    def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
        """Execute row count check on the DataFrame.

        Args:
            df: Polars LazyFrame to check

        Returns:
            List containing a single CheckResult with row count validation,
            or list with single error CheckResult if configuration is invalid
        """
        config = self.config.get(CheckConfigKey.ROW_COUNT_CHECK, {})
        min_rows_expected = config.get(CheckConfigField.MIN_ROWS_EXPECTED, 0)
        max_rows_expected = config.get(CheckConfigField.MAX_ROWS_EXPECTED, None)

        # Validate min_rows_expected
        if min_rows_expected < 0:
            error_msg = f"Invalid min_rows_expected ({min_rows_expected}): must be >= 0"
            logger.error(error_msg)
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="row_count_config_error",
                    is_passed=False,
                    detail=CheckDetail(
                        min_rows_expected=min_rows_expected,
                        error_message=error_msg,
                    ),
                )
            ]

        # Validate max_rows_expected
        if max_rows_expected is not None and max_rows_expected < 0:
            error_msg = f"Invalid max_rows_expected ({max_rows_expected}): must be >= 0"
            logger.error(error_msg)
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="row_count_config_error",
                    is_passed=False,
                    detail=CheckDetail(
                        max_rows_expected=max_rows_expected,
                        error_message=error_msg,
                    ),
                )
            ]

        # Validate min <= max
        if max_rows_expected is not None and min_rows_expected > max_rows_expected:
            error_msg = (
                f"Invalid range: min_rows_expected ({min_rows_expected}) > "
                f"max_rows_expected ({max_rows_expected})"
            )
            logger.error(error_msg)
            return [
                CheckResult(
                    check_rule=self.get_rule_type(),
                    check_name="row_count_config_error",
                    is_passed=False,
                    detail=CheckDetail(
                        min_rows_expected=min_rows_expected,
                        max_rows_expected=max_rows_expected,
                        error_message=error_msg,
                    ),
                )
            ]

        total_rows = df.collect().height
        is_passed = min_rows_expected <= total_rows
        if max_rows_expected is not None:
            is_passed = is_passed and total_rows <= max_rows_expected

        return [
            CheckResult(
                check_rule=self.get_rule_type(),
                check_name="row_count",
                is_passed=is_passed,
                detail=CheckDetail(
                    total_rows=total_rows,
                    min_rows_expected=min_rows_expected,
                    max_rows_expected=max_rows_expected,
                ),
            )
        ]
