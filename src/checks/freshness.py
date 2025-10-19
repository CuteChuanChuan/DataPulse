from datetime import datetime

import polars as pl
from loguru import logger

from src.checks import CheckConfigField, CheckConfigKey
from src.checks.base import QualityCheck
from src.models import CheckDetail, CheckResult, CheckRule


class FreshnessCheck(QualityCheck):
    """Check if data is fresh (recently updated).

    This check validates that the most recent timestamp in a specified column
    is within an acceptable age threshold. Useful for detecting stale data or
    broken data pipelines.

    Configuration:
        The config dict must contain a 'freshness_check' key with the
        following structure::

            {
                "freshness_check": {
                    "column": "updated_at",
                    "max_age_hours": 24
                }
            }

        Fields:
            column (str): Required. The timestamp/date column to check

            max_age_hours (float): Required. Maximum acceptable age in hours (must be >= 0)

    Examples:
        >>> from datetime import datetime, timedelta
        >>> now = datetime.now()
        >>> df = pl.LazyFrame({
        ...     "updated_at": [now - timedelta(hours=12), now - timedelta(hours=6)]
        ... })
        >>> check = FreshnessCheck(
...             {"freshness_check": {"column": "updated_at", "max_age_hours": 24}}
...         )
        >>> result = check.execute(df)
        >>> assert result[0].is_passed

    Note:
        - Returns a single CheckResult (not per-row)
        - Empty tables will fail the check
        - All NULL values will fail the check
        - Uses the most recent (max) timestamp for validation
        - Timestamps in the future are treated as age=0 (handles clock skew)
        - Rows with NULL values in the timestamp column are ignored
    """

    def get_rule_type(self) -> CheckRule:
        return CheckRule.FRESHNESS_CHECK

    def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
        """Execute freshness check on the DataFrame.

        Args:
            df: Polars LazyFrame to check

        Returns:
            List containing a single CheckResult with freshness validation,
            or empty list if configuration is invalid (logs warning/error)
        """
        # Extract and validate configuration
        config = self.config.get(CheckConfigKey.FRESHNESS_CHECK, {})
        if not config:
            logger.warning(
                f"No configuration provided for {self.get_rule_type().value}"
            )
            return []

        column_to_check: str | None = config.get(CheckConfigField.COLUMN, None)
        max_age_hours: int | None = config.get(CheckConfigField.MAX_AGE_HOURS, None)

        if column_to_check is None or not column_to_check:
            logger.warning(
                f"No column specified for {self.get_rule_type().value} check"
            )
            return []

        if max_age_hours is None:
            logger.warning(
                f"No max_age_hours specified for {self.get_rule_type().value} check"
            )
            return []

        if max_age_hours < 0:
            logger.error(
                f"Invalid max_age_hours ({max_age_hours}) for {self.get_rule_type().value} check: must be >= 0"
            )
            return []

        df_collected = self._validate_and_collect_columns(df, [column_to_check])
        if df_collected is None:
            return []

        return [
            self._check_column_freshness(df_collected, column_to_check, max_age_hours)
        ]

    def _check_column_freshness(
        self, df: pl.DataFrame, column_to_check: str, max_age_hours: int
    ) -> CheckResult:
        """Check freshness of a single column.

        Args:
            df: Polars DataFrame (already collected)
            column_to_check: Name of the timestamp column
            max_age_hours: Maximum acceptable age in hours

        Returns:
            CheckResult indicating whether the data is fresh enough
        """
        current_timestamp = datetime.now()
        latest_timestamp: datetime = df.select(pl.col(column_to_check).max()).item()

        if latest_timestamp is None:
            return CheckResult(
                check_rule=self.get_rule_type(),
                check_name=f"freshness_{column_to_check}",
                is_passed=False,
                detail=CheckDetail(
                    column=column_to_check,
                    latest_timestamp=None,
                    age_hours=None,
                    max_age_hours=max_age_hours,
                ),
            )

        age_hours = (current_timestamp - latest_timestamp).total_seconds() / 3600
        # Treat future timestamps as age 0 (handles clock skew)
        age_hours = max(0.0, age_hours)

        is_passed: bool = age_hours <= max_age_hours

        return CheckResult(
            check_rule=self.get_rule_type(),
            check_name=f"freshness_{column_to_check}",
            is_passed=is_passed,
            detail=CheckDetail(
                column=column_to_check,
                latest_timestamp=latest_timestamp.isoformat(),
                age_hours=round(age_hours, 2),
                max_age_hours=max_age_hours,
            ),
        )
