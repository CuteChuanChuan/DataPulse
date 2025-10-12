from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from loguru import logger
from polars.exceptions import ColumnNotFoundError

from src.models import CheckResult, CheckRule


class QualityCheck(ABC):
    """Abstract base class for all data quality checks.

    All quality check implementations must inherit from this class and
    implement the required abstract methods.

    Attributes:
        config (dict): Configuration dictionary for the check

    Examples:
        Implementing a custom check::

            class MyCheck(QualityCheck):
                def get_rule_type(self) -> CheckRule:
                    return CheckRule.CUSTOM_CHECK

                def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
                    # Implementation here
                    return [CheckResult(...)]
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the quality check.

        Args:
            config: Configuration dictionary containing check-specific settings
        """
        self.config = config

    @abstractmethod
    def get_rule_type(self) -> CheckRule:
        """Get the type of this quality check.

        Returns:
            CheckRule enum value identifying this check type
        """
        pass

    @abstractmethod
    def execute(self, df: pl.LazyFrame) -> list[CheckResult]:
        """Execute the quality check on the given DataFrame.

        Args:
            df: Polars LazyFrame to check

        Returns:
            List of CheckResult objects containing check outcomes

        Raises:
            Exception: Implementation-specific exceptions for check failures
        """
        pass

    def _validate_and_collect_columns(
        self, df: pl.LazyFrame, columns: list[str]
    ) -> pl.DataFrame | None:
        """Validate columns exist and collect the DataFrame.

        Args:
            df: LazyFrame to validate
            columns: List of column names to check

        Returns:
            Collected DataFrame with selected columns, or None if validation fails

        Raises:
            Logs error and returns None if columns don't exist
        """
        if not columns:
            logger.warning(f"No columns provided for {self.get_rule_type().value}")
            return None

        try:
            return df.select(columns).collect()
        except ColumnNotFoundError:
            columns_non_exist = [
                col for col in columns if col not in df.collect_schema().names()
            ]
            logger.error(f"Columns {columns_non_exist} not found in DataFrame")
            return None
