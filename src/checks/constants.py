"""Constants for quality check configurations."""

from enum import Enum


class CheckConfigKey(str, Enum):
    """Configuration keys for different check types."""

    UNIQUE_CHECK = "unique_check"
    NOT_NULL_CHECK = "non_null_check"
    ROW_COUNT_CHECK = "row_count_check"
    FRESHNESS_CHECK = "freshness_check"
    CUSTOM_CHECK = "custom_check"


class CheckConfigField(str, Enum):
    """Common configuration field names."""

    COLUMN = "column"
    COLUMNS = "columns"
    EXCLUDE_NULLS = "exclude_nulls"
    MIN_ROWS_EXPECTED = "min"
    MAX_ROWS_EXPECTED = "max"
    MAX_AGE_HOURS = "max_age_hours"
