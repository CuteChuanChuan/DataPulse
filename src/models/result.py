import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceType(str, Enum):
    POSTGRES = "postgres"
    ICEBERG = "iceberg"


class CheckRule(str, Enum):
    UNIQUE_CHECK = "unique_check"
    NOT_NULL_CHECK = "not_null_check"
    ROWS_COUNT_CHECK = "rows_count_check"
    FRESHNESS_CHECK = "freshness_check"
    SAMPLE_DATA_CHECK = "sample_data_check"
    CUSTOM_CHECK = "custom_check"


class CheckDetail(BaseModel):
    column: str | None = None
    total_rows: int | None = None
    null_count: int | None = None
    duplicated_count: int | None = None
    duplicated_samples: list[str] | None = None
    exclude_nulls: bool | None = None


class CheckResult(BaseModel):
    model_config = {"frozen": True}

    check_rule: CheckRule
    check_name: str
    is_passed: bool
    detail: CheckDetail


class CheckSummary(BaseModel):
    total_checks: int
    passed_checks: int
    failed_checks: int
    overall_passed_rate: float = Field(ge=0.0, le=1.0)


class CheckResponse(BaseModel):
    job_id: uuid.UUID
    source: str
    table: str
    status: CheckStatus
    summary: CheckSummary
    check_results: list[CheckResult]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
