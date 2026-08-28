"""Wire models for the Nori-Rel enterprise control plane."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_QUALIFIED_COLUMN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_$]*\.[A-Za-z_][A-Za-z0-9_$]*$"
)
_DURATION = re.compile(
    r"^(?P<value>[1-9][0-9]*)\s+"
    r"(?P<unit>second|seconds|minute|minutes|hour|hours|day|days|week|weeks)$",
    re.IGNORECASE,
)
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class WireModel(BaseModel):
    """Strict base model shared by every Nori-Rel wire object."""

    model_config = ConfigDict(extra="forbid")


class ConnectorType(str, Enum):
    POSTGRESQL = "postgresql"


class DatabaseStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


class PredictionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Aggregation(str, Enum):
    FIRST = "first"
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


class RelationalOutputType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    QUANTILES = "quantiles"


class AwsSecret(BaseModel):
    """Reference to a secret resolved by the connector agent's AWS identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["aws_secrets_manager"] = "aws_secrets_manager"
    secret_id: str = Field(min_length=1)
    region_name: Optional[str] = None


class EnvironmentCredential(BaseModel):
    """Reference to a connection URL stored in the connector agent environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["environment"] = "environment"
    variable: str = Field(min_length=1)

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        if not _ENVIRONMENT_VARIABLE.fullmatch(value):
            raise ValueError("variable must be a valid environment-variable name")
        return value


CredentialReference = Union[AwsSecret, EnvironmentCredential]


class DatabaseCreateRequest(WireModel):
    name: str = Field(min_length=1, max_length=128)
    connector: ConnectorType
    credential: CredentialReference
    schema_name: str = Field(default="public", min_length=1, max_length=128)
    tables: Optional[List[str]] = None
    time_columns: Dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "schema_name")
    @classmethod
    def strip_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tables")
    @classmethod
    def validate_tables(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = [table.strip() for table in value]
        if not cleaned or any(not table for table in cleaned):
            raise ValueError("tables must contain at least one non-blank table")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("tables must not contain duplicates")
        return cleaned

    @field_validator("time_columns")
    @classmethod
    def validate_time_columns(cls, value: Dict[str, str]) -> Dict[str, str]:
        if any(
            not _IDENTIFIER.fullmatch(table) or not _IDENTIFIER.fullmatch(column)
            for table, column in value.items()
        ):
            raise ValueError("time_columns must map table names to column names")
        return value

    @model_validator(mode="after")
    def validate_table_configuration(self) -> "DatabaseCreateRequest":
        if self.tables is not None:
            unknown = set(self.time_columns).difference(self.tables)
            if unknown:
                raise ValueError("time_columns contains a table outside tables")
        return self


class Database(WireModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    connector: ConnectorType
    schema_name: str
    status: DatabaseStatus
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class ConnectionStatus(WireModel):
    database_id: str
    status: DatabaseStatus
    latency_ms: Optional[float] = Field(default=None, ge=0)
    error: Optional[str] = None


class ColumnSchema(WireModel):
    name: str
    data_type: str
    nullable: bool


class ForeignKeySchema(WireModel):
    name: Optional[str] = None
    columns: List[str] = Field(min_length=1)
    referred_table: str
    referred_columns: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_arity(self) -> "ForeignKeySchema":
        if len(self.columns) != len(self.referred_columns):
            raise ValueError("foreign-key columns must have matching arity")
        return self


class TableSchema(WireModel):
    name: str
    columns: List[ColumnSchema]
    primary_key: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKeySchema] = Field(default_factory=list)
    time_column: Optional[str] = None


class ConnectorCapabilities(WireModel):
    consistent_snapshot: bool
    schemas: bool
    server_side_cursor: bool


class SchemaGraph(WireModel):
    database_id: str
    schema_name: str
    tables: List[TableSchema]
    capabilities: ConnectorCapabilities
    discovered_at: datetime


class PredictionRequest(WireModel):
    database: str = Field(min_length=1)
    entity_table: str = Field(min_length=1)
    target: str
    event_time: str
    aggregation: Aggregation
    horizon: str
    as_of: Optional[datetime] = None
    relationship_path: Optional[List[str]] = None
    entity_ids: Optional[List[Any]] = None
    output_type: RelationalOutputType = RelationalOutputType.MEDIAN
    quantiles: Optional[List[float]] = None

    @field_validator("target", "event_time")
    @classmethod
    def validate_qualified_column(cls, value: str) -> str:
        value = value.strip()
        if not _QUALIFIED_COLUMN.fullmatch(value):
            raise ValueError("must be a qualified column in 'table.column' form")
        return value

    @field_validator("database", "entity_table")
    @classmethod
    def strip_required_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("horizon")
    @classmethod
    def normalize_horizon(cls, value: str) -> str:
        match = _DURATION.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "horizon must be a positive integer followed by seconds, minutes, "
                "hours, days, or weeks"
            )
        amount = int(match.group("value"))
        unit = match.group("unit").lower()
        if amount == 1:
            unit = unit.removesuffix("s")
        elif not unit.endswith("s"):
            unit += "s"
        return f"{amount} {unit}"

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("as_of must include a timezone")
        return value

    @field_validator("entity_ids")
    @classmethod
    def validate_entity_ids(cls, value: Optional[List[Any]]) -> Optional[List[Any]]:
        if value is not None and not value:
            raise ValueError("entity_ids must not be empty")
        return value

    @model_validator(mode="after")
    def validate_prediction(self) -> "PredictionRequest":
        target_table, _ = self.target.split(".", 1)
        time_table, _ = self.event_time.split(".", 1)
        if target_table != time_table:
            raise ValueError(
                "version 1 requires target and event_time to belong to the same table"
            )
        if self.output_type == RelationalOutputType.QUANTILES:
            if not self.quantiles:
                raise ValueError("quantiles are required for output_type='quantiles'")
            if any(level <= 0 or level >= 1 for level in self.quantiles):
                raise ValueError("quantiles must lie strictly between 0 and 1")
        elif self.quantiles is not None:
            raise ValueError("quantiles are only valid with output_type='quantiles'")
        if self.relationship_path is not None and len(self.relationship_path) < 2:
            raise ValueError("relationship_path must contain at least two table names")
        if self.relationship_path is not None:
            if self.relationship_path[0] != self.entity_table:
                raise ValueError("relationship_path must start at entity_table")
            if self.relationship_path[-1] != target_table:
                raise ValueError("relationship_path must end at the target table")
        if self.quantiles is not None:
            if len(set(self.quantiles)) != len(self.quantiles):
                raise ValueError("quantiles must not contain duplicates")
            if self.quantiles != sorted(self.quantiles):
                raise ValueError("quantiles must be sorted")
        return self


class PredictionJobRecord(WireModel):
    id: str
    status: PredictionStatus
    progress: float = Field(ge=0, le=1)
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class PredictionRow(WireModel):
    entity_id: Any
    as_of: datetime
    prediction: Optional[float]
    quantiles: Optional[List[Optional[float]]] = None


class PredictionResult(WireModel):
    job_id: str
    entity_key: str
    output_type: RelationalOutputType
    rows: List[PredictionRow]
    taus: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
