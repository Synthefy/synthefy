"""Nori-Rel database prediction client."""

from synthefy.relational._runtime_config import RelationalLimits
from synthefy.relational.client import (
    PredictionOperation,
    PredictionTask,
    SynthefyNoriRelClient,
)
from synthefy.relational.models import (
    AwsSecret,
    ConnectionStatus,
    ConnectorCapabilities,
    ConnectorType,
    CredentialReference,
    Database,
    DatabaseCreateRequest,
    DatabaseStatus,
    DatabaseUrl,
    DirectTarget,
    EnvironmentCredential,
    PredictionRequest,
    RelationalOutputType,
    SchemaGraph,
    TableSchema,
    TemporalTarget,
)

__all__ = [
    "AwsSecret",
    "ConnectionStatus",
    "ConnectorCapabilities",
    "ConnectorType",
    "CredentialReference",
    "Database",
    "DatabaseCreateRequest",
    "DatabaseStatus",
    "DatabaseUrl",
    "DirectTarget",
    "EnvironmentCredential",
    "PredictionOperation",
    "PredictionRequest",
    "PredictionTask",
    "RelationalLimits",
    "RelationalOutputType",
    "SchemaGraph",
    "SynthefyNoriRelClient",
    "TableSchema",
    "TemporalTarget",
]
