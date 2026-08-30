"""Local relational feature processing with hosted Nori inference."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

import pandas as pd

from synthefy.relational.models import (
    ConnectionStatus,
    ConnectorType,
    CredentialReference,
    Database,
    DatabaseCreateRequest,
    DatabaseUrl,
    DirectTarget,
    PredictionRequest,
    RelationalOutputType,
    SchemaGraph,
    TargetDefinition,
    TemporalTarget,
)
from synthefy.relational.models import (
    PredictionOperation as WirePredictionOperation,
)
from synthefy.relational.models import (
    PredictionTask as WirePredictionTask,
)

if TYPE_CHECKING:
    from synthefy.relational._runtime import LocalRelationalRuntime
    from synthefy.relational._runtime_config import RelationalLimits

PredictionOperation = Literal["next", "average", "total", "minimum", "maximum", "count"]
PredictionTask = Literal["regression", "classification"]
_OPERATIONS = {operation.value for operation in WirePredictionOperation}


class SynthefyNoriRelClient:
    """Predict from a relational database without a separate Nori-Rel service.

    Database reads and FastDFS run in this Python process. Only the resulting
    context and query feature matrices are sent to hosted Nori.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "nori-30m",
        classification_model: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        limits: Optional[RelationalLimits] = None,
        _runtime: Optional[LocalRelationalRuntime] = None,
    ) -> None:
        api_key = api_key or os.getenv("SYNTHEFY_API_KEY")
        if not api_key:
            raise ValueError(
                "A Synthefy API key is required through api_key or SYNTHEFY_API_KEY"
            )
        if not model or not model.strip():
            raise ValueError("model must not be blank")
        if classification_model is not None and not classification_model.strip():
            raise ValueError("classification_model must not be blank")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.model = model
        self.classification_model = classification_model or model
        self.timeout = timeout
        self.max_retries = max_retries
        if _runtime is None:
            try:
                from synthefy.relational._runtime import LocalRelationalRuntime
            except ImportError as exc:
                raise ImportError(
                    'Nori-Rel requires `pip install "synthefy[relational]"`'
                ) from exc
            _runtime = LocalRelationalRuntime(
                api_key=api_key,
                model=model,
                classification_model=classification_model,
                timeout=timeout,
                max_retries=max_retries,
                limits=limits,
            )
        self._runtime = _runtime

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"classification_model={self.classification_model!r})"
        )

    def __enter__(self) -> SynthefyNoriRelClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release client resources.

        Version 1 opens database and model connections per operation, so there are
        no persistent resources to close.
        """

    @staticmethod
    def _build_target(
        target: str,
        target_time: Optional[str],
        operation: Optional[PredictionOperation],
        lookahead: Optional[str],
    ) -> TargetDefinition:
        temporal = (target_time, operation, lookahead)
        if all(value is None for value in temporal):
            return DirectTarget(column=target)
        if any(value is None for value in temporal):
            raise ValueError(
                "target_time, operation, and lookahead must be provided together"
            )
        target_time = cast(str, target_time)
        operation = cast(PredictionOperation, operation)
        lookahead = cast(str, lookahead)
        if operation not in _OPERATIONS:
            choices = ", ".join(sorted(_OPERATIONS))
            raise ValueError(f"operation must be one of: {choices}")
        return TemporalTarget(
            column=target,
            time_column=target_time,
            operation=operation,
            lookahead=lookahead,
        )

    def connect(
        self,
        *,
        name: str,
        database_url: Optional[str] = None,
        credential: Optional[CredentialReference] = None,
        connector: Union[ConnectorType, str] = ConnectorType.POSTGRESQL,
        schema_name: str = "public",
        tables: Optional[List[str]] = None,
        time_columns: Optional[dict] = None,
    ) -> Database:
        """Connect one database using a URL or an indirect credential reference."""

        if (database_url is None) == (credential is None):
            raise ValueError("provide exactly one of database_url or credential")
        if database_url is not None:
            if not database_url.strip():
                raise ValueError("database_url must not be blank")
            credential = DatabaseUrl(value=database_url)
        credential = cast(CredentialReference, credential)
        config = DatabaseCreateRequest(
            name=name,
            connector=connector,
            credential=credential,
            schema_name=schema_name,
            tables=tables,
            time_columns=time_columns or {},
        )
        return self._runtime.connect(config)

    def test_connection(self, source: Database) -> ConnectionStatus:
        return self._runtime.test_connection(source)

    def discover(self, source: Database) -> SchemaGraph:
        return self._runtime.discover(source)

    def predict(
        self,
        *,
        source: Database,
        entity: str,
        target: str,
        task: Union[WirePredictionTask, PredictionTask, str] = "regression",
        target_time: Optional[str] = None,
        operation: Optional[PredictionOperation] = None,
        lookahead: Optional[str] = None,
        as_of: Optional[datetime] = None,
        relationship_path: Optional[List[str]] = None,
        entity_ids: Optional[List[Any]] = None,
        output_type: Optional[Union[RelationalOutputType, str]] = None,
        quantiles: Optional[List[float]] = None,
        positive_class: Any = None,
        decision_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        request = PredictionRequest(
            database=source.id,
            entity_table=entity,
            task=task,
            target=self._build_target(target, target_time, operation, lookahead),
            as_of=as_of,
            relationship_path=relationship_path,
            entity_ids=entity_ids,
            output_type=output_type,
            quantiles=quantiles,
            positive_class=positive_class,
            decision_threshold=decision_threshold,
        )
        return self._runtime.predict(source, request)
