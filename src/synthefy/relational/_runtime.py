"""In-process relational execution with hosted Nori inference."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from synthefy.nori_client import SynthefyNoriClient
from synthefy.relational._features import FastDFSFeatureEngine
from synthefy.relational._outcomes import build_examples
from synthefy.relational._runtime_config import RelationalLimits
from synthefy.relational.connectors.base import RelationalConnector
from synthefy.relational.connectors.postgres import PostgreSQLConnector
from synthefy.relational.models import (
    ConnectionStatus,
    Database,
    DatabaseCreateRequest,
    DatabaseStatus,
    PredictionRequest,
    PredictionTask,
    RelationalOutputType,
    SchemaGraph,
    TemporalTarget,
)


class Predictor(Protocol):
    def model_for(self, task: PredictionTask) -> str: ...

    def predict(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        query: pd.DataFrame,
        request: PredictionRequest,
    ) -> tuple[list[float | None], list[list[float | None]] | None]: ...


class HostedNoriPredictor:
    """Send only flat feature matrices to the hosted Nori endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        classification_model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.regression_model = model
        self.classification_model = classification_model or model
        self.timeout = timeout
        self.max_retries = max_retries

    def model_for(self, task: PredictionTask) -> str:
        if task is PredictionTask.CLASSIFICATION:
            return self.classification_model
        return self.regression_model

    def predict(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        query: pd.DataFrame,
        request: PredictionRequest,
    ) -> tuple[list[float | None], list[list[float | None]] | None]:
        with SynthefyNoriClient(
            api_key=self.api_key,
            model=self.model_for(request.task),
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as client:
            if request.task is PredictionTask.CLASSIFICATION:
                result = client.predict(
                    train,
                    target,
                    query,
                    task="classification",
                    as_pandas=True,
                )
                if not isinstance(result, pd.Series):
                    raise TypeError("Nori returned an invalid classification response")
                probabilities = result.astype(float).to_numpy()
                if (
                    not np.isfinite(probabilities).all()
                    or ((probabilities < 0) | (probabilities > 1)).any()
                ):
                    raise ValueError("Nori returned invalid class probabilities")
                return probabilities.tolist(), None

            if request.output_type is None:
                raise RuntimeError("regression output type is missing")
            result = client.predict(
                train,
                target,
                query,
                output_type=request.output_type.value,
                quantiles=request.quantiles,
                as_pandas=True,
            )

        if request.output_type is RelationalOutputType.QUANTILES:
            if not isinstance(result, pd.DataFrame):
                raise TypeError("Nori returned an invalid quantile response")
            rows = result.astype(float).replace({np.nan: None}).values.tolist()
            levels = request.quantiles or []
            median_index = min(
                range(len(levels)), key=lambda index: abs(levels[index] - 0.5)
            )
            return [row[median_index] for row in rows], rows
        if not isinstance(result, pd.Series):
            raise TypeError("Nori returned an invalid point response")
        return result.astype(float).replace({np.nan: None}).tolist(), None


class LocalRelationalRuntime:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        classification_model: str | None,
        timeout: float,
        max_retries: int,
        limits: RelationalLimits | None = None,
        connector_factory: Callable[..., RelationalConnector] = PostgreSQLConnector,
        feature_engine: FastDFSFeatureEngine | None = None,
        predictor: Predictor | None = None,
    ) -> None:
        self.limits = limits or RelationalLimits()
        self.connector_factory = connector_factory
        self.feature_engine = feature_engine or FastDFSFeatureEngine(
            max_depth=2, max_features=self.limits.max_features
        )
        self.predictor = predictor or HostedNoriPredictor(
            api_key,
            model,
            classification_model=classification_model,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._sources: dict[str, DatabaseCreateRequest] = {}

    def connect(self, config: DatabaseCreateRequest) -> Database:
        database_id = str(uuid.uuid4())
        self.connector_factory(config, self.limits).test()
        now = datetime.now(timezone.utc)
        self._sources[database_id] = config
        return Database(
            id=database_id,
            name=config.name,
            connector=config.connector,
            schema_name=config.schema_name,
            status=DatabaseStatus.READY,
            created_at=now,
            updated_at=now,
        )

    def _config(self, source: Database) -> DatabaseCreateRequest:
        try:
            return self._sources[source.id]
        except KeyError as exc:
            raise ValueError("source was not created by this client") from exc

    def test_connection(self, source: Database) -> ConnectionStatus:
        latency = self.connector_factory(self._config(source), self.limits).test()
        return ConnectionStatus(
            database_id=source.id,
            status=DatabaseStatus.READY,
            latency_ms=latency,
        )

    def discover(self, source: Database) -> SchemaGraph:
        return self.connector_factory(self._config(source), self.limits).discover(
            source.id
        )

    def predict(self, source: Database, request: PredictionRequest) -> pd.DataFrame:
        config = self._config(source)
        if isinstance(request.target, TemporalTarget):
            target_table, target_time = request.target.time_column.split(".", 1)
            config = config.model_copy(
                update={
                    "time_columns": {
                        **config.time_columns,
                        target_table: target_time,
                    }
                }
            )

        as_of = request.as_of or datetime.now(timezone.utc)
        snapshot = self.connector_factory(config, self.limits).snapshot(
            source.id, as_of
        )
        examples = build_examples(
            snapshot,
            request,
            max_anchors=self.limits.max_training_anchors,
            min_training_rows=self.limits.min_training_rows,
        )
        matrices = self.feature_engine.compute(snapshot, examples, request.entity_table)
        feature_count = len(matrices.feature_names)
        context_rows = min(
            self.limits.max_context_rows,
            self.limits.max_feature_elements // feature_count,
        )
        if context_rows < self.limits.min_training_rows:
            raise RuntimeError(
                "feature matrix is too wide for the configured Nori context budget"
            )
        train = matrices.train.tail(context_rows).reset_index(drop=True)
        target = examples.y_train.tail(context_rows).reset_index(drop=True)
        predictions, quantiles = self.predictor.predict(
            train, target, matrices.query, request
        )

        rows: list[dict[str, Any]] = []
        for index, (entity_id, prediction) in enumerate(
            zip(examples.entity_ids, predictions, strict=True)
        ):
            row: dict[str, Any] = {
                "entity_id": entity_id,
                "as_of": as_of,
                "prediction": prediction,
            }
            if request.task is PredictionTask.CLASSIFICATION:
                if prediction is None or examples.class_labels is None:
                    raise RuntimeError("classification returned an incomplete result")
                negative, positive = examples.class_labels
                threshold = request.decision_threshold
                if threshold is None:
                    raise RuntimeError("classification threshold is missing")
                row["prediction"] = positive if prediction >= threshold else negative
                row["probability"] = prediction
            elif quantiles is not None:
                row["quantiles"] = quantiles[index]
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame.attrs.update(
            {
                "entity_key": examples.entity_key,
                "task": request.task.value,
                "output_type": (
                    None if request.output_type is None else request.output_type.value
                ),
                "taus": request.quantiles,
                "metadata": {
                    "model": self.predictor.model_for(request.task),
                    "feature_engine": "fastdfs",
                    "feature_depth": self.feature_engine.max_depth,
                    "feature_count": feature_count,
                    "training_rows": len(target),
                    "training_anchors": [
                        anchor.isoformat() for anchor in examples.training_anchors
                    ],
                    "positive_class": (
                        None
                        if examples.class_labels is None
                        else examples.class_labels[1]
                    ),
                    "decision_threshold": request.decision_threshold,
                    "relationship_path": examples.relationship_path,
                    "snapshot_as_of": snapshot.as_of.isoformat(),
                    "snapshot_rows": sum(
                        len(table) for table in snapshot.tables.values()
                    ),
                    "snapshot_bytes": snapshot.approximate_bytes,
                },
            }
        )
        return frame
