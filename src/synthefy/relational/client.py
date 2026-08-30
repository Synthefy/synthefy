"""Public client for enterprise Nori-Rel predictions."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional, Union
from urllib.parse import quote

import httpx
import pandas as pd

from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    _raise_for_status,
)
from synthefy.relational.models import (
    ConnectionStatus,
    ConnectorType,
    CredentialReference,
    Database,
    DatabaseCreateRequest,
    DirectTarget,
    PredictionJobRecord,
    PredictionRequest,
    PredictionResult,
    PredictionStatus,
    RelationalOutputType,
    SchemaGraph,
    TargetDefinition,
    TemporalTarget,
)
from synthefy.relational.models import (
    PredictionOperation as WirePredictionOperation,
)
from synthefy.relational.models import PredictionTask as WirePredictionTask

NORI_REL_BASE_URL_ENV = "SYNTHEFY_NORI_REL_BASE_URL"
NORI_REL_API_KEY_ENV = "SYNTHEFY_NORI_REL_API_KEY"
_TERMINAL_STATUSES = frozenset(
    {
        PredictionStatus.SUCCEEDED,
        PredictionStatus.FAILED,
        PredictionStatus.CANCELLED,
    }
)
PredictionOperation = Literal[
    "next", "average", "total", "minimum", "maximum", "count"
]
PredictionTask = Literal["regression", "classification"]
_OPERATIONS = {operation.value for operation in WirePredictionOperation}


class PredictionFailedError(RuntimeError):
    """A submitted prediction reached a non-success terminal state."""


class PredictionJob:
    """Handle for a non-blocking Nori-Rel prediction."""

    def __init__(
        self, client: "SynthefyNoriRelClient", record: PredictionJobRecord
    ) -> None:
        self._client = client
        self._record = record

    @property
    def id(self) -> str:
        return self._record.id

    @property
    def status(self) -> PredictionStatus:
        self.refresh()
        return self._record.status

    @property
    def progress(self) -> float:
        self.refresh()
        return self._record.progress

    @property
    def error(self) -> Optional[str]:
        self.refresh()
        return self._record.error

    def refresh(self) -> PredictionJobRecord:
        self._record = self._client.get_job(self.id)
        return self._record

    def cancel(self) -> PredictionJobRecord:
        self._record = self._client.cancel(self.id)
        return self._record

    def wait(
        self, *, timeout: Optional[float] = None, poll_interval: float = 1.0
    ) -> pd.DataFrame:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        started = time.monotonic()
        while True:
            record = self.refresh()
            if record.status in _TERMINAL_STATUSES:
                break
            if timeout is not None and time.monotonic() - started >= timeout:
                raise APITimeoutError(
                    f"Prediction job {self.id} did not finish within {timeout} seconds"
                )
            remaining = None
            if timeout is not None:
                remaining = max(0.0, timeout - (time.monotonic() - started))
            time.sleep(
                poll_interval if remaining is None else min(poll_interval, remaining)
            )

        if record.status != PredictionStatus.SUCCEEDED:
            detail = record.error or "no error detail was returned"
            raise PredictionFailedError(
                f"Prediction job {self.id} ended as {record.status.value}: {detail}"
            )
        return self._client.get_result(self.id)


class SynthefyNoriRelClient:
    """Client for enterprise relational prediction jobs.

    Database access, FastDFS, and Nori execution happen in the connector agent,
    never inside this lightweight package.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        user_agent: Optional[str] = None,
    ) -> None:
        base_url = base_url or os.getenv(NORI_REL_BASE_URL_ENV)
        if not base_url or not base_url.strip():
            raise ValueError(
                "A Nori-Rel agent URL is required through base_url or "
                f"{NORI_REL_BASE_URL_ENV}"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if api_key is None:
            api_key = os.getenv(NORI_REL_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                "A Nori-Rel API key is required through api_key, "
                f"or {NORI_REL_API_KEY_ENV}"
            )

        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = (
            user_agent or f"synthefy-nori-rel httpx/{httpx.__version__}"
        )
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"timeout={self.timeout!r}, max_retries={self.max_retries!r})"
        )

    def __enter__(self) -> "SynthefyNoriRelClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(
                    method, path, json=json, headers=headers
                )
            except httpx.TimeoutException as exc:
                last_error = APITimeoutError(str(exc))
            except httpx.RequestError as exc:
                last_error = APIConnectionError(str(exc))
            else:
                if response.status_code not in (429,) and response.status_code < 500:
                    _raise_for_status(response)
                    return response
                try:
                    _raise_for_status(response)
                except Exception as exc:
                    last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))

        if last_error is None:  # pragma: no cover - defensive invariant
            raise APIConnectionError("Nori-Rel request failed without an error")
        raise last_error

    @property
    def api_key(self) -> str:
        """Authentication key. Never included in the client's representation."""

        return self._api_key

    @staticmethod
    def _path_id(value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("resource id must not be blank")
        return quote(value, safe="")

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
        assert target_time is not None
        assert operation is not None
        assert lookahead is not None
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
        credential: CredentialReference,
        connector: Union[ConnectorType, str] = ConnectorType.POSTGRESQL,
        schema_name: str = "public",
        tables: Optional[List[str]] = None,
        time_columns: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> Database:
        request = DatabaseCreateRequest(
            name=name,
            connector=connector,
            credential=credential,
            schema_name=schema_name,
            tables=tables,
            time_columns=time_columns or {},
        )
        response = self._request(
            "POST",
            "/v1/databases",
            json=request.model_dump(mode="json"),
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )
        return Database.model_validate(response.json())

    def test_connection(self, database: Union[Database, str]) -> ConnectionStatus:
        database_id = database.id if isinstance(database, Database) else database
        database_id = self._path_id(database_id)
        response = self._request("POST", f"/v1/databases/{database_id}/test")
        return ConnectionStatus.model_validate(response.json())

    def discover(self, database: Union[Database, str]) -> SchemaGraph:
        database_id = database.id if isinstance(database, Database) else database
        database_id = self._path_id(database_id)
        response = self._request("POST", f"/v1/databases/{database_id}/discover")
        return SchemaGraph.model_validate(response.json())

    def submit(
        self,
        *,
        source: Union[Database, str],
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
        idempotency_key: Optional[str] = None,
    ) -> PredictionJob:
        database_id = source.id if isinstance(source, Database) else source
        request = PredictionRequest(
            database=database_id,
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
        response = self._request(
            "POST",
            "/v1/predictions",
            json=request.model_dump(mode="json", exclude_none=True),
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )
        return PredictionJob(self, PredictionJobRecord.model_validate(response.json()))

    def predict(
        self,
        *,
        source: Union[Database, str],
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
        idempotency_key: Optional[str] = None,
        wait_timeout: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> pd.DataFrame:
        return self.submit(
            source=source,
            entity=entity,
            target=target,
            task=task,
            target_time=target_time,
            operation=operation,
            lookahead=lookahead,
            as_of=as_of,
            relationship_path=relationship_path,
            entity_ids=entity_ids,
            output_type=output_type,
            quantiles=quantiles,
            positive_class=positive_class,
            decision_threshold=decision_threshold,
            idempotency_key=idempotency_key,
        ).wait(
            timeout=wait_timeout, poll_interval=poll_interval
        )

    def get_job(self, job_id: str) -> PredictionJobRecord:
        job_id = self._path_id(job_id)
        response = self._request("GET", f"/v1/predictions/{job_id}")
        return PredictionJobRecord.model_validate(response.json())

    def cancel(self, job_id: str) -> PredictionJobRecord:
        job_id = self._path_id(job_id)
        response = self._request("DELETE", f"/v1/predictions/{job_id}")
        return PredictionJobRecord.model_validate(response.json())

    def get_result(self, job_id: str) -> pd.DataFrame:
        job_id = self._path_id(job_id)
        response = self._request("GET", f"/v1/predictions/{job_id}/result")
        result = PredictionResult.model_validate(response.json())
        excluded = (
            {"probability"}
            if result.task is WirePredictionTask.REGRESSION
            else {"quantiles"}
        )
        rows = [
            row.model_dump(mode="python", exclude=excluded) for row in result.rows
        ]
        frame = pd.DataFrame(rows)
        frame.attrs.update(
            {
                "job_id": result.job_id,
                "entity_key": result.entity_key,
                "task": result.task.value,
                "output_type": (
                    None if result.output_type is None else result.output_type.value
                ),
                "taus": result.taus,
                "metadata": result.metadata,
            }
        )
        return frame
