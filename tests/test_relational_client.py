"""Contract tests for the public Nori-Rel client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

import httpx
import pytest
from pydantic import ValidationError

from synthefy import SynthefyNoriRelClient
from synthefy.relational import (
    AwsSecret,
    EnvironmentCredential,
    PredictionFailedError,
    PredictionRequest,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _attach_mock(client: SynthefyNoriRelClient, handler: Handler) -> None:
    client.close()
    client.client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )


def _job(status: str, *, error: str | None = None) -> dict:
    return {
        "id": "job-1",
        "status": status,
        "progress": 1.0 if status in {"succeeded", "failed"} else 0.25,
        "created_at": "2026-08-28T10:00:00Z",
        "updated_at": "2026-08-28T10:01:00Z",
        "error": error,
    }


def test_client_uses_relational_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHEFY_NORI_REL_API_KEY", "rel-secret")
    monkeypatch.setenv("SYNTHEFY_NORI_REL_BASE_URL", "https://agent.example.test/")

    client = SynthefyNoriRelClient()

    assert client.api_key == "rel-secret"
    assert client.base_url == "https://agent.example.test"
    assert "rel-secret" not in repr(client)
    client.close()


def test_connect_sends_only_a_credential_reference() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "db-1",
                "name": "production-rds",
                "connector": "postgresql",
                "schema_name": "public",
                "status": "pending",
                "created_at": "2026-08-28T10:00:00Z",
                "updated_at": "2026-08-28T10:00:00Z",
            },
        )

    client = SynthefyNoriRelClient("control-key", base_url="https://unit.test")
    _attach_mock(client, handler)
    database = client.connect(
        name="production-rds",
        credential=AwsSecret(secret_id="prod/nori-rel/postgres"),
        tables=["drivers", "results"],
        time_columns={"results": "race_date"},
        idempotency_key="create-db-1",
    )

    assert database.id == "db-1"
    assert captured["headers"]["authorization"] == "Bearer control-key"
    assert captured["headers"]["idempotency-key"] == "create-db-1"
    assert captured["body"]["connector"] == "postgresql"
    assert captured["body"]["credential"] == {
        "provider": "aws_secrets_manager",
        "secret_id": "prod/nori-rel/postgres",
        "region_name": None,
    }
    assert "password" not in json.dumps(captured["body"]).lower()
    client.close()


def test_prediction_request_rejects_ambiguous_or_unsafe_contracts() -> None:
    common = {
        "database": "db-1",
        "entity_table": "drivers",
        "target": "results.position",
        "event_time": "results.race_date",
        "aggregation": "first",
        "horizon": "30 days",
    }

    with pytest.raises(ValidationError, match="timezone"):
        PredictionRequest(**common, as_of=datetime(2026, 8, 28))
    with pytest.raises(ValidationError, match="start at entity_table"):
        PredictionRequest(**common, relationship_path=["teams", "results"])
    with pytest.raises(ValidationError, match="quantiles must be sorted"):
        PredictionRequest(
            **common, output_type="quantiles", quantiles=[0.9, 0.1]
        )


def test_predict_waits_for_job_and_returns_dataframe() -> None:
    requests: list[tuple[str, str]] = []
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        requests.append((request.method, request.url.path))
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["database"] == "db-1"
            assert body["entity_table"] == "drivers"
            assert body["target"] == "results.position"
            assert body["event_time"] == "results.race_date"
            assert body["aggregation"] == "first"
            assert body["horizon"] == "1 day"
            assert body["as_of"] == "2026-08-28T10:00:00Z"
            return httpx.Response(202, json=_job("pending"))
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "job_id": "job-1",
                    "entity_key": "drivers.driver_id",
                    "output_type": "median",
                    "rows": [
                        {
                            "entity_id": 44,
                            "as_of": "2026-08-28T10:00:00Z",
                            "prediction": 3.5,
                        }
                    ],
                    "metadata": {"model": "nori-30m"},
                },
            )
        polls += 1
        return httpx.Response(200, json=_job("succeeded" if polls > 1 else "running"))

    client = SynthefyNoriRelClient("key", base_url="https://unit.test")
    _attach_mock(client, handler)
    result = client.predict(
        source="db-1",
        entity="drivers",
        target="results.position",
        target_time="results.race_date",
        operation="next",
        lookahead="1 days",
        as_of=datetime(2026, 8, 28, 10, tzinfo=timezone.utc),
        relationship_path=["drivers", "results"],
        poll_interval=0.001,
    )

    assert result.to_dict("records") == [
        {
            "entity_id": 44,
            "as_of": datetime(2026, 8, 28, 10, tzinfo=timezone.utc),
            "prediction": 3.5,
            "quantiles": None,
        }
    ]
    assert result.attrs["metadata"] == {"model": "nori-30m"}
    assert requests == [
        ("POST", "/v1/predictions"),
        ("GET", "/v1/predictions/job-1"),
        ("GET", "/v1/predictions/job-1"),
        ("GET", "/v1/predictions/job-1/result"),
    ]
    client.close()


def test_wait_surfaces_failed_job_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json=_job("pending"))
        return httpx.Response(200, json=_job("failed", error="row limit exceeded"))

    client = SynthefyNoriRelClient("key", base_url="https://unit.test")
    _attach_mock(client, handler)
    job = client.submit(
        source="db-1",
        entity="drivers",
        target="results.position",
        target_time="results.race_date",
        operation="average",
        lookahead="30 days",
    )

    with pytest.raises(PredictionFailedError, match="row limit exceeded"):
        job.wait(poll_interval=0.001)
    client.close()


def test_prediction_rejects_unknown_operation() -> None:
    client = SynthefyNoriRelClient("key", base_url="https://unit.test")
    with pytest.raises(ValueError, match="operation must be one of"):
        client.submit(
            source="db-1",
            entity="drivers",
            target="results.position",
            target_time="results.race_date",
            operation="median",  # type: ignore[arg-type]
            lookahead="30 days",
        )
    client.close()


def test_environment_credential_accepts_only_variable_names() -> None:
    assert EnvironmentCredential(variable="NORI_REL_DATABASE_URL").variable == (
        "NORI_REL_DATABASE_URL"
    )
    with pytest.raises(ValidationError):
        EnvironmentCredential(variable="postgresql://user:password@host/database")
