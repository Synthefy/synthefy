from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from synthefy.relational import (
    AwsSecret,
    Database,
    DatabaseStatus,
    DatabaseUrl,
    SynthefyNoriRelClient,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.connected = None
        self.prediction = None
        self.source = Database(
            id="db-1",
            name="production-rds",
            connector="postgresql",
            schema_name="public",
            status=DatabaseStatus.READY,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def connect(self, config):
        self.connected = config
        return self.source

    def test_connection(self, source):
        return {"source": source}

    def discover(self, source):
        return {"source": source}

    def predict(self, source, request):
        self.prediction = (source, request)
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})


def _client(runtime: FakeRuntime | None = None) -> SynthefyNoriRelClient:
    return SynthefyNoriRelClient(
        api_key="hosted-model-key",
        _runtime=runtime or FakeRuntime(),
    )


def test_client_uses_the_existing_synthefy_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SYNTHEFY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYNTHEFY_API_KEY"):
        SynthefyNoriRelClient(_runtime=FakeRuntime())

    monkeypatch.setenv("SYNTHEFY_API_KEY", "environment-key")
    client = SynthefyNoriRelClient(_runtime=FakeRuntime())

    assert "environment-key" not in repr(client)
    assert not hasattr(client, "base_url")


def test_connect_keeps_the_database_endpoint_on_the_source() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)

    source = client.connect(
        name="production-rds",
        database_url="postgresql://readonly:secret@db.internal/customer",
        tables=["customers"],
    )

    assert source is runtime.source
    assert isinstance(runtime.connected.credential, DatabaseUrl)
    assert "secret" not in repr(runtime.connected)


def test_connect_accepts_an_indirect_secret_reference() -> None:
    runtime = FakeRuntime()
    secret = AwsSecret(secret_id="production/postgres", region_name="us-east-1")

    _client(runtime).connect(name="production-rds", credential=secret)

    assert runtime.connected.credential == secret


@pytest.mark.parametrize(
    ("database_url", "credential"),
    [(None, None), ("postgresql://db/name", AwsSecret(secret_id="db"))],
)
def test_connect_requires_one_database_credential(database_url, credential) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _client().connect(
            name="production-rds",
            database_url=database_url,
            credential=credential,
        )


def test_predict_builds_a_temporal_request_and_runs_locally() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)
    source = runtime.source
    as_of = datetime(2026, 1, 10, tzinfo=UTC)

    result = client.predict(
        source=source,
        entity="drivers",
        target="results.position",
        target_time="results.race_date",
        operation="next",
        lookahead="30 days",
        as_of=as_of,
    )

    sent_source, request = runtime.prediction
    assert sent_source is source
    assert request.database == "db-1"
    assert request.target.operation == "next"
    assert request.target.lookahead == "30 days"
    assert request.as_of == as_of
    assert result["prediction"].tolist() == [3.5]


def test_predict_supports_direct_binary_classification() -> None:
    runtime = FakeRuntime()

    _client(runtime).predict(
        source=runtime.source,
        entity="customers",
        target="customers.churned",
        task="classification",
        positive_class=True,
    )

    request = runtime.prediction[1]
    assert request.target.kind == "direct"
    assert request.task == "classification"
    assert request.decision_threshold == 0.5


def test_partial_temporal_target_is_rejected_before_database_access() -> None:
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="provided together"):
        _client(runtime).predict(
            source=runtime.source,
            entity="drivers",
            target="results.position",
            target_time="results.race_date",
        )
    assert runtime.prediction is None


def test_version_one_is_synchronous() -> None:
    assert not hasattr(_client(), "submit")
