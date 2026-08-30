from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from synthefy.relational._features import FastDFSFeatureEngine, FeatureMatrices
from synthefy.relational._outcomes import OutcomeError, build_examples
from synthefy.relational._runtime import HostedNoriPredictor, LocalRelationalRuntime
from synthefy.relational._runtime_config import RelationalLimits
from synthefy.relational.connectors.base import Snapshot
from synthefy.relational.models import (
    ColumnSchema,
    DatabaseCreateRequest,
    EnvironmentCredential,
    ForeignKeySchema,
    PredictionRequest,
    SchemaGraph,
    TableSchema,
)


@pytest.fixture
def relational_snapshot() -> Snapshot:
    drivers = pd.DataFrame(
        {
            "driver_id": [1, 2, 3, 4],
            "experience": [3, 5, 8, 13],
            "team": ["a", "a", "b", "b"],
        }
    )
    results = pd.DataFrame(
        [
            {
                "result_id": day * 10 + driver_id,
                "driver_id": driver_id,
                "race_date": pd.Timestamp(f"2026-01-{day:02d}", tz="UTC"),
                "position": float(driver_id + day),
            }
            for day in range(1, 11)
            for driver_id in drivers["driver_id"]
        ]
    )
    schema = SchemaGraph(
        database_id="db-1",
        schema_name="public",
        tables=[
            TableSchema(
                name="drivers",
                columns=[
                    ColumnSchema(name="driver_id", data_type="INTEGER", nullable=False),
                    ColumnSchema(
                        name="experience", data_type="INTEGER", nullable=False
                    ),
                    ColumnSchema(name="team", data_type="TEXT", nullable=False),
                ],
                primary_key=["driver_id"],
            ),
            TableSchema(
                name="results",
                columns=[
                    ColumnSchema(name="result_id", data_type="INTEGER", nullable=False),
                    ColumnSchema(name="driver_id", data_type="INTEGER", nullable=False),
                    ColumnSchema(
                        name="race_date", data_type="TIMESTAMPTZ", nullable=False
                    ),
                    ColumnSchema(name="position", data_type="FLOAT", nullable=False),
                ],
                primary_key=["result_id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["driver_id"],
                        referred_table="drivers",
                        referred_columns=["driver_id"],
                    )
                ],
                time_column="race_date",
            ),
        ],
        discovered_at=datetime.now(UTC),
    )
    return Snapshot(
        tables={"drivers": drivers, "results": results},
        primary_keys={"drivers": ["driver_id"], "results": ["result_id"]},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
        time_columns={"results": "race_date"},
        schema=schema,
        as_of=datetime(2026, 1, 10, tzinfo=UTC),
        approximate_bytes=10_000,
    )


def _temporal_request(**changes) -> PredictionRequest:
    values = {
        "database": "db-1",
        "entity_table": "drivers",
        "target": {
            "kind": "temporal",
            "column": "results.position",
            "time_column": "results.race_date",
            "operation": "next",
            "lookahead": "2 days",
        },
        "as_of": datetime(2026, 1, 10, tzinfo=UTC),
    }
    values.update(changes)
    return PredictionRequest(**values)


def test_historical_labels_use_strict_completed_future_windows(
    relational_snapshot: Snapshot,
) -> None:
    examples = build_examples(
        relational_snapshot,
        _temporal_request(entity_ids=[1]),
        max_anchors=2,
        min_training_rows=2,
    )

    assert examples.training_anchors == [
        datetime(2026, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 8, tzinfo=UTC),
    ]
    assert examples.y_train.tolist() == [8.0, 10.0]
    assert examples.query_targets["driver_id"].tolist() == [1]


def test_fastdfs_builds_aligned_context_and_query_tables(
    relational_snapshot: Snapshot,
) -> None:
    request = _temporal_request()
    examples = build_examples(
        relational_snapshot, request, max_anchors=3, min_training_rows=4
    )

    matrices = FastDFSFeatureEngine(max_depth=2).compute(
        relational_snapshot, examples, "drivers"
    )

    assert len(matrices.train) == len(examples.y_train)
    assert len(matrices.query) == 4
    assert matrices.train.columns.tolist() == matrices.query.columns.tolist()
    assert matrices.feature_names


def test_direct_target_is_removed_before_fastdfs(
    relational_snapshot: Snapshot,
) -> None:
    drivers = relational_snapshot.tables["drivers"].assign(
        churned=pd.Series([False, True, False, None], dtype="boolean")
    )
    snapshot = Snapshot(
        **{
            **relational_snapshot.__dict__,
            "tables": {**relational_snapshot.tables, "drivers": drivers},
        }
    )
    request = PredictionRequest(
        database="db-1",
        entity_table="drivers",
        task="classification",
        target={"kind": "direct", "column": "drivers.churned"},
    )
    examples = build_examples(snapshot, request, max_anchors=2, min_training_rows=3)

    matrices = FastDFSFeatureEngine(max_depth=2).compute(snapshot, examples, "drivers")

    assert examples.y_train.tolist() == [0.0, 1.0, 0.0]
    assert not any("churned" in name.lower() for name in matrices.feature_names)


def test_ambiguous_relationship_paths_fail_closed(
    relational_snapshot: Snapshot,
) -> None:
    snapshot = Snapshot(
        **{
            **relational_snapshot.__dict__,
            "tables": {
                **relational_snapshot.tables,
                "teams": relational_snapshot.tables["drivers"].copy(),
                "owners": relational_snapshot.tables["drivers"].copy(),
            },
            "foreign_keys": [
                ("results", "driver_id", "teams", "driver_id"),
                ("teams", "driver_id", "drivers", "driver_id"),
                ("results", "driver_id", "owners", "driver_id"),
                ("owners", "driver_id", "drivers", "driver_id"),
            ],
        }
    )
    with pytest.raises(OutcomeError, match="multiple relationship paths"):
        build_examples(
            snapshot, _temporal_request(), max_anchors=2, min_training_rows=2
        )


def test_runtime_executes_database_and_features_locally(
    relational_snapshot: Snapshot,
) -> None:
    captured = {}

    class Connector:
        def __init__(self, config, limits) -> None:
            captured["credential"] = config.credential

        def test(self) -> float:
            return 1.0

        def discover(self, database_id):
            return relational_snapshot.schema.model_copy(
                update={"database_id": database_id}
            )

        def snapshot(self, database_id, as_of):
            captured["snapshot"] = (database_id, as_of)
            return Snapshot(**{**relational_snapshot.__dict__, "as_of": as_of})

    class Features:
        max_depth = 2

        def compute(self, snapshot, examples, entity_table):
            captured["features"] = (snapshot, examples, entity_table)
            return FeatureMatrices(
                train=pd.DataFrame({"feature": range(len(examples.y_train))}),
                query=pd.DataFrame({"feature": range(len(examples.entity_ids))}),
                feature_names=["feature"],
            )

    class Predictor:
        def model_for(self, task):
            return "nori-30m"

        def predict(self, train, target, query, request):
            captured["model_input"] = (train, target, query, request)
            return [4.5] * len(query), None

    runtime = LocalRelationalRuntime(
        api_key="hosted-model-key",
        model="nori-30m",
        classification_model=None,
        timeout=30,
        max_retries=0,
        limits=RelationalLimits(min_training_rows=4, max_training_anchors=2),
        connector_factory=Connector,
        feature_engine=Features(),
        predictor=Predictor(),
    )
    source = runtime.connect(
        DatabaseCreateRequest(
            name="testdb",
            connector="postgresql",
            credential=EnvironmentCredential(variable="DATABASE_URL"),
        )
    )

    result = runtime.predict(source, _temporal_request(database=source.id))

    train, target, query, _ = captured["model_input"]
    assert list(train.columns) == ["feature"]
    assert len(train) == len(target)
    assert len(query) == 4
    assert result["prediction"].tolist() == [4.5] * 4
    assert result.attrs["metadata"]["feature_engine"] == "fastdfs"


def test_hosted_boundary_receives_only_flat_feature_matrices(monkeypatch) -> None:
    captured = {}

    class NoriClient:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **kwargs):
            captured["prediction"] = (train, target, query, kwargs)
            return pd.Series([2.5])

    monkeypatch.setattr("synthefy.relational._runtime.SynthefyNoriClient", NoriClient)
    request = PredictionRequest(
        database="db-1",
        entity_table="customers",
        target={"kind": "direct", "column": "customers.value"},
    )
    predictor = HostedNoriPredictor("synthefy-key", "nori-30m")
    train = pd.DataFrame({"orders.total": [1.0, 2.0]})
    target = pd.Series([1.5, 3.0])
    query = pd.DataFrame({"orders.total": [4.0]})

    predictions, quantiles = predictor.predict(train, target, query, request)

    assert captured["client"]["api_key"] == "synthefy-key"
    assert captured["client"]["model"] == "nori-30m"
    assert captured["prediction"][0] is train
    assert captured["prediction"][1] is target
    assert captured["prediction"][2] is query
    assert predictions == [2.5]
    assert quantiles is None
