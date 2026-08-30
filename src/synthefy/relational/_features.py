"""FastDFS feature generation at entity-specific temporal cutoffs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from synthefy.relational._compat import install_pkg_resources_compatibility
from synthefy.relational._outcomes import _CUTOFF, Examples, OutcomeError
from synthefy.relational.connectors.base import Snapshot

install_pkg_resources_compatibility()

from fastdfs import DFSConfig, compute_dfs_features, create_rdb  # noqa: E402


@dataclass(frozen=True)
class FeatureMatrices:
    train: pd.DataFrame
    query: pd.DataFrame
    feature_names: list[str]


class FastDFSFeatureEngine:
    def __init__(self, *, max_depth: int = 2, max_features: int = -1) -> None:
        self.max_depth = max_depth
        self.max_features = max_features

    def compute(
        self, snapshot: Snapshot, examples: Examples, entity_table: str
    ) -> FeatureMatrices:
        entity_column = examples.entity_key.split(".", 1)[1]
        primary_keys = {
            table: columns[0]
            for table, columns in snapshot.primary_keys.items()
            if len(columns) == 1
        }
        if entity_table not in primary_keys:
            raise OutcomeError("FastDFS requires a single-column entity primary key")
        usable_foreign_keys = [
            relationship
            for relationship in snapshot.foreign_keys
            if relationship[0] in snapshot.tables
            and relationship[2] in snapshot.tables
            and relationship[2] in primary_keys
        ]
        tables = {name: frame.copy() for name, frame in snapshot.tables.items()}
        for table, columns in (examples.excluded_columns or {}).items():
            tables[table] = tables[table].drop(columns=columns)
        rdb = create_rdb(
            tables=tables,
            name="nori-rel-snapshot",
            primary_keys=primary_keys,
            foreign_keys=usable_foreign_keys,
            time_columns=snapshot.time_columns,
        )
        combined = pd.concat(
            [examples.train_targets, examples.query_targets], ignore_index=True
        )
        features = compute_dfs_features(
            rdb=rdb,
            target_dataframe=combined,
            key_mappings={entity_column: f"{entity_table}.{entity_column}"},
            cutoff_time_column=_CUTOFF,
            config=DFSConfig(
                max_depth=self.max_depth,
                use_cutoff_time=True,
                include_cutoff_time=True,
                max_features=self.max_features,
                engine="dfs2sql",
                n_jobs=1,
            ),
        )
        drop = [column for column in (entity_column, _CUTOFF) if column in features]
        features = features.drop(columns=drop)
        features = features.loc[:, ~features.columns.duplicated()].replace(
            [float("inf"), float("-inf")], pd.NA
        )
        if features.shape[1] == 0:
            raise OutcomeError("FastDFS generated no model features")
        split = len(examples.train_targets)
        train = features.iloc[:split].reset_index(drop=True)
        query = features.iloc[split:].reset_index(drop=True)
        return FeatureMatrices(
            train=train,
            query=query,
            feature_names=[str(column) for column in features.columns],
        )
