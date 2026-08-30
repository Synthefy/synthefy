"""Relationship resolution and leakage-safe outcome construction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from synthefy.relational.connectors.base import Snapshot
from synthefy.relational.models import (
    DirectTarget,
    PredictionOperation,
    PredictionRequest,
    PredictionTask,
    TemporalTarget,
)

_ENTITY = "__nori_rel_entity_id"
_CUTOFF = "__nori_rel_cutoff_time"


class OutcomeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Examples:
    train_targets: pd.DataFrame
    y_train: pd.Series
    query_targets: pd.DataFrame
    entity_ids: list[Any]
    entity_key: str
    relationship_path: list[str]
    training_anchors: list[datetime]
    class_labels: tuple[Any, Any] | None = None
    excluded_columns: dict[str, list[str]] | None = None


def _adjacency(snapshot: Snapshot) -> dict[str, set[str]]:
    result = {name: set() for name in snapshot.tables}
    for child_table, _, parent_table, _ in snapshot.foreign_keys:
        if child_table in result and parent_table in result:
            result[child_table].add(parent_table)
            result[parent_table].add(child_table)
    return result


def resolve_path(
    snapshot: Snapshot,
    entity_table: str,
    target_table: str,
    explicit: list[str] | None,
) -> list[str]:
    if entity_table not in snapshot.tables:
        raise OutcomeError(f"entity table {entity_table!r} is not in the source")
    if target_table not in snapshot.tables:
        raise OutcomeError(f"target table {target_table!r} is not in the source")
    adjacency = _adjacency(snapshot)
    if explicit is not None:
        for left, right in zip(explicit, explicit[1:], strict=False):
            if right not in adjacency.get(left, set()):
                raise OutcomeError(
                    f"no foreign-key edge connects {left!r} to {right!r}"
                )
        return explicit
    if entity_table == target_table:
        return [entity_table]

    queue: deque[list[str]] = deque([[entity_table]])
    shortest: list[list[str]] = []
    shortest_length: int | None = None
    while queue:
        path = queue.popleft()
        if shortest_length is not None and len(path) > shortest_length:
            break
        current = path[-1]
        if current == target_table:
            shortest.append(path)
            shortest_length = len(path)
            continue
        for neighbor in sorted(adjacency[current]):
            if neighbor not in path:
                queue.append([*path, neighbor])
    if not shortest:
        raise OutcomeError(
            f"no foreign-key path connects {entity_table!r} to {target_table!r}"
        )
    if len(shortest) > 1:
        raise OutcomeError(
            "multiple relationship paths are possible; pass relationship_path "
            "explicitly"
        )
    return shortest[0]


def _single_primary_key(snapshot: Snapshot, table: str) -> str:
    keys = snapshot.primary_keys.get(table, [])
    if len(keys) != 1:
        raise OutcomeError(
            f"version 1 requires one primary-key column on entity table {table!r}"
        )
    return keys[0]


def map_target_rows(snapshot: Snapshot, path: list[str]) -> tuple[pd.DataFrame, str]:
    entity_table = path[0]
    entity_key = _single_primary_key(snapshot, entity_table)
    if _ENTITY in snapshot.tables[entity_table].columns:
        raise OutcomeError(f"reserved column {_ENTITY!r} exists in {entity_table!r}")
    current = snapshot.tables[entity_table].copy()
    current[_ENTITY] = current[entity_key]

    for left, right in zip(path, path[1:], strict=False):
        candidates = [
            relationship
            for relationship in snapshot.foreign_keys
            if {relationship[0], relationship[2]} == {left, right}
        ]
        if len(candidates) != 1:
            raise OutcomeError(
                f"expected one single-column foreign key between {left!r} and {right!r}"
            )
        child_table, child_column, parent_table, parent_column = candidates[0]
        if left == child_table:
            current_join = child_column
            next_join = parent_column
        else:
            current_join = parent_column
            next_join = child_column
        links = current[[_ENTITY, current_join]].rename(
            columns={current_join: "__nori_rel_join_key"}
        )
        next_frame = snapshot.tables[right]
        current = next_frame.merge(
            links,
            how="inner",
            left_on=next_join,
            right_on="__nori_rel_join_key",
            validate="many_to_many",
        ).drop(columns="__nori_rel_join_key")

    target_pk = snapshot.primary_keys.get(path[-1], [])
    if len(target_pk) == 1 and target_pk[0] in current:
        current = current.drop_duplicates(subset=[_ENTITY, target_pk[0]])
    return current, entity_key


def _anchor_times(
    event_times: pd.Series,
    as_of: datetime,
    horizon: timedelta,
    max_anchors: int,
) -> list[datetime]:
    earliest = event_times.min()
    final = pd.Timestamp(as_of) - horizon
    if pd.isna(earliest) or final < earliest:
        return []
    anchors: list[datetime] = []
    cursor = final
    while cursor >= earliest and len(anchors) < max_anchors:
        anchors.append(cursor.to_pydatetime())
        cursor -= horizon
    return list(reversed(anchors))


def _aggregate(
    events: pd.DataFrame,
    operation: PredictionOperation,
    target_column: str,
) -> pd.Series:
    grouped = events.groupby(_ENTITY, sort=False)[target_column]
    if operation is PredictionOperation.NEXT:
        return grouped.first()
    if operation is PredictionOperation.AVERAGE:
        return grouped.mean()
    if operation is PredictionOperation.TOTAL:
        return grouped.sum(min_count=1)
    if operation is PredictionOperation.MINIMUM:
        return grouped.min()
    if operation is PredictionOperation.MAXIMUM:
        return grouped.max()
    return grouped.count()


def _encode_targets(
    labels: pd.Series,
    request: PredictionRequest,
) -> tuple[pd.Series, tuple[Any, Any] | None]:
    if request.task is PredictionTask.REGRESSION:
        numeric = pd.to_numeric(labels, errors="coerce")
        keep = numeric.notna() & np.isfinite(numeric)
        return numeric.loc[keep].astype(float), None

    values = labels.dropna().reset_index(drop=True)
    classes = [
        value.item() if isinstance(value, np.generic) else value
        for value in pd.unique(values)
    ]
    if len(classes) != 2:
        raise OutcomeError(
            f"binary classification requires exactly two classes; found {len(classes)}"
        )
    positive = request.positive_class
    if positive is None:
        class_set = set(classes)
        if class_set == {False, True}:
            positive = True
        elif class_set == {0, 1}:
            positive = 1
        else:
            raise OutcomeError(
                "positive_class is required unless classification labels are boolean "
                "or 0/1"
            )
    matches = [value for value in classes if value == positive]
    if len(matches) != 1:
        raise OutcomeError("positive_class is not present in the training labels")
    positive = matches[0]
    negative = next(value for value in classes if value != positive)
    encoded = values.eq(positive).astype(float)
    return encoded, (negative, positive)


def _build_direct_examples(
    snapshot: Snapshot,
    request: PredictionRequest,
    *,
    min_training_rows: int,
) -> Examples:
    if not isinstance(request.target, DirectTarget):
        raise TypeError("direct target builder received a temporal target")
    target_table, target_column = request.target.column.split(".", 1)
    entity_key = _single_primary_key(snapshot, request.entity_table)
    if target_table not in snapshot.tables:
        raise OutcomeError(f"entity table {target_table!r} is not in the source")
    frame = snapshot.tables[target_table]
    if target_column not in frame:
        raise OutcomeError(
            f"target column {request.target.column!r} is not in the source"
        )
    if target_column == entity_key:
        raise OutcomeError("the entity primary key cannot be a prediction target")
    if frame[entity_key].duplicated().any():
        raise OutcomeError("entity primary keys must be unique")

    if request.entity_ids is None:
        query_mask = frame[target_column].isna()
        if not query_mask.any():
            raise OutcomeError(
                "a direct target needs null target rows or explicit entity_ids "
                "to predict"
            )
    else:
        requested = set(request.entity_ids)
        available = set(frame[entity_key].tolist())
        missing = requested.difference(available)
        if missing:
            raise OutcomeError(f"unknown entity_ids: {sorted(missing, key=str)[:10]}")
        query_mask = frame[entity_key].isin(requested)

    train_frame = frame.loc[~query_mask & frame[target_column].notna()]
    raw_targets = train_frame[target_column].reset_index(drop=True)
    encoded, class_labels = _encode_targets(raw_targets, request)
    if request.task is PredictionTask.REGRESSION:
        keep = encoded.index
        train_frame = train_frame.reset_index(drop=True).loc[keep]
    else:
        train_frame = train_frame.reset_index(drop=True)
    if len(train_frame) < min_training_rows:
        raise OutcomeError(
            f"only {len(train_frame)} labeled rows are available; "
            f"at least {min_training_rows} are required"
        )

    as_of = request.as_of or snapshot.as_of
    train_targets = pd.DataFrame(
        {entity_key: train_frame[entity_key].tolist(), _CUTOFF: as_of}
    )
    query_frame = frame.loc[query_mask]
    entity_ids = query_frame[entity_key].tolist()
    if not entity_ids:
        raise OutcomeError("no entities are available for prediction")
    query_targets = pd.DataFrame({entity_key: entity_ids, _CUTOFF: as_of})
    return Examples(
        train_targets=train_targets,
        y_train=encoded.reset_index(drop=True),
        query_targets=query_targets,
        entity_ids=entity_ids,
        entity_key=f"{request.entity_table}.{entity_key}",
        relationship_path=[request.entity_table],
        training_anchors=[],
        class_labels=class_labels,
        excluded_columns={target_table: [target_column]},
    )


def _build_temporal_examples(
    snapshot: Snapshot,
    request: PredictionRequest,
    *,
    max_anchors: int,
    min_training_rows: int,
) -> Examples:
    if not isinstance(request.target, TemporalTarget):
        raise TypeError("temporal target builder received a direct target")
    target_table, target_column = request.target.column.split(".", 1)
    _, time_column = request.target.time_column.split(".", 1)
    configured_time = snapshot.time_columns.get(target_table)
    if configured_time != time_column:
        raise OutcomeError(
            f"register {target_table}.{time_column} as the table's time column"
        )
    path = resolve_path(
        snapshot,
        request.entity_table,
        target_table,
        request.relationship_path,
    )
    mapped, entity_key = map_target_rows(snapshot, path)
    if target_column not in mapped or time_column not in mapped:
        raise OutcomeError("target or event-time column is unavailable after path join")
    try:
        mapped[time_column] = pd.to_datetime(
            mapped[time_column], utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise OutcomeError("event_time contains invalid timestamps") from exc

    as_of = request.as_of or snapshot.as_of
    horizon = request.target.lookahead_delta()
    entity_frame = snapshot.tables[request.entity_table]
    entity_values = entity_frame[entity_key].drop_duplicates()
    if request.entity_ids is not None:
        requested = set(request.entity_ids)
        entity_values = entity_values[entity_values.isin(requested)]
        missing = requested.difference(entity_values.tolist())
        if missing:
            raise OutcomeError(f"unknown entity_ids: {sorted(missing, key=str)[:10]}")
    entity_ids = entity_values.tolist()
    if not entity_ids:
        raise OutcomeError("no entities are available for prediction")

    anchors = _anchor_times(
        mapped[time_column], as_of, horizon, max_anchors=max_anchors
    )
    target_frames: list[pd.DataFrame] = []
    labels: list[pd.Series] = []
    for anchor in anchors:
        anchor_timestamp = pd.Timestamp(anchor)
        window = mapped[
            (mapped[time_column] > anchor_timestamp)
            & (mapped[time_column] <= anchor_timestamp + horizon)
        ].sort_values(time_column)
        outcome = _aggregate(window, request.target.operation, target_column)
        frame = pd.DataFrame({entity_key: entity_ids, _CUTOFF: anchor})
        label = frame[entity_key].map(outcome)
        if request.target.operation in {
            PredictionOperation.TOTAL,
            PredictionOperation.COUNT,
        }:
            label = label.fillna(0.0)
        if request.task is PredictionTask.REGRESSION:
            numeric = pd.to_numeric(label, errors="coerce")
            keep = label.notna() & np.isfinite(numeric)
            label = numeric
        else:
            keep = label.notna()
        target_frames.append(frame.loc[keep].reset_index(drop=True))
        labels.append(label.loc[keep].reset_index(drop=True))

    if not target_frames:
        raise OutcomeError("not enough history exists for one completed horizon")
    train_targets = pd.concat(target_frames, ignore_index=True)
    raw_targets = pd.concat(labels, ignore_index=True)
    y_train, class_labels = _encode_targets(raw_targets, request)
    if len(train_targets) < min_training_rows:
        raise OutcomeError(
            f"only {len(train_targets)} labeled rows are available; "
            f"at least {min_training_rows} are required"
        )
    query_targets = pd.DataFrame({entity_key: entity_ids, _CUTOFF: as_of})
    return Examples(
        train_targets=train_targets,
        y_train=y_train,
        query_targets=query_targets,
        entity_ids=entity_ids,
        entity_key=f"{request.entity_table}.{entity_key}",
        relationship_path=path,
        training_anchors=anchors,
        class_labels=class_labels,
    )


def build_examples(
    snapshot: Snapshot,
    request: PredictionRequest,
    *,
    max_anchors: int,
    min_training_rows: int,
) -> Examples:
    if isinstance(request.target, DirectTarget):
        return _build_direct_examples(
            snapshot,
            request,
            min_training_rows=min_training_rows,
        )
    return _build_temporal_examples(
        snapshot,
        request,
        max_anchors=max_anchors,
        min_training_rows=min_training_rows,
    )
