"""Connector contract shared by PostgreSQL and future relational backends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from synthefy.relational.models import SchemaGraph


@dataclass(frozen=True)
class Snapshot:
    tables: dict[str, pd.DataFrame]
    primary_keys: dict[str, list[str]]
    foreign_keys: list[tuple[str, str, str, str]]
    time_columns: dict[str, str]
    schema: SchemaGraph
    as_of: datetime
    approximate_bytes: int


class RelationalConnector(Protocol):
    def test(self) -> float: ...

    def discover(self, database_id: str) -> SchemaGraph: ...

    def snapshot(self, database_id: str, as_of: datetime) -> Snapshot: ...
