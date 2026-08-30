"""Bounded, read-only PostgreSQL discovery and snapshot extraction."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.sqltypes import Date, DateTime

from synthefy.relational._credentials import CredentialResolver
from synthefy.relational._runtime_config import RelationalLimits
from synthefy.relational.connectors.base import Snapshot
from synthefy.relational.models import (
    ColumnSchema,
    ConnectorCapabilities,
    DatabaseCreateRequest,
    ForeignKeySchema,
    SchemaGraph,
    TableSchema,
)


class ConnectorError(RuntimeError):
    pass


class SnapshotLimitError(ConnectorError):
    pass


def _is_local_host(host: str | None) -> bool:
    if not host or host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class PostgreSQLConnector:
    def __init__(
        self,
        config: DatabaseCreateRequest,
        settings: RelationalLimits,
        resolver: CredentialResolver | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.resolver = resolver or CredentialResolver()

    def _engine(self) -> Engine:
        url = self._secure_url(self.resolver.resolve(self.config.credential))
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=0,
            connect_args={
                "connect_timeout": 10,
                "application_name": "nori-rel",
                "options": (
                    f"-c statement_timeout={self.settings.statement_timeout_ms} "
                    "-c default_transaction_read_only=on"
                ),
            },
        )

    @staticmethod
    def _secure_url(url: URL) -> URL:
        query = {
            key: value
            for key, value in url.query.items()
            if key not in {"options", "application_name", "connect_timeout"}
        }
        if not _is_local_host(url.host):
            sslmode = query.get("sslmode", "require")
            if sslmode not in {"require", "verify-ca", "verify-full"}:
                raise ConnectorError("remote PostgreSQL connections require TLS")
            query["sslmode"] = sslmode
        return url.set(query=query)

    def _read_bounded(self, query: Any, connection: Any) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        rows = 0
        size = 0
        for chunk in pd.read_sql(query, connection, chunksize=50_000):
            rows += len(chunk)
            if rows > self.settings.max_rows_per_table:
                raise SnapshotLimitError(
                    f"table exceeds the row limit of {self.settings.max_rows_per_table}"
                )
            size += int(chunk.memory_usage(index=True, deep=True).sum())
            if size > self.settings.max_snapshot_bytes:
                raise SnapshotLimitError(
                    "table exceeds the configured snapshot memory limit"
                )
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame(
                columns=[column.name for column in query.selected_columns]
            )
        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    @contextmanager
    def _read_only_transaction(engine: Engine) -> Iterator[Any]:
        with engine.connect() as connection, connection.begin():
            # Some managed poolers ignore PostgreSQL startup options. Enforce the
            # invariant again inside every transaction that touches customer data.
            connection.execute(text("SET TRANSACTION READ ONLY"))
            yield connection

    def test(self) -> float:
        started = time.monotonic()
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise ConnectorError("PostgreSQL connection failed") from exc
        finally:
            engine.dispose()
        return (time.monotonic() - started) * 1000

    def discover(self, database_id: str) -> SchemaGraph:
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                return self._discover(connection, database_id)
        except SQLAlchemyError as exc:
            raise ConnectorError("PostgreSQL schema discovery failed") from exc
        finally:
            engine.dispose()

    def _discover(self, connection: Any, database_id: str) -> SchemaGraph:
        inspector = inspect(connection)
        available = inspector.get_table_names(schema=self.config.schema_name)
        selected = self.config.tables or sorted(available)
        missing = sorted(set(selected).difference(available))
        if missing:
            raise ConnectorError(f"configured tables do not exist: {missing}")
        if len(selected) > self.settings.max_tables:
            raise SnapshotLimitError(
                f"schema has {len(selected)} selected tables; limit is "
                f"{self.settings.max_tables}"
            )

        tables: list[TableSchema] = []
        for table_name in selected:
            columns_raw = inspector.get_columns(
                table_name, schema=self.config.schema_name
            )
            column_names = {column["name"] for column in columns_raw}
            time_column = self.config.time_columns.get(table_name)
            if time_column is not None and time_column not in column_names:
                raise ConnectorError(
                    f"time column {table_name}.{time_column} does not exist"
                )
            if time_column is not None:
                raw = next(c for c in columns_raw if c["name"] == time_column)
                if not isinstance(raw["type"], (Date, DateTime)):
                    raise ConnectorError(
                        f"time column {table_name}.{time_column} is not temporal"
                    )
            pk = (
                inspector.get_pk_constraint(
                    table_name, schema=self.config.schema_name
                ).get("constrained_columns")
                or []
            )
            fks = []
            for fk in inspector.get_foreign_keys(
                table_name, schema=self.config.schema_name
            ):
                referred_schema = fk.get("referred_schema")
                if referred_schema not in (None, self.config.schema_name):
                    continue
                fks.append(
                    ForeignKeySchema(
                        name=fk.get("name"),
                        columns=fk.get("constrained_columns") or [],
                        referred_table=fk["referred_table"],
                        referred_columns=fk.get("referred_columns") or [],
                    )
                )
            tables.append(
                TableSchema(
                    name=table_name,
                    columns=[
                        ColumnSchema(
                            name=column["name"],
                            data_type=str(column["type"]),
                            nullable=bool(column.get("nullable", True)),
                        )
                        for column in columns_raw
                    ],
                    primary_key=list(pk),
                    foreign_keys=fks,
                    time_column=time_column,
                )
            )
        return SchemaGraph(
            database_id=database_id,
            schema_name=self.config.schema_name,
            tables=tables,
            capabilities=ConnectorCapabilities(),
            discovered_at=datetime.now(timezone.utc),
        )

    def snapshot(self, database_id: str, as_of: datetime) -> Snapshot:
        engine = self._engine()
        try:
            with (
                engine.connect().execution_options(
                    isolation_level="REPEATABLE READ"
                ) as connection,
                connection.begin(),
            ):
                connection.execute(text("SET TRANSACTION READ ONLY"))
                schema = self._discover(connection, database_id)
                metadata = MetaData(schema=self.config.schema_name)
                frames: dict[str, pd.DataFrame] = {}
                total_rows = 0
                total_bytes = 0
                for table_schema in schema.tables:
                    table = Table(
                        table_schema.name,
                        metadata,
                        autoload_with=connection,
                    )
                    query = select(table)
                    if table_schema.time_column is not None:
                        query = query.where(table.c[table_schema.time_column] <= as_of)
                    query = query.limit(self.settings.max_rows_per_table + 1)
                    frame = self._read_bounded(query, connection)
                    if len(frame) > self.settings.max_rows_per_table:
                        raise SnapshotLimitError(
                            f"table {table_schema.name!r} exceeds the row limit "
                            f"of {self.settings.max_rows_per_table}"
                        )
                    total_rows += len(frame)
                    if total_rows > self.settings.max_total_rows:
                        raise SnapshotLimitError(
                            f"snapshot exceeds the total row limit of "
                            f"{self.settings.max_total_rows}"
                        )
                    total_bytes += int(frame.memory_usage(index=True, deep=True).sum())
                    if total_bytes > self.settings.max_snapshot_bytes:
                        raise SnapshotLimitError(
                            "snapshot exceeds the configured memory limit"
                        )
                    frames[table_schema.name] = frame
        except ConnectorError:
            raise
        except SQLAlchemyError as exc:
            raise ConnectorError("PostgreSQL snapshot failed") from exc
        finally:
            engine.dispose()

        foreign_keys: list[tuple[str, str, str, str]] = []
        primary_keys: dict[str, list[str]] = {}
        for table in schema.tables:
            primary_keys[table.name] = table.primary_key
            for foreign_key in table.foreign_keys:
                if len(foreign_key.columns) == len(foreign_key.referred_columns) == 1:
                    foreign_keys.append(
                        (
                            table.name,
                            foreign_key.columns[0],
                            foreign_key.referred_table,
                            foreign_key.referred_columns[0],
                        )
                    )
        return Snapshot(
            tables=frames,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=dict(self.config.time_columns),
            schema=schema,
            as_of=as_of,
            approximate_bytes=total_bytes,
        )
