"""Resolve database credentials only inside the caller's process."""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.engine import URL, make_url

from synthefy.relational.models import AwsSecret, CredentialReference, DatabaseUrl


class CredentialResolutionError(RuntimeError):
    pass


class CredentialResolver:
    """Resolve a URL without caching, serializing, or returning its secret value."""

    def resolve(self, reference: CredentialReference) -> URL:
        if isinstance(reference, DatabaseUrl):
            raw = reference.value.get_secret_value()
        elif isinstance(reference, AwsSecret):
            try:
                client = boto3.client(
                    "secretsmanager", region_name=reference.region_name
                )
                response = client.get_secret_value(SecretId=reference.secret_id)
            except (BotoCoreError, ClientError) as exc:
                raise CredentialResolutionError(
                    "database secret could not be resolved"
                ) from exc
            raw = response.get("SecretString")
            if not raw:
                raise CredentialResolutionError("database secret has no SecretString")
        else:
            raw = os.getenv(reference.variable)
            if not raw:
                raise CredentialResolutionError(
                    f"database environment variable {reference.variable!r} is not set"
                )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> URL:
        if raw.lstrip().startswith("{"):
            try:
                data: dict[str, Any] = json.loads(raw)
                url = URL.create(
                    drivername="postgresql+psycopg",
                    username=data["username"],
                    password=data["password"],
                    host=data["host"],
                    port=int(data.get("port", 5432)),
                    database=data.get("dbname") or data["database"],
                    query={"sslmode": data.get("sslmode", "require")},
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CredentialResolutionError(
                    "database secret must be a PostgreSQL URL or RDS credential JSON"
                ) from exc
        else:
            try:
                url = make_url(raw)
            except Exception as exc:
                raise CredentialResolutionError("database URL is invalid") from exc
        if not url.drivername.startswith("postgresql"):
            raise CredentialResolutionError("only PostgreSQL is supported in version 1")
        if url.drivername != "postgresql+psycopg":
            url = url.set(drivername="postgresql+psycopg")
        return url
