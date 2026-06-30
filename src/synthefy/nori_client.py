"""Standalone client for Synthefy Nori in-context regression.

Synthefy Nori is an in-context learning regressor: each call supplies labeled
context rows (``X_train``, ``y_train``) and query rows (``X_test``), and the model
returns one predicted value per query row in a single forward pass -- there is no
training step.

This module is self-contained and does not depend on the forecasting client
(:class:`synthefy.api_client.SynthefyAPIClient`). It does, however, reuse the
package-wide exception types and HTTP error handling from
:mod:`synthefy.api_client` so that errors behave consistently across the SDK.

A single :class:`SynthefyNoriClient` runs predictions in one of three modes,
selected with the ``mode`` constructor argument:

- ``"remote"`` (default) -- calls the hosted Baseten endpoint over HTTPS.
- ``"local"`` -- runs the same prediction in-process via the optional
  ``synthefy-nori`` package (``pip install "synthefy[local]"``), no network and
  no API key.
- ``"auto"`` -- use ``"local"`` if the ``synthefy-nori`` package is installed,
  otherwise fall back to ``"remote"``.
"""

import importlib.util
import os
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel

from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    _raise_for_status,
)

# Gateway endpoint (default): routes to the model by name, body carries "model".
GATEWAY_BASE_URL = "https://inference.baseten.co"
GATEWAY_ENDPOINT = "/predict"

# Nori model family — Baseten gateway slugs. The gateway routes a request to a
# model by this slug (sent in the request body as "model") and meters usage
# under it. Today the family has a single public model; more flavours (e.g.
# larger or reasoning variants) are planned and will be added here as they ship.
# "synthefy/nori-dev" is an internal development deployment and is intentionally
# not listed. ``model=`` still accepts any string, so a newly published slug
# works before this client is updated.
NORI = "synthefy/nori"  # general-purpose tabular in-context regressor
KNOWN_NORI_MODELS = (NORI,)

# Default model slug for remote (gateway) requests.
GATEWAY_MODEL = NORI

# Dedicated endpoint: a specific production deployment; body carries no "model".
# To target it, pass base_url/endpoint to the constructor and set model=None.
DEDICATED_BASE_URL = "https://model-qrj00rr3.api.baseten.co"
DEDICATED_ENDPOINT = "/environments/production/predict"

DEFAULT_TASK = "regression"

Mode = Literal["remote", "local", "auto"]
_VALID_MODES = ("remote", "local", "auto")

# Authorization header scheme for remote requests. The Baseten inference
# *gateway* accepts only ``Bearer``; dedicated deployments use ``Api-Key``.
AuthScheme = Literal["Bearer", "Api-Key"]
_VALID_AUTH_SCHEMES = ("Bearer", "Api-Key")
DEFAULT_AUTH_SCHEME: AuthScheme = "Bearer"

# Array-like inputs accepted by ``predict`` -- nested Python sequences, numpy
# arrays, or pandas DataFrames/Series (all coerced to plain numeric arrays).
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray, pd.DataFrame]
VectorLike = Union[Sequence[float], np.ndarray, pd.Series, pd.DataFrame]


class NoriPredictRequest(BaseModel):
    """Request payload for a Synthefy Nori prediction.

    Mirrors the hosted inference contract exactly::

        {"X_train": [[...], ...], "y_train": [...], "X_test": [[...], ...],
         "task": "regression"}

    Parameters
    ----------
    X_train : List[List[float]]
        Labeled context rows. Shape ``(n_context, n_features)``.
    y_train : List[float]
        Target value for each context row. Length ``n_context``.
    X_test : List[List[float]]
        Query rows to predict. Shape ``(n_query, n_features)``.
    task : str, default "regression"
        The prediction task. Currently only ``"regression"`` is supported.
    """

    X_train: List[List[float]]
    y_train: List[float]
    X_test: List[List[float]]
    task: str = DEFAULT_TASK


class NoriPredictResponse(BaseModel):
    """Response payload from a Synthefy Nori prediction.

    Parameters
    ----------
    task : str
        The task echoed back by the server (e.g. ``"regression"``).
    predictions : List[float]
        One predicted value per row of ``X_test``.
    """

    task: str
    predictions: List[float]


def _frame_columns(arr: Any) -> Optional[List[Any]]:
    """Return a DataFrame's column labels, or ``None`` if ``arr`` is not one.

    Used to align ``X_test`` to ``X_train`` by column name when both inputs are
    pandas DataFrames (see :func:`_build_nori_request`).
    """
    if isinstance(arr, pd.DataFrame):
        return list(arr.columns)
    return None


def _target_name(y_train: Any) -> Any:
    """Name for the ``as_pandas`` output Series, taken from ``y_train``.

    Uses the ``Series.name`` or the single-column ``DataFrame``'s column label;
    falls back to ``"prediction"`` when ``y_train`` carries no name (lists/arrays).
    """
    if isinstance(y_train, pd.Series):
        return y_train.name if y_train.name is not None else "prediction"
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        return y_train.columns[0]
    return "prediction"


def _result_index(X_test: Any) -> Optional[Any]:
    """Index for the ``as_pandas`` output, copied from ``X_test`` when it is a
    pandas object so predictions join straight back; ``None`` (default RangeIndex)
    otherwise."""
    if isinstance(X_test, (pd.DataFrame, pd.Series)):
        return X_test.index
    return None


def _reject_non_numeric_columns(frame: pd.DataFrame, name: str) -> None:
    """Raise ``ValueError`` if any column is not numeric.

    Nori is a numeric-only model, so categorical/text/datetime columns must be
    encoded by the caller before prediction. ``bool`` and integer columns are
    treated as numeric.
    """
    non_numeric = [
        str(col)
        for col in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        raise ValueError(
            f"{name} has non-numeric column(s) {non_numeric}; Nori is a "
            "numeric-only model. Encode categorical/text/datetime columns "
            "(e.g. one-hot or ordinal encoding) before calling predict()."
        )


def _coerce_matrix(arr: MatrixLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 2D float ``np.ndarray`` or raise ``ValueError``.

    Accepts nested Python sequences, numpy arrays, and pandas DataFrames. A
    pandas DataFrame is checked for non-numeric columns first (so the caller
    gets a clear message rather than a cryptic float-cast error). NaN/missing
    values are preserved and forwarded for server-side imputation.
    """
    if isinstance(arr, pd.DataFrame):
        _reject_non_numeric_columns(arr, name)
        matrix = arr.to_numpy(dtype=float)
    else:
        try:
            matrix = np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{name} must be a numeric 2D array/list with equal-length rows; "
                f"got error: {exc}. If it has categorical/string columns, pass a "
                "pandas DataFrame (with both X_train and X_test as DataFrames) so "
                "they can be one-hot encoded."
            ) from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be 2D with shape (n_rows, n_features); "
            f"got {matrix.ndim}D with shape {matrix.shape}"
        )
    return matrix


def _coerce_vector(arr: VectorLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 1D float ``np.ndarray`` or raise ``ValueError``.

    Accepts nested Python sequences, numpy arrays, a pandas Series, or a
    single-column pandas DataFrame. NaN/missing values are preserved and
    forwarded for server-side imputation.
    """
    if isinstance(arr, pd.DataFrame):
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} must be 1D; got a DataFrame with {arr.shape[1]} "
                "columns. Pass a single column (a Series) for the targets."
            )
        _reject_non_numeric_columns(arr, name)
        vector = arr.to_numpy(dtype=float).reshape(-1)
    elif isinstance(arr, pd.Series):
        if not pd.api.types.is_numeric_dtype(arr):
            raise ValueError(
                f"{name} must be numeric; got a non-numeric Series "
                f"(dtype {arr.dtype})."
            )
        vector = arr.to_numpy(dtype=float)
    else:
        try:
            vector = np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{name} must be a numeric 1D array/list; got error: {exc}"
            ) from exc
    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be 1D with shape (n_rows,); "
            f"got {vector.ndim}D with shape {vector.shape}"
        )
    return vector


# Default cap on a categorical column's distinct values before one-hot encoding.
# Columns above this are dropped (with a warning) rather than exploding the
# feature matrix — matches the model repo's offline evaluator, which drops
# string columns with >100 unique values.
_DEFAULT_MAX_CARDINALITY = 100


def _has_encodable_columns(frame: pd.DataFrame) -> bool:
    """``True`` if any column is non-numeric (so featurization is needed)."""
    return any(
        not pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns
    )


def _numeric_categories_to_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert ``category`` columns whose categories are numeric back to a plain
    numeric dtype, so they are kept as magnitudes rather than one-hot exploded
    (``is_numeric_dtype`` is ``False`` for any ``category`` dtype, even integer
    ones). Returns ``frame`` unchanged — no copy — when there is nothing to
    convert.
    """
    out = frame
    for col in frame.columns:
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
            s.cat.categories
        ):
            if out is frame:
                out = frame.copy()
            # cast to float (not the categories' dtype) so a missing value in an
            # *integer*-category column promotes to NaN instead of raising
            # "Cannot convert NaN to integer".
            out[col] = s.astype("float64")
    return out


def _featurize_frames(
    X_train: pd.DataFrame, X_test: pd.DataFrame, max_cardinality: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode non-numeric columns of two aligned frames (fit on train).

    Numeric columns (including ``bool``, and ``category`` columns whose
    categories are numeric) pass through unchanged. Datetime columns, columns
    with no non-missing values, and categorical columns with more than
    ``max_cardinality`` distinct *training* values are dropped with a
    ``UserWarning``; ``timedelta`` columns are unsupported and raise. Categories
    come from ``X_train``: a value seen only in ``X_test`` maps to an all-zeros
    indicator group, a category absent from ``X_test`` is still emitted as a zero
    column, and a missing value (in either frame) gets its own indicator column
    (``dummy_na=True``, but the indicator is dropped when no row is missing) — so
    both frames come out with identical numeric columns and the server receives a
    fully model-ready matrix (no reliance on server-side category detection).

    A column that is numeric in one frame but not the other raises ``ValueError``
    (rather than failing later with a confusing message). ``X_train`` and
    ``X_test`` must already share the same columns (callers align them by name
    first). Row order and count are preserved.
    """
    # category-of-numeric -> plain numeric, so it is kept as a magnitude (not
    # one-hot exploded). Applied to both frames before any dtype inspection.
    X_train = _numeric_categories_to_values(X_train)
    X_test = _numeric_categories_to_values(X_test)

    # A column must be the same kind (numeric vs not) in both frames; otherwise
    # featurization would silently mis-handle it (or crash later in the float
    # cast with a misleading message). Fail loud and specific instead.
    mismatched = [
        col
        for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col])
        != pd.api.types.is_numeric_dtype(X_test[col])
    ]
    if mismatched:
        raise ValueError(
            f"Column(s) {mismatched} are numeric in one of X_train/X_test but "
            "not the other; X_train and X_test must have matching column types "
            "(a common cause is object-dtype numbers, e.g. from read_csv — cast "
            "them with pd.to_numeric first)."
        )

    numeric_cols: List[Any] = []
    cat_cols: List[Any] = []
    dropped: List[str] = []
    for col in X_train.columns:
        s = X_train[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dropped.append(f"{col!r} (datetime)")
        elif pd.api.types.is_timedelta64_dtype(s) or isinstance(
            s.dtype, pd.PeriodDtype
        ):
            raise ValueError(
                f"Column {col!r} has unsupported dtype {s.dtype}; convert it to a "
                "number (e.g. .dt.total_seconds()) or a string before calling "
                "predict()."
            )
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique == 0:
                dropped.append(f"{col!r} (no non-missing values)")
            elif n_unique > max_cardinality:
                dropped.append(f"{col!r} (>{max_cardinality} unique values)")
            else:
                cat_cols.append(col)

    if dropped:
        warnings.warn(
            "Nori one-hot featurization dropped non-encodable column(s): "
            + ", ".join(dropped)
            + ". Encode them yourself (e.g. target/hash encoding) if you need them.",
            stacklevel=4,
        )

    if cat_cols:
        # dummy_na=True gives missing values their own indicator column, so NaN is
        # a distinct category rather than silently all-zeros. get_dummies always
        # emits a NaN column per categorical; drop columns that are all-zero in
        # TRAIN — that removes the dead NaN-indicator when a column has no missing
        # rows (so a legitimate literal "nan" value column doesn't collide with
        # it). Done positionally so a transient duplicate label can't break it.
        train_d = pd.get_dummies(
            X_train[cat_cols].astype(object), columns=cat_cols,
            dummy_na=True, dtype=np.uint8,
        )
        train_d = train_d.loc[:, (train_d.to_numpy() != 0).any(axis=0)]
        if train_d.columns.has_duplicates:
            raise ValueError(
                "One-hot encoding produced duplicate column names — a column name "
                "and value collide under '<column>_<value>' naming. Rename the "
                "offending column(s) before calling predict()."
            )
        test_d = pd.get_dummies(
            X_test[cat_cols].astype(object), columns=cat_cols,
            dummy_na=True, dtype=np.uint8,
        )
        # Drop test all-zero columns too (e.g. the dummy_na column when X_test has
        # no missing rows) so test_d has no duplicate label before reindex.
        test_d = test_d.loc[:, (test_d.to_numpy() != 0).any(axis=0)]
        test_d = test_d.reindex(columns=train_d.columns, fill_value=0)
        X_train_feat = pd.concat(
            [X_train[numeric_cols].reset_index(drop=True),
             train_d.reset_index(drop=True)],
            axis=1,
        )
        X_test_feat = pd.concat(
            [X_test[numeric_cols].reset_index(drop=True),
             test_d.reset_index(drop=True)],
            axis=1,
        )
        if X_train_feat.columns.has_duplicates:
            raise ValueError(
                "Featurized columns are not unique — a numeric column name "
                "collides with a generated one-hot column name. Rename the "
                "offending column(s) before calling predict()."
            )
    else:
        X_train_feat = X_train[numeric_cols].reset_index(drop=True)
        X_test_feat = X_test[numeric_cols].reset_index(drop=True)

    if X_train_feat.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after one-hot featurization (every "
            "column was dropped — temporal, all-missing, or above the "
            f"max_categorical_cardinality={max_cardinality} cap)."
        )
    return X_train_feat, X_test_feat


def _build_nori_request(
    X_train: MatrixLike,
    y_train: VectorLike,
    X_test: MatrixLike,
    task: str = DEFAULT_TASK,
    max_categorical_cardinality: int = _DEFAULT_MAX_CARDINALITY,
) -> NoriPredictRequest:
    """Validate shapes and build a :class:`NoriPredictRequest`.

    Accepts Python lists, numpy arrays, or pandas DataFrames/Series. When both
    ``X_train`` and ``X_test`` are DataFrames, ``X_test`` is aligned to
    ``X_train``'s columns *by name* (so column order is irrelevant), and a
    mismatch in the column sets raises ``ValueError``; then any non-numeric
    columns are one-hot encoded (fit on ``X_train``, applied to ``X_test``) so
    the request carries a fully numeric matrix. Otherwise columns are matched
    positionally, as before. Raises ``ValueError`` on any shape mismatch before a
    request leaves the process. NaN/missing values are preserved and imputed
    server-side.
    """
    if max_categorical_cardinality < 1:
        raise ValueError(
            "max_categorical_cardinality must be a positive integer; got "
            f"{max_categorical_cardinality}."
        )
    train_cols = _frame_columns(X_train)
    test_cols = _frame_columns(X_test)
    if (train_cols is None) != (test_cols is None):
        # one side is a DataFrame, the other isn't: we can't one-hot/align by
        # name. Give a targeted error if the DataFrame side has columns to encode.
        df, df_name, other = (
            (X_train, "X_train", "X_test")
            if train_cols is not None
            else (X_test, "X_test", "X_train")
        )
        if _has_encodable_columns(df):
            raise ValueError(
                f"{df_name} has non-numeric column(s) to one-hot encode, but "
                f"{other} is not a DataFrame; pass both X_train and X_test as "
                "DataFrames with the same columns so they can be aligned and "
                "encoded (or pre-encode to numeric)."
            )
    if train_cols is not None and test_cols is not None:
        for cols, nm in ((train_cols, "X_train"), (test_cols, "X_test")):
            idx = pd.Index(cols)
            if idx.has_duplicates:
                dups = sorted({str(c) for c in idx[idx.duplicated()]})
                raise ValueError(
                    f"{nm} has duplicate column name(s) {dups}; column names must "
                    "be unique (duplicates break by-name alignment and encoding)."
                )
        if set(train_cols) != set(test_cols):
            raise ValueError(
                "X_train and X_test must have the same feature columns; "
                f"X_train has {train_cols} but X_test has {test_cols}."
            )
        if train_cols != test_cols:
            # Same columns, different order: reorder X_test to match X_train so
            # the model sees features in a consistent position.
            X_test = X_test[train_cols]
        # One-hot encode any non-numeric columns into a fully numeric matrix,
        # fitting on X_train and applying the same layout to X_test. Only the
        # DataFrame/DataFrame case can do this (column names are required). Check
        # both frames so a column that is non-numeric in only one of them is
        # caught with a clear error rather than a later cryptic float-cast.
        if (
            len(X_train)
            and len(X_test)
            and (_has_encodable_columns(X_train) or _has_encodable_columns(X_test))
        ):
            X_train, X_test = _featurize_frames(
                X_train, X_test, max_categorical_cardinality
            )

    X_train_arr = _coerce_matrix(X_train, "X_train")
    X_test_arr = _coerce_matrix(X_test, "X_test")
    y_train_arr = _coerce_vector(y_train, "y_train")

    n_context, n_features = X_train_arr.shape
    if n_context == 0:
        raise ValueError("X_train must contain at least one context row")
    if n_features == 0:
        raise ValueError("X_train must contain at least one feature column")
    if y_train_arr.shape[0] != n_context:
        raise ValueError(
            f"X_train has {n_context} rows but y_train has "
            f"{y_train_arr.shape[0]}; they must match"
        )
    if X_test_arr.shape[0] == 0:
        raise ValueError("X_test must contain at least one query row")
    if X_test_arr.shape[1] != n_features:
        raise ValueError(
            f"X_test has {X_test_arr.shape[1]} features but X_train has "
            f"{n_features}; they must match"
        )
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    return NoriPredictRequest(
        X_train=X_train_arr.tolist(),
        y_train=y_train_arr.tolist(),
        X_test=X_test_arr.tolist(),
        task=task,
    )


def _as_float_list(values: Any) -> List[float]:
    """Coerce a prediction result (list or numpy array) into a flat ``list[float]``."""
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _local_available() -> bool:
    """Return ``True`` if the optional ``synthefy-nori`` package is installed.

    Uses ``find_spec`` so it does not import (and thus does not load) the package.
    """
    return importlib.util.find_spec("synthefy_nori") is not None


def _load_local_predict() -> Any:
    """Lazily import ``synthefy_nori.predict`` with a helpful error if absent."""
    try:
        from synthefy_nori import predict as local_predict
    except ImportError as exc:
        raise ImportError(
            "Local nori inference requires the optional 'synthefy-nori' "
            'package. Install it with: pip install "synthefy[local]".'
        ) from exc
    return local_predict


class SynthefyNoriClient:
    """Client for Synthefy Nori in-context regression.

    Each :meth:`predict` call performs in-context regression: the labeled context
    rows are supplied alongside the query rows and one value per query row is
    returned in a single forward pass.

    The ``mode`` argument selects how predictions run:

    - ``"remote"`` (default): call the hosted Baseten endpoint over HTTPS.
      Requires a Baseten API key (``api_key`` argument or ``BASETEN_API_KEY``
      environment variable), sent as ``Authorization: <auth_scheme> <key>``
      (``Bearer`` by default).
    - ``"local"``: run in-process via the optional ``synthefy-nori`` package
      (``pip install "synthefy[local]"``). No network and no API key.
    - ``"auto"``: use ``"local"`` if ``synthefy-nori`` is installed, otherwise
      fall back to ``"remote"`` (which then requires an API key).

    For remote mode, the client targets the Baseten inference *gateway* by default
    (``https://inference.baseten.co/predict``) and includes
    ``"model": "synthefy/nori"`` in the request body. The gateway authenticates
    with the ``Bearer`` scheme (the default ``auth_scheme``). To target a
    dedicated deployment instead, pass ``base_url=DEDICATED_BASE_URL``,
    ``endpoint=DEDICATED_ENDPOINT``, ``model=None`` and ``auth_scheme="Api-Key"``
    (the dedicated endpoint takes the body verbatim with no ``model`` field and
    authenticates with the ``Api-Key`` scheme).

    Parameters
    ----------
    api_key : str or None, optional
        Baseten API key (remote mode only). If ``None``, falls back to the
        ``BASETEN_API_KEY`` environment variable. A :class:`ValueError` is raised
        if neither is set when remote mode is in effect.
    mode : {"remote", "local", "auto"}, default "remote"
        How predictions run. See above.
    timeout : float, default 300.0
        Per-request timeout in seconds (remote mode).
    max_retries : int, default 2
        Number of retries for transient errors (timeouts, connection errors,
        429 and 5xx responses) with exponential backoff (remote mode).
    base_url : str, default GATEWAY_BASE_URL
        Base URL of the inference host (remote mode).
    endpoint : str, default GATEWAY_ENDPOINT
        Path appended to ``base_url`` for predictions (remote mode).
    model : str or None, default GATEWAY_MODEL
        Nori model slug included in the request body (remote mode); see
        ``KNOWN_NORI_MODELS`` for the recognized slugs (currently just
        ``"synthefy/nori"``). Any string is accepted, so a newly published model
        works before this client is updated. Required by the gateway; set to
        ``None`` for dedicated deployments.
    auth_scheme : {"Bearer", "Api-Key"}, default "Bearer"
        HTTP ``Authorization`` scheme prefixed to the API key (remote mode). The
        inference gateway requires ``"Bearer"``; dedicated deployments use
        ``"Api-Key"``.
    user_agent : str or None, optional
        Custom ``User-Agent`` header (remote mode).

    Attributes
    ----------
    mode : str
        The resolved mode (``"auto"`` is resolved to ``"local"`` or ``"remote"``
        at construction).

    Examples
    --------
    >>> from synthefy import SynthefyNoriClient
    >>> client = SynthefyNoriClient(api_key="...")  # or BASETEN_API_KEY
    >>> preds = client.predict(
    ...     X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    ...     y_train=[1.0, 1.0, 2.0],
    ...     X_test=[[2.0, 2.0]],
    ... )
    >>> len(preds)
    1

    Run the same prediction locally (no API key, needs ``synthefy[local]``):

    >>> client = SynthefyNoriClient(mode="local")  # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        mode: Mode = "remote",
        timeout: float = 300.0,
        max_retries: int = 2,
        base_url: str = GATEWAY_BASE_URL,
        endpoint: str = GATEWAY_ENDPOINT,
        model: Optional[str] = GATEWAY_MODEL,
        auth_scheme: AuthScheme = DEFAULT_AUTH_SCHEME,
        user_agent: Optional[str] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}; got {mode!r}"
            )
        if auth_scheme not in _VALID_AUTH_SCHEMES:
            raise ValueError(
                f"auth_scheme must be one of {_VALID_AUTH_SCHEMES}; "
                f"got {auth_scheme!r}"
            )
        if mode == "auto":
            mode = "local" if _local_available() else "remote"
        self.mode: str = mode

        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.endpoint = endpoint
        self.model = model
        self.auth_scheme = auth_scheme
        self.user_agent = (
            user_agent or f"synthefy-python httpx/{httpx.__version__}"
        )

        if mode == "remote":
            if api_key is None:
                api_key = os.getenv("BASETEN_API_KEY")
            if not api_key:
                raise ValueError(
                    "A Baseten API key must be provided either as the `api_key` "
                    "argument or through the BASETEN_API_KEY environment variable "
                    "when mode='remote'"
                )
            self.api_key: Optional[str] = api_key
            self.client: Optional[httpx.Client] = httpx.Client(
                base_url=self.base_url
            )
        else:  # local
            self.api_key = api_key  # unused in local mode; may be None
            self.client = None

    # Context manager support (sync) and utilities
    def __enter__(self) -> "SynthefyNoriClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def predict(
        self,
        X_train: MatrixLike,
        y_train: VectorLike,
        X_test: MatrixLike,
        task: str = DEFAULT_TASK,
        *,
        as_pandas: bool = False,
        max_categorical_cardinality: int = _DEFAULT_MAX_CARDINALITY,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Union[List[float], pd.Series]:
        """Predict a value for each query row via in-context regression.

        Parameters
        ----------
        X_train : array-like of shape (n_context, n_features)
            Labeled context rows. Python lists, numpy arrays, or a pandas
            DataFrame are accepted. In the DataFrame/DataFrame case, non-numeric
            columns are one-hot encoded for you (see ``X_test`` and
            ``max_categorical_cardinality``); otherwise all columns must be
            numeric. Missing values are allowed: NaN in a numeric column is
            imputed server-side; NaN in a categorical column becomes its own
            one-hot indicator.
        y_train : array-like of shape (n_context,)
            Target value for each context row. A Python list, numpy array, or a
            pandas Series / single-column DataFrame is accepted.
        X_test : array-like of shape (n_query, n_features)
            Query rows to predict. Must have the same number of features as
            ``X_train``. When both ``X_train`` and ``X_test`` are DataFrames,
            ``X_test`` is aligned to ``X_train``'s columns *by name* (column
            order is irrelevant; a mismatch in the column sets raises), and any
            non-numeric columns are **one-hot encoded** — fit on ``X_train`` and
            applied to ``X_test`` — into a fully numeric matrix. Categories come
            from ``X_train``: a value seen only in ``X_test`` becomes an
            all-zeros indicator group, and a missing value gets its own
            indicator column. Datetime columns and categorical columns with more
            than ``max_categorical_cardinality`` distinct training values are
            dropped with a warning; ``timedelta`` columns are unsupported and
            raise (convert them to a number or string first).
        task : str, default "regression"
            The prediction task. Currently only ``"regression"`` is supported.
        as_pandas : bool, default False
            If ``True``, return a pandas ``Series`` instead of a list: one value
            per ``X_test`` row, named after ``y_train`` (its ``Series`` name or
            single-column ``DataFrame`` label, else ``"prediction"``) and indexed
            by ``X_test``'s index when ``X_test`` is a pandas object (so the
            predictions join straight back). Default is the plain ``list``.
        max_categorical_cardinality : int, default 100
            Maximum number of distinct training values a non-numeric column may
            have to be one-hot encoded (DataFrame inputs only). Columns above
            this cap are dropped with a warning instead of exploding the feature
            matrix. Ignored when inputs are already numeric.
        timeout : float or None, optional
            Override the client timeout for this request (remote mode only;
            ignored in local mode).
        extra_headers : dict of str to str, optional
            Additional HTTP headers to send with the request (remote mode only;
            ignored in local mode).

        Returns
        -------
        list of float, or pandas.Series if ``as_pandas=True``
            One predicted value per row of ``X_test``.

        Raises
        ------
        ValueError
            If the input shapes are inconsistent (e.g. ``X_train`` and
            ``y_train`` row counts differ, or ``X_test`` has a different number
            of features than ``X_train``); if DataFrame ``X_train``/``X_test``
            have mismatched column sets or duplicate column names; if a column is
            numeric in one of ``X_train``/``X_test`` but not the other; if a
            column has unsupported ``timedelta`` dtype; if a non-DataFrame input
            contains non-numeric values; or if one-hot featurization leaves no
            usable columns.
        ImportError
            In local mode, if the optional ``synthefy-nori`` package is not
            installed (with guidance to ``pip install "synthefy[local]"``).
        BadRequestError
            In remote mode, if the server rejects the request (HTTP 400),
            carrying the server's ``error`` string as the message.
        AuthenticationError
            In remote mode, if the API key is missing or invalid (HTTP 401).
        APITimeoutError
            In remote mode, if the request times out.
        APIConnectionError
            In remote mode, if a network/connection error occurs.
        """
        request = _build_nori_request(
            X_train, y_train, X_test, task,
            max_categorical_cardinality=max_categorical_cardinality,
        )
        if self.mode == "local":
            predictions = self._predict_local(request)
        else:
            predictions = self._predict_remote(
                request, timeout=timeout, extra_headers=extra_headers
            )
        if as_pandas:
            return pd.Series(
                predictions,
                index=_result_index(X_test),
                name=_target_name(y_train),
                dtype=float,
            )
        return predictions

    # ------------------------------------------------------------------ #
    # Local mode
    # ------------------------------------------------------------------ #

    def _predict_local(self, request: NoriPredictRequest) -> List[float]:
        local_predict = _load_local_predict()
        result = local_predict(
            request.X_train,
            request.y_train,
            request.X_test,
            task=request.task,
        )
        return _as_float_list(result)

    # ------------------------------------------------------------------ #
    # Remote mode
    # ------------------------------------------------------------------ #

    def _predict_remote(
        self,
        request: NoriPredictRequest,
        *,
        timeout: Optional[float],
        extra_headers: Optional[Dict[str, str]],
    ) -> List[float]:
        payload = request.model_dump()
        if self.model is not None:
            payload["model"] = self.model

        response = self._post_with_retries(
            self.endpoint,
            json=payload,
            headers=self._headers(extra_headers=extra_headers),
            timeout=timeout,
        )
        parsed = NoriPredictResponse(**response.json())
        return parsed.predictions

    def _headers(
        self, *, extra_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Authorization": f"{self.auth_scheme} {self.api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _should_retry(
        self, response: Optional[httpx.Response], exc: Optional[Exception]
    ) -> bool:
        if exc is not None:
            # Connection errors/timeouts are retryable
            return True
        if response is None:
            return False
        if (
            response.status_code in (408, 409, 425, 429)
            or 500 <= response.status_code <= 599
        ):
            return True
        return False

    def _compute_backoff(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> float:
        if response is not None:
            retry_after = response.headers.get(
                "retry-after"
            ) or response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        # Exponential backoff with jitter
        base = min(2**attempt, 30)
        return base * (0.5 + 0.5 * (os.urandom(1)[0] / 255))

    def _post_with_retries(
        self,
        endpoint: str,
        json: Dict[str, Any],
        *,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        assert self.client is not None  # remote mode always has a client
        last_exc: Optional[Exception] = None
        response: Optional[httpx.Response] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            # Reset per attempt so the "no more retries" block below reflects the
            # final attempt only. Otherwise an exception from an earlier attempt
            # (e.g. a transient connection error) would be re-raised in place of
            # the true final error (e.g. a 5xx that should map to a server error).
            last_exc = None
            response = None
            try:
                response = self.client.post(
                    endpoint,
                    json=json,
                    headers=headers or self._headers(),
                    timeout=timeout or self.timeout,
                )
                if not self._should_retry(response, None):
                    _raise_for_status(response)
                    return response
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc))
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc))

            # Decide to retry
            if attempt < attempts - 1 and self._should_retry(
                response, last_exc
            ):
                delay = self._compute_backoff(attempt, response)
                time.sleep(delay)
                continue

            # No more retries
            if last_exc is not None:
                raise last_exc
            if response is not None:
                _raise_for_status(response)
                return response

        # Should not reach here
        raise APIConnectionError("Request failed after retries")
