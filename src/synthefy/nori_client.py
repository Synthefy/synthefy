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
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import httpx
import numpy as np
from pydantic import BaseModel

from synthefy.api_client import (
    APIConnectionError,
    APITimeoutError,
    _raise_for_status,
)

# Gateway endpoint (default): routes to the model by name, body carries "model".
GATEWAY_BASE_URL = "https://inference.baseten.co"
GATEWAY_ENDPOINT = "/predict"
GATEWAY_MODEL = "synthefy/nori"

# Dedicated endpoint: a specific production deployment; body carries no "model".
# To target it, pass base_url/endpoint to the constructor and set model=None.
DEDICATED_BASE_URL = "https://model-3m5j7y9w.api.baseten.co"
DEDICATED_ENDPOINT = "/environments/production/predict"

DEFAULT_TASK = "regression"

Mode = Literal["remote", "local", "auto"]
_VALID_MODES = ("remote", "local", "auto")

# Authorization header scheme for remote requests. The Baseten inference
# *gateway* accepts only ``Bearer``; dedicated deployments use ``Api-Key``.
AuthScheme = Literal["Bearer", "Api-Key"]
_VALID_AUTH_SCHEMES = ("Bearer", "Api-Key")
DEFAULT_AUTH_SCHEME: AuthScheme = "Bearer"

# Array-like inputs accepted by ``predict`` -- nested Python sequences or numpy arrays.
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]
VectorLike = Union[Sequence[float], np.ndarray]


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


def _coerce_matrix(arr: MatrixLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 2D float ``np.ndarray`` or raise ``ValueError``."""
    try:
        matrix = np.asarray(arr, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name} must be a numeric 2D array/list with equal-length rows; "
            f"got error: {exc}"
        ) from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be 2D with shape (n_rows, n_features); "
            f"got {matrix.ndim}D with shape {matrix.shape}"
        )
    return matrix


def _coerce_vector(arr: VectorLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 1D float ``np.ndarray`` or raise ``ValueError``."""
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


def _build_nori_request(
    X_train: MatrixLike,
    y_train: VectorLike,
    X_test: MatrixLike,
    task: str = DEFAULT_TASK,
) -> NoriPredictRequest:
    """Validate shapes and build a :class:`NoriPredictRequest`.

    Accepts Python lists or numpy arrays. Raises ``ValueError`` on any shape
    mismatch before a request leaves the process.
    """
    X_train_arr = _coerce_matrix(X_train, "X_train")
    X_test_arr = _coerce_matrix(X_test, "X_test")
    y_train_arr = _coerce_vector(y_train, "y_train")

    n_context, n_features = X_train_arr.shape
    if n_context == 0:
        raise ValueError("X_train must contain at least one context row")
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
        Model identifier included in the request body (remote mode). Required by
        the gateway; set to ``None`` for dedicated deployments.
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
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> List[float]:
        """Predict a value for each query row via in-context regression.

        Parameters
        ----------
        X_train : array-like of shape (n_context, n_features)
            Labeled context rows. Python lists or numpy arrays are accepted.
        y_train : array-like of shape (n_context,)
            Target value for each context row.
        X_test : array-like of shape (n_query, n_features)
            Query rows to predict. Must have the same number of features as
            ``X_train``.
        task : str, default "regression"
            The prediction task. Currently only ``"regression"`` is supported.
        timeout : float or None, optional
            Override the client timeout for this request (remote mode only;
            ignored in local mode).
        extra_headers : dict of str to str, optional
            Additional HTTP headers to send with the request (remote mode only;
            ignored in local mode).

        Returns
        -------
        List[float]
            One predicted value per row of ``X_test``.

        Raises
        ------
        ValueError
            If the input shapes are inconsistent (e.g. ``X_train`` and
            ``y_train`` row counts differ, or ``X_test`` has a different number
            of features than ``X_train``).
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
        request = _build_nori_request(X_train, y_train, X_test, task)
        if self.mode == "local":
            return self._predict_local(request)
        return self._predict_remote(
            request, timeout=timeout, extra_headers=extra_headers
        )

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
