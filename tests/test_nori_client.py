"""Unit tests for the Synthefy Nori client.

Remote mode is tested against a mocked httpx transport so these tests never hit
the network. The real local-inference test is marked ``slow`` and skips unless
the optional ``synthefy-nori`` package is installed.
"""

import builtins
import json
from typing import Callable, Dict, List

import httpx
import numpy as np
import pytest
from synthefy import (
    SynthefyNoriClient,
    NoriPredictRequest,
    NoriPredictResponse,
)
from synthefy.api_client import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
from synthefy.nori_client import (
    DEDICATED_BASE_URL,
    DEDICATED_ENDPOINT,
    GATEWAY_ENDPOINT,
    GATEWAY_MODEL,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _attach_mock(client: SynthefyNoriClient, handler: Handler) -> None:
    """Swap the client's httpx transport for an in-memory mock (no network)."""
    client.close()
    client.client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )


def _ok_handler(predictions: List[float], capture: Dict) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["path"] = request.url.path
        capture["headers"] = request.headers
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"task": "regression", "predictions": predictions}
        )

    return handler


# --------------------------------------------------------------------------- #
# Remote mode -- happy path
# --------------------------------------------------------------------------- #


def test_predict_returns_predictions_and_sends_expected_request():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([10.0, 20.0], capture))

    preds = client.predict(
        X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        y_train=[1.0, 1.0, 2.0],
        X_test=[[2.0, 2.0], [3.0, 3.0]],
    )

    assert preds == [10.0, 20.0]
    assert client.mode == "remote"
    # Gateway is the default: correct path + model field in the body.
    assert capture["path"] == GATEWAY_ENDPOINT
    # Gateway requires the Bearer scheme (the default auth_scheme).
    assert capture["headers"]["authorization"] == "Bearer test-key"
    assert capture["headers"]["content-type"] == "application/json"
    body = capture["body"]
    assert body["X_train"] == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    assert body["y_train"] == [1.0, 1.0, 2.0]
    assert body["X_test"] == [[2.0, 2.0], [3.0, 3.0]]
    assert body["task"] == "regression"
    assert body["model"] == GATEWAY_MODEL


def test_predict_accepts_numpy_arrays():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([42.0], capture))

    preds = client.predict(
        X_train=np.array([[0.0, 1.0], [1.0, 0.0]]),
        y_train=np.array([1.0, 2.0]),
        X_test=np.array([[2.0, 2.0]]),
    )

    assert preds == [42.0]
    # numpy inputs are serialized to plain JSON lists of floats.
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["y_train"] == [1.0, 2.0]


def test_dedicated_endpoint_config_omits_model_field():
    capture: Dict = {}
    client = SynthefyNoriClient(
        api_key="test-key",
        base_url=DEDICATED_BASE_URL,
        endpoint=DEDICATED_ENDPOINT,
        model=None,
        auth_scheme="Api-Key",
    )
    _attach_mock(client, _ok_handler([1.0], capture))

    preds = client.predict(
        X_train=[[0.0], [1.0]], y_train=[0.0, 1.0], X_test=[[2.0]]
    )

    assert preds == [1.0]
    assert capture["path"] == DEDICATED_ENDPOINT
    assert "model" not in capture["body"]
    # Dedicated deployments authenticate with the Api-Key scheme.
    assert capture["headers"]["authorization"] == "Api-Key test-key"


# --------------------------------------------------------------------------- #
# Mode selection / authentication / configuration
# --------------------------------------------------------------------------- #


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "env-key")
    client = SynthefyNoriClient()
    assert client.api_key == "env-key"
    assert client.mode == "remote"


def test_missing_api_key_raises_in_remote_mode(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Baseten API key"):
        SynthefyNoriClient()


def test_local_mode_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    client = SynthefyNoriClient(mode="local")
    assert client.mode == "local"
    assert client.client is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        SynthefyNoriClient(api_key="test-key", mode="nope")


def test_default_auth_scheme_is_bearer():
    client = SynthefyNoriClient(api_key="test-key")
    assert client.auth_scheme == "Bearer"


def test_auth_scheme_override_sets_authorization_header():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", auth_scheme="Api-Key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert capture["headers"]["authorization"] == "Api-Key test-key"


def test_invalid_auth_scheme_raises():
    with pytest.raises(ValueError, match="auth_scheme must be one of"):
        SynthefyNoriClient(api_key="test-key", auth_scheme="Token")


def test_auto_mode_falls_back_to_remote_when_package_absent(monkeypatch):
    # synthefy-nori is not installed in the test environment, so auto -> remote.
    monkeypatch.setattr(
        "synthefy.nori_client._local_available", lambda: False
    )
    client = SynthefyNoriClient(api_key="test-key", mode="auto")
    assert client.mode == "remote"


def test_auto_mode_uses_local_when_package_present(monkeypatch):
    monkeypatch.setattr(
        "synthefy.nori_client._local_available", lambda: True
    )
    client = SynthefyNoriClient(mode="auto")  # no key needed once local
    assert client.mode == "local"


def test_context_manager_closes_client():
    with SynthefyNoriClient(api_key="test-key") as client:
        assert isinstance(client, SynthefyNoriClient)
    assert client.client.is_closed


# --------------------------------------------------------------------------- #
# Shape validation (runs before any network call or local import)
# --------------------------------------------------------------------------- #


@pytest.fixture
def client() -> SynthefyNoriClient:
    # No transport is attached; valid inputs would fail, but these tests assert
    # that validation raises *before* any request is attempted.
    return SynthefyNoriClient(api_key="test-key")


def test_mismatched_train_rows_raises(client):
    with pytest.raises(ValueError, match="they must match"):
        client.predict(X_train=[[1.0], [2.0]], y_train=[1.0], X_test=[[3.0]])


def test_feature_count_mismatch_raises(client):
    with pytest.raises(ValueError, match="features"):
        client.predict(
            X_train=[[1.0, 2.0]], y_train=[1.0], X_test=[[3.0, 4.0, 5.0]]
        )


def test_non_2d_x_train_raises(client):
    with pytest.raises(ValueError, match="X_train must be 2D"):
        client.predict(X_train=[1.0, 2.0, 3.0], y_train=[1.0], X_test=[[3.0]])


def test_empty_x_train_raises(client):
    # A 2D array with zero rows reaches the row-count guard.
    with pytest.raises(ValueError, match="at least one context row"):
        client.predict(
            X_train=np.empty((0, 2)), y_train=[], X_test=[[3.0, 4.0]]
        )


def test_flat_empty_x_train_raises_dimensionality(client):
    # A flat empty list is 1D, so it fails the 2D guard instead.
    with pytest.raises(ValueError, match="X_train must be 2D"):
        client.predict(X_train=[], y_train=[], X_test=[[3.0]])


def test_ragged_x_train_raises(client):
    with pytest.raises(ValueError, match="X_train"):
        client.predict(
            X_train=[[1.0, 2.0], [3.0]], y_train=[1.0, 2.0], X_test=[[3.0, 4.0]]
        )


# --------------------------------------------------------------------------- #
# Remote mode -- error mapping
# --------------------------------------------------------------------------- #


def test_http_400_maps_to_bad_request_error_with_server_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "missing field: y_train"})

    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, handler)

    with pytest.raises(BadRequestError) as exc_info:
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert "missing field: y_train" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_http_401_maps_to_authentication_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    client = SynthefyNoriClient(api_key="bad-key")
    _attach_mock(client, handler)

    with pytest.raises(AuthenticationError) as exc_info:
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert exc_info.value.status_code == 401


def test_retries_on_server_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "temporarily down"})
        return httpx.Response(
            200, json={"task": "regression", "predictions": [7.0]}
        )

    client = SynthefyNoriClient(api_key="test-key", max_retries=2)
    _attach_mock(client, handler)

    preds = client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert preds == [7.0]
    assert calls["n"] == 2


def test_exhausted_retries_raise_internal_server_error(monkeypatch):
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = SynthefyNoriClient(api_key="test-key", max_retries=1)
    _attach_mock(client, handler)

    with pytest.raises(InternalServerError):
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


def test_request_model_roundtrip():
    req = NoriPredictRequest(
        X_train=[[1.0, 2.0]], y_train=[3.0], X_test=[[4.0, 5.0]]
    )
    assert req.model_dump() == {
        "X_train": [[1.0, 2.0]],
        "y_train": [3.0],
        "X_test": [[4.0, 5.0]],
        "task": "regression",
    }


def test_response_model_parses_predictions():
    resp = NoriPredictResponse(
        **{"task": "regression", "predictions": [1.0, 2.0, 3.0]}
    )
    assert resp.predictions == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Local mode
# --------------------------------------------------------------------------- #


def test_local_predict_raises_helpful_error_without_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori" or name.startswith("synthefy_nori."):
            raise ImportError("No module named 'synthefy_nori'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = SynthefyNoriClient(mode="local")
    with pytest.raises(ImportError, match=r"synthefy\[local\]"):
        client.predict(
            X_train=[[1.0, 2.0]], y_train=[3.0], X_test=[[4.0, 5.0]]
        )


def test_local_predict_validates_shapes_before_import():
    # Shape validation happens before the optional dependency is imported, so
    # this raises ValueError regardless of whether synthefy-nori is present.
    client = SynthefyNoriClient(mode="local")
    with pytest.raises(ValueError, match="they must match"):
        client.predict(X_train=[[1.0], [2.0]], y_train=[1.0], X_test=[[3.0]])


@pytest.mark.slow
def test_local_predict_real_inference():
    pytest.importorskip("synthefy_nori")

    client = SynthefyNoriClient(mode="local")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 3))
    y_train = X_train[:, 0] * 2.0 + 1.0
    X_test = rng.normal(size=(5, 3))

    preds = client.predict(X_train, y_train, X_test)

    assert isinstance(preds, list)
    assert len(preds) == 5
    assert all(isinstance(p, float) for p in preds)
