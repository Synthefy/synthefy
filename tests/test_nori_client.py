"""Unit tests for the Synthefy Nori client.

Remote mode is tested against a mocked httpx transport so these tests never hit
the network. The real local-inference test is marked ``slow`` and skips unless
the optional ``synthefy-nori`` package is installed.
"""

import builtins
import json
import math
import warnings
from typing import Callable, Dict, List

import httpx
import numpy as np
import pandas as pd
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
    NORI_VARIANTS,
    _is_thinking_model,
    _resolve_remote_levels,
    _snap_to_levels,
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
    # Default selector is the base gateway slug (mapped server-side to the 30M deployment).
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


# --------------------------------------------------------------------------- #
# pandas inputs -- DataFrame / Series
# --------------------------------------------------------------------------- #


def test_predict_accepts_dataframes_and_series():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([5.0], capture))

    X_train = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    y_train = pd.Series([1.0, 2.0])
    X_test = pd.DataFrame({"a": [2.0], "b": [2.0]})

    preds = client.predict(X_train, y_train, X_test)

    assert preds == [5.0]
    # DataFrame/Series inputs serialize to plain JSON lists of floats.
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["y_train"] == [1.0, 2.0]
    assert capture["body"]["X_test"] == [[2.0, 2.0]]


def test_y_train_single_column_dataframe_is_accepted():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.DataFrame({"target": [1.0, 2.0]}),
        X_test=pd.DataFrame({"a": [2.0]}),
    )
    assert capture["body"]["y_train"] == [1.0, 2.0]


def test_dataframe_xtest_is_aligned_to_xtrain_by_column_name():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([9.0], capture))

    X_train = pd.DataFrame({"a": [0.0, 1.0], "b": [10.0, 11.0]})
    # X_test columns are in the opposite order; they must be realigned to a, b.
    X_test = pd.DataFrame({"b": [12.0], "a": [2.0]})

    client.predict(X_train, [1.0, 2.0], X_test)

    # Realigned to X_train's column order (a, b), not X_test's literal order.
    assert capture["body"]["X_test"] == [[2.0, 12.0]]


def test_dataframe_column_set_mismatch_raises():
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="same feature columns"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "c": [2.0]}),
        )


# --------------------------------------------------------------------------- #
# One-hot featurization of non-numeric DataFrame columns (fit on X_train)
# --------------------------------------------------------------------------- #


def test_non_numeric_columns_are_ordinal_encoded_by_default():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([5.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["y", "x", "y"]}),
        y_train=[1.0, 2.0, 3.0],
        # 'z' is unseen in training -> code -1 (the server's unknown_value).
        X_test=pd.DataFrame({"a": [3.0, 4.0], "cat": ["x", "z"]}),
    )

    assert out == [5.0]
    # one column per categorical, codes in sorted-category order: x=0, y=1
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]
    assert capture["body"]["X_test"] == [[3.0, 0.0], [4.0, -1.0]]


def test_ordinal_missing_categorical_is_forwarded_as_nan():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
    )
    sent = capture["body"]["X_train"]
    # x=0, y=1; the missing row stays NaN for server-side imputation.
    assert sent[0] == [0.0, 0.0] and sent[2] == [2.0, 1.0]
    assert math.isnan(sent[1][1])
    assert capture["body"]["X_test"] == [[5.0, 0.0]]


def test_ordinal_literal_nan_string_is_a_real_category():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["nan", "x"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "cat": ["nan"]}),
    )
    # "nan" (the string) sorts before "x": nan=0, x=1 — not treated as missing.
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 0.0]]


def test_ordinal_name_value_collision_is_a_non_issue():
    # Under one-hot, column 'a' value 'b_x' and column 'a_b' value 'x' collide
    # in the '<column>_<value>' namespace; ordinal never generates columns.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": ["b_x", "c"], "a_b": ["x", "y"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": ["b_x"], "a_b": ["x"]}),
    )
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0]]
    assert capture["body"]["X_test"] == [[0.0, 0.0]]


def test_invalid_categorical_encoding_raises():
    client = SynthefyNoriClient(api_key="test-key")
    df = pd.DataFrame({"cat": ["x", "y"]})
    with pytest.raises(ValueError, match="categorical_encoding"):
        client.predict(
            X_train=df, y_train=[1.0, 2.0], X_test=df,
            categorical_encoding="hashing",
        )


def test_non_numeric_columns_are_one_hot_encoded():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([5.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
        y_train=[1.0, 2.0],
        # 'z' is unseen in training -> its indicator group is all zeros.
        X_test=pd.DataFrame({"a": [2.0], "cat": ["z"]}),
        categorical_encoding="onehot",
    )

    assert out == [5.0]
    # columns: a, cat_x, cat_y  (numerics first, then sorted one-hot groups)
    assert capture["body"]["X_train"] == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 0.0, 0.0]]


def test_one_hot_train_category_absent_in_test_is_kept_as_zero_column():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([9.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", "y", "z"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )

    # train has 3 categories -> cat_x, cat_y, cat_z; test row 'x' -> [1,0,0]
    assert capture["body"]["X_test"] == [[5.0, 1.0, 0.0, 0.0]]


def test_high_cardinality_column_is_dropped_with_warning():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    with pytest.warns(UserWarning, match="unique values"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "hc": ["p", "q", "r"]}),
            y_train=[1.0, 2.0, 3.0],
            X_test=pd.DataFrame({"a": [3.0], "hc": ["p"]}),
            max_categorical_cardinality=2,  # 'hc' has 3 uniques -> dropped
        )

    assert capture["body"]["X_train"] == [[0.0], [1.0], [2.0]]
    assert capture["body"]["X_test"] == [[3.0]]


def test_datetime_column_is_dropped_with_warning():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    with pytest.warns(UserWarning, match="datetime"):
        client.predict(
            X_train=pd.DataFrame(
                {"a": [0.0, 1.0], "d": pd.to_datetime(["2024-01-01", "2024-01-02"])}
            ),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "d": pd.to_datetime(["2024-01-03"])}),
        )

    assert capture["body"]["X_train"] == [[0.0], [1.0]]


def test_bool_columns_pass_through_as_numeric():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    # bool is numeric (is_numeric_dtype) -> not one-hot; True/False -> 1.0/0.0
    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "flag": [True, False]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "flag": [True]}),
    )
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["X_test"] == [[2.0, 1.0]]


def test_all_numeric_dataframe_is_not_featurized():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any featurization warning would fail
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "b": [2.0]}),
        )
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]


def test_numpy_string_array_raises_pointing_to_dataframe():
    # A 2D numpy/list array is single-dtype, so a string column makes the WHOLE
    # array strings — there are no per-column types to one-hot. We raise and
    # point the caller to DataFrames (where each column keeps its own dtype).
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="one-hot"):
        client.predict(
            X_train=np.array([[1.0, "x"], [2.0, "y"]]),
            y_train=[1.0, 2.0],
            X_test=np.array([[3.0, "z"]]),
        )


def test_column_numeric_in_train_but_not_test_raises_clearly():
    # A column numeric in X_train but object-dtype in X_test is caught with a
    # clear type-mismatch error (not a later cryptic float-cast failure).
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="matching column types"):
        client.predict(
            X_train=pd.DataFrame({"b": [1.0, 2.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"b": ["x"]}),  # object dtype, not numeric
        )


def test_numeric_category_dtype_is_treated_as_numeric_not_one_hot():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn / drop / explode
        client.predict(
            X_train=pd.DataFrame(
                {"a": [0.0, 1.0, 2.0],
                 "r": pd.Categorical([1, 2, 3], categories=[1, 2, 3])}
            ),
            y_train=[1.0, 2.0, 3.0],
            X_test=pd.DataFrame(
                {"a": [5.0], "r": pd.Categorical([2], categories=[1, 2, 3])}
            ),
        )
    # 'r' kept as a single numeric column (its values), not exploded to r_1/r_2/r_3
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert capture["body"]["X_test"] == [[5.0, 2.0]]


def test_all_missing_categorical_column_dropped_with_warning():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    with pytest.warns(UserWarning, match="no non-missing"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": [None, None]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "cat": [None]}),
        )
    assert capture["body"]["X_train"] == [[0.0], [1.0]]


def test_timedelta_column_raises_unsupported():
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="timedelta"):
        client.predict(
            X_train=pd.DataFrame(
                {"a": [0.0, 1.0], "d": pd.to_timedelta(["1 days", "2 days"])}
            ),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "d": pd.to_timedelta(["3 days"])}),
        )


def test_nan_in_categorical_gets_its_own_indicator_column():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_x, cat_y, cat_nan (the missing row -> its own indicator)
    assert capture["body"]["X_train"] == [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 0.0],
    ]
    assert capture["body"]["X_test"] == [[5.0, 1.0, 0.0, 0.0]]


def test_integer_category_with_nan_does_not_crash():
    # Regression: demoting an int-category column to numeric must not choke on
    # NaN (it promotes to float and the NaN is forwarded for server imputation).
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame(
            {"r": pd.Categorical([1, 2, None], categories=[1, 2, 3])}
        ),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"r": pd.Categorical([2], categories=[1, 2, 3])}),
    )
    sent = capture["body"]["X_train"]
    assert sent[0] == [1.0] and sent[1] == [2.0]
    assert math.isnan(sent[2][0])  # NaN forwarded, not crashed


def test_duplicate_column_names_raise():
    client = SynthefyNoriClient(api_key="test-key")
    df = pd.DataFrame([[0.0, 1.0, 2.0]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column name"):
        client.predict(X_train=df, y_train=[1.0], X_test=df)


def test_one_hot_name_value_collision_raises_clearly():
    # column 'a' value 'b_x' and column 'a_b' value 'x' both -> dummy 'a_b_x'
    client = SynthefyNoriClient(api_key="test-key")
    Xtr = pd.DataFrame({"a": ["b_x", "b_x"], "a_b": ["x", "y"]})
    Xte = pd.DataFrame({"a": ["b_x"], "a_b": ["x"]})
    with pytest.raises(ValueError, match="duplicate column names"):
        client.predict(X_train=Xtr, y_train=[1.0, 2.0], X_test=Xte,
                       categorical_encoding="onehot")


def test_period_column_raises_unsupported():
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="unsupported dtype"):
        client.predict(
            X_train=pd.DataFrame(
                {"a": [0.0, 1.0], "p": pd.period_range("2024-01", periods=2, freq="M")}
            ),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame(
                {"a": [2.0], "p": pd.period_range("2024-03", periods=1, freq="M")}
            ),
        )


def test_zero_feature_columns_raises():
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="at least one feature column"):
        client.predict(
            X_train=pd.DataFrame(index=[0, 1]),  # 2 rows, 0 columns
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame(index=[0]),
        )


def test_categorical_train_with_array_xtest_points_to_dataframe():
    # X_train is a DataFrame with a categorical column but X_test is an array:
    # we can't align/one-hot, so raise a message pointing at passing DataFrames.
    client = SynthefyNoriClient(api_key="test-key")
    with pytest.raises(ValueError, match="not a DataFrame"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
            y_train=[1.0, 2.0],
            X_test=np.array([[2.0, 0.0]]),
        )


def test_literal_nan_string_category_is_not_treated_as_missing():
    # A real category whose value is the string "nan" (no actual NaN) must encode
    # cleanly and NOT trip the duplicate-column guard against the dummy_na column.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["nan", "x"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "cat": ["nan"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_nan, cat_x  — no NaN-indicator column, no error
    assert capture["body"]["X_train"] == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 1.0, 0.0]]


def test_nonpositive_cardinality_cap_raises():
    client = SynthefyNoriClient(api_key="test-key")
    for cap in (0, -5):
        with pytest.raises(ValueError, match="positive integer"):
            client.predict(
                X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
                y_train=[1.0, 2.0],
                X_test=pd.DataFrame({"a": [2.0], "cat": ["x"]}),
                max_categorical_cardinality=cap,
            )


# --------------------------------------------------------------------------- #
# as_pandas=True -- return a Series (named after y_train, indexed by X_test)
# --------------------------------------------------------------------------- #


def test_default_return_is_a_plain_list_not_series():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([1.0, 2.0], name="demand"),
        X_test=pd.DataFrame({"a": [2.0, 3.0]}),
    )
    assert isinstance(out, list)
    assert out == [1.0, 2.0]


def test_as_pandas_returns_series_named_after_y_train_and_indexed_by_xtest():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([10.0, 20.0], capture))

    X_test = pd.DataFrame({"a": [2.0, 3.0]}, index=["w1", "w2"])
    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([1.0, 2.0], name="demand"),
        X_test=X_test,
        as_pandas=True,
    )

    assert isinstance(out, pd.Series)
    assert out.name == "demand"
    assert list(out.index) == ["w1", "w2"]
    assert out.tolist() == [10.0, 20.0]


def test_as_pandas_uses_single_column_dataframe_y_label():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([7.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.DataFrame({"units": [1.0, 2.0]}),
        X_test=pd.DataFrame({"a": [2.0]}),
        as_pandas=True,
    )
    assert isinstance(out, pd.Series)
    assert out.name == "units"


def test_as_pandas_with_non_pandas_inputs_uses_defaults():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([5.0, 6.0], capture))

    out = client.predict(
        X_train=[[0.0], [1.0]],
        y_train=[1.0, 2.0],
        X_test=[[2.0], [3.0]],
        as_pandas=True,
    )
    assert isinstance(out, pd.Series)
    assert out.name == "prediction"  # no name available from a plain list
    assert list(out.index) == [0, 1]  # default RangeIndex
    assert out.tolist() == [5.0, 6.0]


# --------------------------------------------------------------------------- #
# NaN / missing values are forwarded for server-side imputation (not rejected)
# --------------------------------------------------------------------------- #


def test_nan_is_forwarded_to_the_server():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], capture))

    # A missing value in any input must NOT raise; the model imputes it
    # server-side. The NaN rides through to the request body unchanged.
    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, np.nan]}),
        y_train=[1.0, 2.0],
        X_test=np.array([[2.0, 2.0]]),
    )

    # json.loads parses the non-strict ``NaN`` token back to float('nan').
    sent = capture["body"]["X_train"]
    assert math.isnan(sent[1][1])


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


def test_final_error_wins_over_stale_earlier_attempt(monkeypatch):
    # A transient connection error on attempt 0 followed by a retryable 5xx on the
    # final attempt must surface the FINAL error (InternalServerError), not the
    # stale APIConnectionError from the earlier attempt.
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient")
        return httpx.Response(503, json={"error": "down"})

    client = SynthefyNoriClient(api_key="test-key", max_retries=1)
    _attach_mock(client, handler)

    with pytest.raises(InternalServerError):
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])
    assert calls["n"] == 2


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


# --------------------------------------------------------------------------- #
# Model-variant selector
# --------------------------------------------------------------------------- #

_XTR = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
_YTR = [1.0, 1.0, 2.0]
_XTE = [[2.0, 2.0]]


def test_model_variant_resolves_gateway_and_local():
    c30 = SynthefyNoriClient(api_key="k", model="nori-30m")
    assert c30.model == "synthefy/nori-30m"
    assert c30._local_variant == "nori-30m"

    # "nori" is the default: its synthefy/nori gateway slug is mapped to the 30M deployment
    cdef = SynthefyNoriClient(api_key="k", model="nori")
    assert cdef.model == "synthefy/nori" and cdef._local_variant == "nori-30m"

    # "nori-6m" is the pinned ~6M base: its own gateway slug + explicit local variant, so local
    # mode loads the base even though the package default is now 30M
    c6 = SynthefyNoriClient(api_key="k", model="nori-6m")
    assert c6.model == "synthefy/nori-6m" and c6._local_variant == "nori-6m"

    # a raw gateway slug passes through unchanged
    craw = SynthefyNoriClient(api_key="k", model="synthefy/custom")
    assert craw.model == "synthefy/custom" and craw._local_variant is None

    # None (dedicated endpoint) stays None
    cnone = SynthefyNoriClient(api_key="k", model=None)
    assert cnone.model is None and cnone._local_variant is None


def test_default_model_maps_to_30m():
    # Default selector is the base gateway slug, which is mapped to the 30M deployment
    # (remote) and resolves to the 30M checkpoint (local).
    c = SynthefyNoriClient(api_key="k")
    assert c.model == GATEWAY_MODEL           # "synthefy/nori"
    assert c._local_variant == "nori-30m"


def test_remote_body_uses_variant_gateway_slug():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert capture["body"]["model"] == "synthefy/nori-30m"


def test_local_mode_passes_variant_to_predict(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model="__unset__"):
        seen["model"] = model
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert seen["model"] == "nori-30m"


def test_local_mode_nori_6m_forces_base_variant(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model="__unset__"):
        seen["model"] = model
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    client = SynthefyNoriClient(mode="local", model="nori-6m")  # base 6M
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    # nori-6m forwards its variant explicitly so local loads the base, not the package's 30M default
    assert seen["model"] == "nori-6m"


def test_local_variant_needs_model_selector_on_old_synthefy_nori(monkeypatch):
    def old_predict(X_train, y_train, X_test, *, task=None):  # no model= param
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: old_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ImportError, match="model= selector"):
        client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)


def test_gateway_slug_resolves_to_local_variant():
    # Gateway slugs map to the right local checkpoint, not a raw-repo lookup, so slug users get
    # the intended weights locally. synthefy/nori is mapped to 30M; synthefy/nori-6m is the base.
    cbase = SynthefyNoriClient(api_key="k", model="synthefy/nori-6m")
    assert cbase.model == "synthefy/nori-6m" and cbase._local_variant == "nori-6m"
    cdef = SynthefyNoriClient(api_key="k", model="synthefy/nori")
    assert cdef.model == "synthefy/nori" and cdef._local_variant == "nori-30m"
    c30 = SynthefyNoriClient(api_key="k", model="synthefy/nori-30m")
    assert c30.model == "synthefy/nori-30m" and c30._local_variant == "nori-30m"


# --------------------------------------------------------------------------- #
# Nori Thinking is hosted-API only; no silent fallback to the base model
# --------------------------------------------------------------------------- #


def test_is_thinking_model_matches_every_tier():
    for name in (
        "synthefy/nori-30m-thinking",
        "synthefy/nori-30m-thinking-medium",
        "synthefy/nori-30m-thinking-high",
        "nori-30m-thinking-medium",
        "NORI-THINKING",
    ):
        assert _is_thinking_model(name)
    for name in ("nori", "nori-6m", "nori-30m", "synthefy/nori", "synthefy/custom", None):
        assert not _is_thinking_model(name)


@pytest.mark.parametrize("mode", ["local", "auto"])
@pytest.mark.parametrize(
    "model",
    [
        "synthefy/nori-30m-thinking-medium",  # raw gateway slug
        "nori-30m-thinking",                  # friendly aliases (every tier)
        "nori-30m-thinking-medium",
        "nori-30m-thinking-high",
    ],
)
def test_thinking_model_raises_in_local_and_auto_modes(mode, model):
    # Thinking has no local checkpoint: constructing for local/auto inference must raise a clear
    # error (pointing at mode="remote"), never silently run the base ~6M model.
    with pytest.raises(ValueError, match=r"Thinking.*hosted Synthefy API"):
        SynthefyNoriClient(mode=mode, model=model)


@pytest.mark.parametrize(
    "friendly,slug",
    [
        # Only the medium budget is deployed today; the bare name routes to it too.
        ("nori-30m-thinking", "synthefy/nori-30m-thinking-medium"),
        ("nori-30m-thinking-medium", "synthefy/nori-30m-thinking-medium"),
    ],
)
def test_thinking_friendly_name_resolves_to_gateway_slug_remote(friendly, slug):
    # In remote mode the friendly Thinking name maps to its gateway slug (uniform with nori-30m),
    # so callers never need the raw "synthefy/" prefix.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model=friendly)  # remote (default)
    assert client.model == slug
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert capture["body"]["model"] == slug


def test_unknown_model_raises_in_local_mode_no_base_fallback():
    # A selector with no local checkpoint (a custom deployment slug) must raise in local mode
    # rather than silently substituting the base model.
    with pytest.raises(ValueError, match=r"no local checkpoint"):
        SynthefyNoriClient(mode="local", model="synthefy/custom")


def test_unknown_model_still_passes_through_in_remote_mode():
    # The local guard is scoped to local mode; a custom slug is still a valid remote gateway id.
    client = SynthefyNoriClient(api_key="k", model="synthefy/custom")
    assert client.model == "synthefy/custom"


# --------------------------------------------------------------------------- #
# Categorical-target discretization (discretize= / categorical_levels=)
# --------------------------------------------------------------------------- #


def test_remote_snap_mean_snaps_to_y_train_levels():
    # y_train levels {1, 2}; returned means snap to the nearest level.
    client = SynthefyNoriClient(api_key="k")
    _attach_mock(client, _ok_handler([0.9, 1.6, 2.4], {}))
    preds = client.predict(
        X_train=[[0.0], [1.0], [2.0]],
        y_train=[1.0, 1.0, 2.0],
        X_test=[[0.5], [1.5], [2.5]],
        discretize="snap-mean",
    )
    assert preds == [1.0, 2.0, 2.0]


def test_remote_snap_mean_uses_explicit_categorical_levels():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k")
    _attach_mock(client, _ok_handler([0.9, 4.2], capture))
    preds = client.predict(
        X_train=[[0.0], [1.0], [2.0]],
        y_train=[2.0, 2.0, 3.0],
        X_test=[[0.5], [1.5]],
        discretize="snap-mean",
        categorical_levels=[1, 2, 3, 4, 5],
    )
    assert preds == [1.0, 4.0]
    # Discretization is client-side: the wire payload is unchanged.
    assert set(capture["body"]) == {"X_train", "y_train", "X_test", "task", "model"}


def test_remote_snap_mean_as_pandas_stays_on_lattice():
    client = SynthefyNoriClient(api_key="k")
    _attach_mock(client, _ok_handler([0.9, 1.6], {}))
    X_test = pd.DataFrame({"a": [0.5, 1.5]}, index=[10, 20])
    preds = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0]}),
        y_train=pd.Series([1.0, 1.0, 2.0], name="rating"),
        X_test=X_test,
        as_pandas=True,
        discretize="snap-mean",
    )
    assert isinstance(preds, pd.Series)
    assert preds.name == "rating"
    assert list(preds.index) == [10, 20]
    assert preds.tolist() == [1.0, 2.0]


def test_remote_bank_strategy_raises_with_guidance():
    # Validation runs before the request: no paid round-trip on a bad strategy.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k")
    _attach_mock(client, _ok_handler([1.0], capture))
    with pytest.raises(ValueError, match="snap-mean"):
        client.predict(
            X_train=_XTR, y_train=_YTR, X_test=_XTE, discretize="map-cell"
        )
    assert "body" not in capture


def test_remote_levels_without_strategy_raises_with_guidance():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k")
    _attach_mock(client, _ok_handler([1.0], capture))
    with pytest.raises(ValueError, match='discretize="snap-mean"'):
        client.predict(
            X_train=_XTR, y_train=_YTR, X_test=_XTE, categorical_levels=[1, 2]
        )
    assert "body" not in capture


def test_remote_empty_or_nonfinite_levels_raise():
    for bad in ([], [1.0, float("nan")]):
        with pytest.raises(ValueError, match="finite"):
            _resolve_remote_levels([1.0, 2.0], "snap-mean", bad)


def test_remote_all_nan_y_train_raises_instead_of_opaque_error():
    with pytest.raises(ValueError, match="categorical_levels explicitly"):
        _resolve_remote_levels([float("nan"), float("nan")], "snap-mean", None)


def test_remote_levels_order_and_duplicates_are_irrelevant():
    levels = _resolve_remote_levels([1.0], "snap-mean", [3, 1, 2, 1, 3])
    assert levels.tolist() == [1.0, 2.0, 3.0]


def test_snap_to_levels_preserves_nan_predictions():
    levels = _resolve_remote_levels([1.0, 2.0, 3.0], "snap-mean", None)
    snapped = _snap_to_levels([0.9, float("nan"), 2.6], levels)
    assert snapped[0] == 1.0 and snapped[2] == 3.0
    assert math.isnan(snapped[1])


def test_local_mode_passes_discretize_and_levels(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, **kwargs):
        seen.update(kwargs)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: True)
    # model=None forwards no variant selector, isolating this to discretize forwarding
    # (variant selection is covered by its own test).
    client = SynthefyNoriClient(mode="local", model=None)
    client.predict(
        X_train=_XTR,
        y_train=_YTR,
        X_test=_XTE,
        discretize="median-cell",
        categorical_levels=[1, 2],
    )
    assert seen["discretize"] == "median-cell"
    assert seen["categorical_levels"] == [1, 2]


def test_local_mode_levels_alone_activate_package_default(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, **kwargs):
        seen.update(kwargs)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: True)
    client = SynthefyNoriClient(mode="local", model=None)  # no variant selector (see note above)
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE, categorical_levels=[1, 2])
    assert "discretize" not in seen  # package picks its own default (map-cell)
    assert seen["categorical_levels"] == [1, 2]


def test_local_mode_without_discretize_sends_no_extra_kwargs(monkeypatch):
    def strict_predict(X_train, y_train, X_test, *, task=None):  # no **kwargs
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: strict_predict)
    client = SynthefyNoriClient(mode="local", model=None)  # no variant selector; bare predict signature works
    assert client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE) == [1.0]


def test_local_discretize_needs_newer_synthefy_nori(monkeypatch):
    monkeypatch.setattr(
        "synthefy.nori_client._load_local_predict", lambda: (lambda *a, **k: [1.0])
    )
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: False)
    client = SynthefyNoriClient(mode="local")
    with pytest.raises(ImportError, match=r"synthefy\[local\]"):
        client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE, discretize="map-cell")


@pytest.mark.slow
def test_local_discretize_real_inference():
    pytest.importorskip("synthefy_nori.discretize")

    client = SynthefyNoriClient(mode="local")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 3))
    y_train = np.clip(np.round(X_train[:, 0] * 2.0 + 3.0), 1, 5)
    X_test = rng.normal(size=(5, 3))

    preds = client.predict(X_train, y_train, X_test, discretize="map-cell")
    levels = set(np.unique(y_train).tolist())
    assert all(p in levels for p in preds)


# --------------------------------------------------------------------------- #
# Text features -- client-side embedding (text_columns=...)
# --------------------------------------------------------------------------- #

def _fake_embed(texts):
    """Deterministic 8-d embedding, so tests need no sentence-transformers/model."""
    import hashlib
    out = []
    for t in texts:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        out.append(np.frombuffer(h[:8], dtype=np.uint8).astype(np.float32) / 255.0)
    return np.stack(out)


def test_text_columns_embeds_client_side_and_sends_numeric():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    df_train = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "brand": ["a", "b", "a", "b", "a", "b"],           # categorical -> 1 col
        "review": ["good", "bad", "ok", "great", "poor", "fine"],  # text -> SVD
    })
    df_test = pd.DataFrame({"x1": [1.5, 5.5], "brand": ["a", "b"],
                            "review": ["nice", "awful"]})

    preds = client.predict(df_train, [1., 2., 3., 4., 5., 6.], df_test,
                           text_columns=["review"], svd_dim=4, embedder=_fake_embed)

    assert preds == [1.0, 2.0]
    # x1 (numeric) + brand (1 categorical col) + 4 SVD text cols = 6 numeric features
    assert len(capture["body"]["X_train"][0]) == 6
    assert len(capture["body"]["X_test"][0]) == 6
    # the payload is fully numeric (text was embedded away client-side)
    assert all(isinstance(v, (int, float)) for row in capture["body"]["X_test"] for v in row)


def test_text_columns_requires_dataframe():
    client = SynthefyNoriClient(api_key="test-key")
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError):
        client.predict([[1.0, 2.0]], [1.0], [[1.0, 2.0]], text_columns=["review"])
