"""Live rung tests: the client, over the network, against a real Nori deployment.

These intentionally hit a real deployment, and they are the **third** runner of one shared
definition. The cases come from ``synthefy_nori.testing.rung_cases``, which also drives:

- the in-process engine test on a local GPU (synthefy-nori-internal)
- the raw-HTTP deployment smoke, ``ci/baseten_smoke_memory.py`` (same repo)

Sharing them across repo boundaries is the point. The serving-memory ladder is decided by
hardware, so "the rung is right locally", "the rung is right on the deployment" and "the rung is
right through the client" are three different claims — and this is the only one that exercises
what a customer actually runs: the client's own request building, its response parsing, and the
network in between. Three hand-maintained copies of the expectations would drift silently,
because each side keeps passing its own suite.

Requires:
    SYNTHEFY_NORI_API_KEY   a key granted the model slug below
    NORI_MODEL              slug to target, default "nori-6m"
    synthefy-nori >= 0.13.0 installed (the cases ship in it) -- skipped otherwise, which is
                            also how this file behaves before that version reaches PyPI

Run with:  uv run pytest tests/online_tests/test_nori_memory_rungs.py -v
"""

import importlib.util
import os

import pytest

from synthefy import SynthefyNoriClient
from synthefy.api_client import BadRequestError

def _rung_cases_installed() -> bool:
    """Is a synthefy-nori that ships the shared rung cases importable?

    Two steps on purpose: ``find_spec`` on a DOTTED name imports the parent package, so probing
    ``synthefy_nori.testing`` directly raises ModuleNotFoundError when synthefy-nori is not
    installed at all -- which is the common case here, since it is an optional extra. Check the
    top-level name first (the same way the client's own ``_local_available`` does), and only then
    the submodule, which is the part that dates the version.
    """
    if importlib.util.find_spec("synthefy_nori") is None:
        return False
    return importlib.util.find_spec("synthefy_nori.testing") is not None


# The cases live in synthefy-nori, an optional dependency here (the `local` extra), and only from
# 0.13.0. Skip rather than fail, so this file is harmless before that release reaches PyPI.
if not _rung_cases_installed():
    pytest.skip(
        "needs synthefy-nori >= 0.13.0, which ships the shared rung cases "
        '(pip install -U "synthefy[local]")',
        allow_module_level=True,
    )

from synthefy_nori.testing import rung_cases as rungs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("SYNTHEFY_NORI_API_KEY"),
    reason="SYNTHEFY_NORI_API_KEY is required to reach a real deployment",
)


@pytest.fixture(scope="module")
def client() -> SynthefyNoriClient:
    """A remote client, optionally pointed at a deployment URL instead of the gateway.

    ``DIRECT_MODEL_URL`` exists because the gateway and the deployment are separately
    breakable: a route can be bound and still not carry traffic. Overriding the base URL keeps
    the client path under test when the difference is the gateway's.
    """
    direct = os.environ.get("DIRECT_MODEL_URL")
    kwargs = {}
    if direct:
        kwargs = {"base_url": direct.rsplit("/predict", 1)[0], "endpoint": "/predict"}
    with SynthefyNoriClient(
        api_key=os.environ["SYNTHEFY_NORI_API_KEY"],
        model=os.environ.get("NORI_MODEL", "nori-6m"),
        timeout=900.0,  # a cold replica can take ~60-100s before it answers at all
        **kwargs,
    ) as remote:
        yield remote


@pytest.fixture(scope="module")
def table():
    body = rungs.build_table()
    return body["X_train"], body["y_train"], body["X_test"]


@pytest.fixture(scope="module")
def baseline(client, table):
    """The exact-rung predictions every other case is compared against."""
    predictions = client.predict(*table, memory=rungs.BASELINE.memory)
    report = client.last_memory_report
    assert report is not None, "the deployment ignored memory= (it predates the field)"
    assert report["query_chunk"] < rungs.N_QUERY, (
        f"query_chunk={report['query_chunk']} does not chunk {rungs.N_QUERY} query rows, so no "
        "cache is reused and every rung assertion below would be vacuous"
    )
    return predictions


class TestRungsThroughTheClient:
    @pytest.mark.parametrize("case", rungs.CASES, ids=lambda c: c.label)
    def test_the_rung_the_case_asks_for_is_what_runs(self, client, table, baseline, case):
        predictions = client.predict(*table, memory=case.memory)
        report = client.last_memory_report

        assert report is not None, "no memory_report came back"
        assert report["rung"] == case.rung, (
            f"{case.label}: expected {case.rung!r}, got {report['rung']!r} — {case.why}"
        )
        assert len(predictions) == rungs.N_QUERY
        assert all(p is not None for p in predictions)

        if case.bit_exact:
            assert predictions == baseline, (
                f"{case.label} must be bit-identical to the baseline ({case.why})"
            )
        elif case is not rungs.BASELINE:
            # A lossy rung returning the baseline's exact values did not actually engage.
            assert predictions != baseline, (
                f"{case.label} reported {report['rung']!r} but returned the baseline's exact "
                "predictions, so the rung did not change what ran"
            )


class TestPolicyPlumbing:
    def test_a_request_without_a_policy_reports_nothing(self, client, table):
        client.predict(*table[:2], table[2][:8])
        assert client.last_memory_report is None

    def test_an_incoherent_policy_is_rejected_before_any_inference(self, client, table):
        # Validation lives in the library and runs server-side before the forward pass, so this
        # should come back fast and carry the library's own message.
        with pytest.raises(BadRequestError) as excinfo:
            client.predict(*table[:2], table[2][:8], memory={"int8": True})
        assert "int8" in str(excinfo.value)

    def test_a_forbidden_subsample_is_a_400_not_a_500(self, client, table):
        """Regression test for a caller-triggerable 500 (synthefy-nori-internal#300)."""
        with pytest.raises(BadRequestError) as excinfo:
            client.predict(
                *table[:2], table[2][:8],
                memory={"allow_subsample": False, "elements_budget": 500},
            )
        assert "allow_subsample" in str(excinfo.value)

    def test_the_host_budget_is_clamped_and_says_so(self, client, table):
        client.predict(*table[:2], table[2][:8], memory={"host_budget_frac": 0.95})
        report = client.last_memory_report or {}
        assert "host_budget_frac" in (report.get("clamped") or []), (
            "an over-large host budget must be capped and reported, never applied silently: "
            "exceeding the container's cgroup limit is a SIGKILL that costs the next caller a "
            "cold start"
        )
