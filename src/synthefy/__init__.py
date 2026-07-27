from synthefy.api_client import SynthefyAPIClient, SynthefyAsyncAPIClient
from synthefy.nori_data_models import (
    MEMORY_PRESETS,
    MEMORY_RUNGS,
    MemoryPolicy,
    MemoryReport,
)
from synthefy.data_models import (
    NoriPredictRequest,
    NoriPredictResponse,
)
from synthefy.nori_client import SynthefyNoriClient

__version__ = "6.3.0"

__all__ = [
    "MEMORY_PRESETS",
    "MEMORY_RUNGS",
    "MemoryPolicy",
    "MemoryReport",
    "SynthefyAPIClient",
    "SynthefyAsyncAPIClient",
    "SynthefyNoriClient",
    "NoriPredictRequest",
    "NoriPredictResponse",
    "NoriTSForecaster",
    "__version__",
]


def __getattr__(name):
    # Lazily surface the local time-series forecaster. Kept out of the eager
    # imports above so a plain `import synthefy` never pulls nori_ts's heavier
    # deps (gluonts, datasets) — they arrive only with `synthefy[timeseries]`.
    if name == "NoriTSForecaster":
        try:
            from synthefy_nori.nori_ts import NoriTSForecaster
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "NoriTSForecaster requires the time-series extra: "
                'pip install "synthefy[timeseries]"'
            ) from exc
        return NoriTSForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
