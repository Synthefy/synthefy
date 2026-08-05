"""The local NoriTSForecaster is advertised on the package surface.

Kept dependency-free: asserts the export contract only, without resolving
NoriTSForecaster (which needs the optional `synthefy[timeseries]` deps).
"""
import pytest

import synthefy


def test_nori_ts_forecaster_in_all():
    assert "NoriTSForecaster" in synthefy.__all__


def test_unknown_attr_still_raises_attributeerror():
    with pytest.raises(AttributeError):
        synthefy.definitely_not_an_attribute
