"""Temporal encoder boundary.

The implementation is temporarily re-exported from the proven legacy module;
the public model only depends on this boundary, so the implementation can be
moved without changing checkpoint names or forward semantics.
"""

from model.TS_encoder.ts_encoder import PreparedTimeSeriesInputs, TimeSeriesEncoder

__all__ = ["PreparedTimeSeriesInputs", "TimeSeriesEncoder"]
