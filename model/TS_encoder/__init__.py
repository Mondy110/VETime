"""
DEPRECATED: This module re-exports from src.models.ts_encoder for backward compatibility.
New code should import directly from src.models.ts_encoder:
    from src.models.ts_encoder import TS_Model, TimeSeriesEncoder, TimeSeriesConfig
"""

from src.models.ts_encoder.ts_model import TS_Model
from src.models.ts_encoder.ts_encoder import TimeSeriesEncoder
from src.models.ts_encoder.config import TimeSeriesConfig, default_config_t

__all__ = ['TS_Model', 'TimeSeriesEncoder', 'TimeSeriesConfig', 'default_config_t']
