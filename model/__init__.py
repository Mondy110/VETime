"""
DEPRECATED: This module now re-exports from src.models for backward compatibility.
New code should import directly from src.models:
    from src.models import VETIME
    from src.models.vision_encoder import V_model
    from src.models.ts_encoder import TS_Model

This module provides the VETime model and its components for
multimodal time series anomaly detection.
"""

# Main models
from src.models.vetime import VETIME

# Vision encoder components
from src.models.vision_encoder.v_encoder import V_model

# Time series encoder components
from src.models.ts_encoder.ts_encoder import TimeSeriesEncoder
from src.models.ts_encoder.ts_model import TS_Model
from src.models.ts_encoder.config import TimeSeriesConfig, default_config_t

__all__ = [
    'VETIME',
    'V_model',
    'TimeSeriesEncoder',
    'TS_Model',
    'TimeSeriesConfig',
    'default_config_t',
]
