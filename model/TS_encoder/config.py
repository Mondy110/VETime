"""
DEPRECATED: This module re-exports from src.models.ts_encoder.config for backward compatibility.
New code should import directly from src.models.ts_encoder:
    from src.models.ts_encoder import TimeSeriesConfig, default_config_t
"""

from src.models.ts_encoder.config import TimeSeriesConfig, default_config_t

__all__ = ['TimeSeriesConfig', 'default_config_t']
