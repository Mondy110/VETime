"""
DEPRECATED: This module re-exports from src.models.ts_encoder.ts_model for backward compatibility.
New code should import directly from src.models.ts_encoder:
    from src.models.ts_encoder import TS_Model
"""

from src.models.ts_encoder.ts_model import TS_Model

__all__ = ['TS_Model']
