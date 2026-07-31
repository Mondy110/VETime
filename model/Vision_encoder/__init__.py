"""
DEPRECATED: This module re-exports from src.models.vision_encoder for backward compatibility.
New code should import directly from src.models.vision_encoder:
    from src.models.vision_encoder import V_model
"""

from src.models.vision_encoder.v_encoder import V_model

__all__ = ['V_model']
