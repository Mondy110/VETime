from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vetime.models.temporal.model import TemporalModel
from vetime.models.temporal.config import TemporalModelConfig
from vetime.infrastructure.checkpointing.temporal_legacy import map_legacy_temporal_state_dict

class TS_Model(TemporalModel):
    """Compatibility alias retaining the legacy constructor name."""

    def __init__(self, config_t, **kwargs):
        super().__init__(TemporalModelConfig.from_legacy(config_t))
        self.config = config_t
        self.token_hidden_size = 512
        self.MAX_L = 5000

    @property
    def ts_encoder(self):
        """Legacy attribute view without registering a duplicate module."""
        registered = self.__dict__.get("_modules", {}).get("ts_encoder")
        if registered is not None:
            return registered
        return self.__dict__.get("_legacy_ts_encoder", self.encoder)

    @ts_encoder.setter
    def ts_encoder(self, value):
        """Allow the transitional VETIME wrapper to replace the legacy view."""
        object.__setattr__(self, "_legacy_ts_encoder", value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept old ``ts_encoder.*`` keys while the legacy entry point migrates."""
        if any(key.removeprefix("module.").startswith("ts_encoder.") for key in state_dict):
            state_dict, _ = map_legacy_temporal_state_dict(state_dict, target_prefix="")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
