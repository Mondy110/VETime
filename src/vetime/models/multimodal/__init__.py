"""Composed temporal/vision VETime models.

Submodules are intentionally not imported eagerly: temporal rotary utilities
depend on the CMRG primitives in this package.
"""

__all__ = ["VETimeMultimodalModel", "VETimeOptions"]


def __getattr__(name):
    if name in __all__:
        from .model import VETimeMultimodalModel, VETimeOptions

        return {"VETimeMultimodalModel": VETimeMultimodalModel, "VETimeOptions": VETimeOptions}[name]
    raise AttributeError(name)
