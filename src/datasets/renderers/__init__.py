# src/datasets/renderers/__init__.py
"""渲染器模块：注册表 + 工厂函数。"""

from typing import Dict, Type, Optional
from .base import BaseRenderer


class RendererRegistry:
    """渲染器注册表，支持按名称获取渲染器类。"""

    _renderers: Dict[str, Type[BaseRenderer]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册渲染器类。

        Args:
            name: 渲染器名称（如 'vico', 'gaf'）

        Returns:
            装饰器函数

        Raises:
            ValueError: 如果名称已被注册
        """
        def decorator(renderer_class: Type[BaseRenderer]):
            if name in cls._renderers:
                raise ValueError(f"Renderer '{name}' already registered")
            cls._renderers[name] = renderer_class
            return renderer_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRenderer]:
        """获取渲染器类。

        Args:
            name: 渲染器名称

        Returns:
            渲染器类

        Raises:
            ValueError: 如果名称未注册
        """
        if name not in cls._renderers:
            available = list(cls._renderers.keys())
            raise ValueError(
                f"Unknown renderer '{name}'. Available: {available}"
            )
        return cls._renderers[name]

    @classmethod
    def list_available(cls) -> list:
        """列出所有已注册渲染器名称。"""
        return list(cls._renderers.keys())


def create_renderer(name: str, **kwargs) -> BaseRenderer:
    """工厂函数：创建渲染器实例。

    Args:
        name: 渲染器名称
        **kwargs: 传递给渲染器构造函数的参数

    Returns:
        渲染器实例
    """
    renderer_class = RendererRegistry.get(name)
    return renderer_class(**kwargs)


__all__ = [
    'BaseRenderer',
    'RendererRegistry',
    'create_renderer',
]
