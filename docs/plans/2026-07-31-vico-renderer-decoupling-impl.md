# ViCO 渲染器解耦实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 ViCO 分支图像生成解耦为可配置、可扩展的渲染器模块，支持通过配置文件切换不同渲染算法。

**Architecture:** 采用策略模式 + 注册表机制。BaseRenderer 定义统一接口，RendererRegistry 管理渲染器注册，Trainer 通过工厂函数创建渲染器实例。

**Tech Stack:** Python ABC, PyTorch, OmegaConf (Hydra 配置)

---

## Task 1: 创建渲染器模块目录结构

**Files:**
- Create: `src/datasets/renderers/__init__.py`
- Create: `src/datasets/renderers/base.py`

**Step 1: 创建目录**

```bash
mkdir -p src/datasets/renderers
```

**Step 2: 创建空文件初始化**

```bash
touch src/datasets/renderers/__init__.py src/datasets/renderers/base.py
```

**Step 3: 验证目录结构**

```bash
ls -la src/datasets/renderers/
```
Expected: 显示 `__init__.py`, `base.py`

**Step 4: 提交**

```bash
git add src/datasets/renderers/
git commit -m "feat: create renderers module directory structure"
```

---

## Task 2: 实现 BaseRenderer 抽象基类

**Files:**
- Modify: `src/datasets/renderers/base.py`

**Step 1: 编写 BaseRenderer 代码**

```python
# src/datasets/renderers/base.py
"""渲染器抽象基类。"""

from abc import ABC, abstractmethod
import torch
from typing import Optional


class BaseRenderer(ABC):
    """频域图像渲染器基类。

    所有渲染器必须实现 render_batch 方法，将时序数据转换为图像。
    """

    @abstractmethod
    def render_batch(
        self,
        time_series: torch.Tensor,    # [B, T, F] 原始时序
        att_mask: Optional[torch.Tensor] = None,  # [B, T]
        img_size: int = 224,
    ) -> torch.Tensor:
        """渲染频域图像批次。

        Args:
            time_series: [B, T, F] 原始时序数据（未归一化）
            att_mask: [B, T] 注意力掩码，True=有效，False=padding
            img_size: 输出图像尺寸

        Returns:
            [B, 3, img_size, img_size] float32 tensor，值在 [0, 255] 范围
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"
```

**Step 2: 验证语法**

```bash
python -c "from src.datasets.renderers.base import BaseRenderer; print(BaseRenderer)"
```
Expected: `<class 'src.datasets.renderers.base.BaseRenderer'>`

**Step 3: 提交**

```bash
git add src/datasets/renderers/base.py
git commit -m "feat: add BaseRenderer abstract base class"
```

---

## Task 3: 实现 RendererRegistry 注册表

**Files:**
- Modify: `src/datasets/renderers/__init__.py`

**Step 1: 编写注册表代码**

```python
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
```

**Step 2: 验证注册表工作正常**

```bash
python -c "
from src.datasets.renderers import RendererRegistry, BaseRenderer

@RendererRegistry.register('test')
class TestRenderer(BaseRenderer):
    def render_batch(self, ts, att_mask=None, img_size=224):
        return ts

print('Registered:', RendererRegistry.list_available())
"
```
Expected: `Registered: ['test']`

**Step 3: 提交**

```bash
git add src/datasets/renderers/__init__.py
git commit -m "feat: add RendererRegistry and create_renderer factory"
```

---

## Task 4: 实现 ViCORenderer（包装现有实现）

**Files:**
- Create: `src/datasets/renderers/vico.py`
- Modify: `src/datasets/renderers/__init__.py` (添加导入)

**Step 1: 创建 ViCORenderer 文件**

```python
# src/datasets/renderers/vico.py
"""ViCO 频域渲染器：STFT + 热力图 + 梯度图三视图。"""

import torch
from typing import Optional
from .base import BaseRenderer
from . import RendererRegistry


@RendererRegistry.register('vico')
class ViCORenderer(BaseRenderer):
    """ViCO 频域渲染器。

    使用 STFT 频谱图 + 周期折叠热力图 + 梯度图组成三通道 RGB 图像。
    周期通过 FFT 自动检测（find_period_fft）。
    """

    def render_batch(
        self,
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        img_size: int = 224,
    ) -> torch.Tensor:
        """渲染 ViCO 频域图像批次。

        内部调用 src.datasets.pre_image.render_vico_batch。
        """
        from src.datasets.pre_image import render_vico_batch
        return render_vico_batch(time_series, att_mask, img_size)
```

**Step 2: 更新 __init__.py 导入 ViCORenderer**

在 `src/datasets/renderers/__init__.py` 末尾添加：

```python
# 导入已注册的渲染器（触发注册）
from .vico import ViCORenderer

__all__ = [
    'BaseRenderer',
    'RendererRegistry',
    'create_renderer',
    'ViCORenderer',
]
```

**Step 3: 验证 ViCORenderer 注册成功**

```bash
python -c "
from src.datasets.renderers import RendererRegistry, create_renderer, ViCORenderer

print('Available renderers:', RendererRegistry.list_available())
renderer = create_renderer('vico')
print('Created:', renderer)
"
```
Expected:
```
Available renderers: ['vico']
Created: ViCORenderer()
```

**Step 4: 提交**

```bash
git add src/datasets/renderers/
git commit -m "feat: add ViCORenderer wrapping existing implementation"
```

---

## Task 5: 编写渲染器单元测试

**Files:**
- Create: `tests/datasets/renderers/__init__.py`
- Create: `tests/datasets/renderers/test_vico_renderer.py`

**Step 1: 创建测试目录**

```bash
mkdir -p tests/datasets/renderers
touch tests/datasets/renderers/__init__.py
```

**Step 2: 编写测试文件**

```python
# tests/datasets/renderers/test_vico_renderer.py
"""ViCO 渲染器单元测试。"""

import pytest
import torch
import numpy as np


class TestRendererRegistry:
    """注册表测试。"""

    def test_register_and_get(self):
        """测试注册和获取。"""
        from src.datasets.renderers import RendererRegistry, BaseRenderer

        @RendererRegistry.register('test_renderer')
        class TestRenderer(BaseRenderer):
            def render_batch(self, ts, att_mask=None, img_size=224):
                return ts

        assert 'test_renderer' in RendererRegistry.list_available()
        cls = RendererRegistry.get('test_renderer')
        assert cls is TestRenderer

    def test_duplicate_register_raises(self):
        """测试重复注册抛出异常。"""
        from src.datasets.renderers import RendererRegistry, BaseRenderer

        with pytest.raises(ValueError, match="already registered"):
            @RendererRegistry.register('vico')  # 已存在
            class Another(BaseRenderer):
                def render_batch(self, ts, att_mask=None, img_size=224):
                    return ts

    def test_unknown_renderer_raises(self):
        """测试获取未注册渲染器抛出异常。"""
        from src.datasets.renderers import RendererRegistry

        with pytest.raises(ValueError, match="Unknown renderer"):
            RendererRegistry.get('nonexistent')


class TestViCORenderer:
    """ViCO 渲染器测试。"""

    def test_create_renderer_factory(self):
        """测试工厂函数创建。"""
        from src.datasets.renderers import create_renderer, ViCORenderer

        renderer = create_renderer('vico')
        assert isinstance(renderer, ViCORenderer)

    def test_render_batch_output_shape(self):
        """测试输出形状正确。"""
        from src.datasets.renderers import create_renderer

        renderer = create_renderer('vico')

        # 创建测试数据 [B=2, T=100, F=1]
        ts = torch.randn(2, 100, 1)

        output = renderer.render_batch(ts, img_size=224)

        assert output.shape == (2, 3, 224, 224)
        assert output.dtype == torch.float32

    def test_render_batch_with_att_mask(self):
        """测试带注意力掩码的渲染。"""
        from src.datasets.renderers import create_renderer

        renderer = create_renderer('vico')

        ts = torch.randn(2, 100, 1)
        att_mask = torch.ones(2, 100, dtype=torch.bool)
        att_mask[1, 50:] = False  # 第二个样本 padding

        output = renderer.render_batch(ts, att_mask=att_mask, img_size=224)

        assert output.shape == (2, 3, 224, 224)
```

**Step 3: 运行测试**

```bash
python -m pytest tests/datasets/renderers/test_vico_renderer.py -v
```
Expected: 所有测试通过

**Step 4: 提交**

```bash
git add tests/datasets/renderers/
git commit -m "test: add unit tests for RendererRegistry and ViCORenderer"
```

---

## Task 6: 更新配置文件

**Files:**
- Modify: `configs/base.yaml`

**Step 1: 添加 vision_branch 配置块**

在 `configs/base.yaml` 的 `model:` 部分添加：

```yaml
# ==================== 模型配置 ====================
model:
  model_name: VETime-query-FGVA
  vision_name: mae_visualize_base.pth
  ts_finetune_type: lora
  use_vectorized_fold: true
  use_gradient_checkpointing: false

  # ==================== 视觉双分支配置 ====================
  vision_branch:
    vico_renderer: vico              # 渲染器类型：vico / gaf / stft / ...
    renderer_params: {}              # 渲染器特定参数（预留扩展）

  # ==================== 解码器配置 ====================
  use_query_decoder: true
```

**Step 2: 验证配置可读取**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/base.yaml')
print('vico_renderer:', cfg.model.vision_branch.vico_renderer)
"
```
Expected: `vico_renderer: vico`

**Step 3: 提交**

```bash
git add configs/base.yaml
git commit -m "feat: add vision_branch config for vico_renderer selection"
```

---

## Task 7: 改造 Trainer 集成渲染器

**Files:**
- Modify: `src/engines/trainer.py`

**Step 1: 添加渲染器导入**

在文件顶部导入区域添加：

```python
# src/engines/trainer.py
# ... 现有导入 ...

from src.datasets.renderers import create_renderer  # 新增
```

**Step 2: 修改 Trainer.__init__ 添加渲染器初始化**

在 `__init__` 方法中添加（约第 41 行后）：

```python
def __init__(self, cfg, model, train_loader, val_loader, accelerator,
             data_setting, patch_size):
    # ... 现有代码 ...
    self.patch_size = patch_size

    self.global_step = 0
    # ...

    # === 新增：初始化 ViCO 渲染器 ===
    renderer_name = self._get_renderer_name(cfg)
    self.vico_renderer = create_renderer(renderer_name)
    logger.info(f"ViCO 渲染器: {self.vico_renderer}")

    # ... 后续代码 ...
```

**Step 3: 添加 _get_renderer_name 辅助方法**

在 `__init__` 方法后添加：

```python
def _get_renderer_name(self, cfg) -> str:
    """从配置中获取渲染器名称，默认 'vico'。"""
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'vision_branch'):
        return getattr(cfg.model.vision_branch, 'vico_renderer', 'vico')
    return 'vico'
```

**Step 4: 替换 train_epoch 中的 render_vico_batch 调用**

找到 `train_epoch` 方法中的调用点（约第 328 行和第 385 行）：

```python
# 改造前：
# images_vico_chunk = render_vico_batch(ts_raw_part, att_mask=att_mask_part)
# images_vico = render_vico_batch(time_series_raw, att_mask=att_mask)

# 改造后：
images_vico_chunk = self.vico_renderer.render_batch(
    ts_raw_part, att_mask=att_mask_part, img_size=self.img_size
)
# ...
images_vico = self.vico_renderer.render_batch(
    time_series_raw, att_mask=att_mask, img_size=self.img_size
)
```

**Step 5: 替换 _evaluate_split 中的 render_vico_batch 调用**

找到 `_evaluate_split` 方法中的调用点（约第 610 行和第 641 行）：

```python
# 改造前：
# images_vico_chunk = render_vico_batch(ts_raw_part, att_mask=att_mask_part)
# images_vico = render_vico_batch(time_series_raw, att_mask=att_mask)

# 改造后：
images_vico_chunk = self.vico_renderer.render_batch(
    ts_raw_part, att_mask=att_mask_part, img_size=self.img_size
)
# ...
images_vico = self.vico_renderer.render_batch(
    time_series_raw, att_mask=att_mask, img_size=self.img_size
)
```

**Step 6: 移除旧的 render_vico_batch 导入**

删除顶部的：
```python
from src.datasets.pre_image import render_vico_batch  # 删除这行
```

**Step 7: 验证语法正确**

```bash
python -c "from src.engines.trainer import Trainer; print('Trainer imported successfully')"
```
Expected: `Trainer imported successfully`

**Step 8: 提交**

```bash
git add src/engines/trainer.py
git commit -m "feat: integrate renderer into Trainer, replace render_vico_batch calls"
```

---

## Task 8: 集成测试

**Files:**
- Create: `tests/integration/test_renderer_integration.py`

**Step 1: 编写集成测试**

```python
# tests/integration/test_renderer_integration.py
"""渲染器集成测试：验证 Trainer 正确使用渲染器。"""

import pytest
import torch
from unittest.mock import MagicMock, patch


def test_trainer_uses_configured_renderer():
    """测试 Trainer 使用配置指定的渲染器。"""
    from omegaconf import OmegaConf
    from src.engines.trainer import Trainer

    # 创建 mock 对象
    cfg = OmegaConf.create({
        'model': {
            'model_name': 'test',
            'vision_branch': {'vico_renderer': 'vico'}
        },
        'training': {'total_epochs': 1, 'stage1_epochs': 0, 'early_stopping': {'patience': 1}},
        'loss': {},
    })

    model = MagicMock()
    model.MAX_L = 5000
    train_loader = MagicMock()
    val_loader = MagicMock()
    accelerator = MagicMock()
    accelerator.device = 'cpu'
    data_setting = {'img_size': 224}
    patch_size = 16

    trainer = Trainer(
        cfg, model, train_loader, val_loader, accelerator,
        data_setting, patch_size
    )

    # 验证渲染器已初始化
    from src.datasets.renderers import ViCORenderer
    assert isinstance(trainer.vico_renderer, ViCORenderer)


def test_trainer_default_renderer_without_config():
    """测试无配置时默认使用 vico 渲染器。"""
    from omegaconf import OmegaConf
    from src.engines.trainer import Trainer

    cfg = OmegaConf.create({
        'model': {'model_name': 'test'},
        'training': {'total_epochs': 1, 'stage1_epochs': 0, 'early_stopping': {'patience': 1}},
        'loss': {},
    })

    model = MagicMock()
    model.MAX_L = 5000
    train_loader = MagicMock()
    val_loader = MagicMock()
    accelerator = MagicMock()
    accelerator.device = 'cpu'
    data_setting = {'img_size': 224}
    patch_size = 16

    trainer = Trainer(
        cfg, model, train_loader, val_loader, accelerator,
        data_setting, patch_size
    )

    from src.datasets.renderers import ViCORenderer
    assert isinstance(trainer.vico_renderer, ViCORenderer)
```

**Step 2: 运行集成测试**

```bash
python -m pytest tests/integration/test_renderer_integration.py -v
```
Expected: 所有测试通过

**Step 3: 提交**

```bash
git add tests/integration/test_renderer_integration.py
git commit -m "test: add integration tests for renderer in Trainer"
```

---

## Task 9: 最终验证与清理

**Step 1: 运行完整测试套件**

```bash
python -m pytest tests/datasets/renderers/ tests/integration/test_renderer_integration.py -v
```
Expected: 所有测试通过

**Step 2: 验证训练入口正常工作**

```bash
python -c "
from train import train_univariate_hydra
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/base.yaml')
print('Config loaded, vico_renderer:', cfg.model.vision_branch.vico_renderer)
print('Integration OK')
"
```
Expected: 无错误输出

**Step 3: 更新 MEMORY.md 记录新架构**

在 `/home/cjm/.claude/projects/-mnt-sda-cjmProject-VETime/memory/MEMORY.md` 添加：

```markdown
## ViCO 渲染器解耦（2026-07-31）
- 新增 `src/datasets/renderers/` 模块：BaseRenderer + RendererRegistry
- 配置切换：`model.vision_branch.vico_renderer`
- 扩展：新建渲染器文件 + `@RendererRegistry.register('name')`
```

**Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: complete ViCO renderer decoupling implementation"
```

---

## 完成检查清单

- [ ] `src/datasets/renderers/base.py` - BaseRenderer 抽象基类
- [ ] `src/datasets/renderers/__init__.py` - RendererRegistry + create_renderer
- [ ] `src/datasets/renderers/vico.py` - ViCORenderer 包装现有实现
- [ ] `tests/datasets/renderers/` - 单元测试
- [ ] `configs/base.yaml` - vision_branch 配置块
- [ ] `src/engines/trainer.py` - 集成渲染器
- [ ] `tests/integration/test_renderer_integration.py` - 集成测试
- [ ] MEMORY.md 更新
