# ViCO 分支图像生成解耦设计

**日期**: 2026-07-31
**状态**: 设计完成，待实现

---

## 1. 背景

VETime 采用视觉双分支架构：
- **分支 A**: VETime 时域图像（`ts2image_*` 函数生成）
- **分支 B**: ViCO 频域图像（`vico_render_timeseries` 函数生成）

当前问题：ViCO 分支的图像生成逻辑硬编码在 `pre_image.py` 和 `trainer.py` 中，无法灵活切换不同的图像生成方式。

**目标**: 将 ViCO 分支的图像生成解耦为可配置、可扩展的渲染器模块。

---

## 2. 设计方案

采用 **策略模式 + 注册表** 实现渲染器解耦。

### 2.1 核心组件

| 组件 | 职责 |
|------|------|
| `BaseRenderer` | 抽象基类，定义统一接口 |
| `RendererRegistry` | 注册表，管理渲染器类的注册与获取 |
| `create_renderer()` | 工厂函数，创建渲染器实例 |
| 具体渲染器 | 实现特定图像生成算法 |

### 2.2 目录结构

```
src/
├── datasets/
│   ├── renderers/                    # 新增
│   │   ├── __init__.py               # RendererRegistry + 便捷导入
│   │   ├── base.py                   # BaseRenderer 抽象基类
│   │   └── vico.py                   # ViCO 渲染器（包装现有实现）
│   └── pre_image.py                  # 保留（VETime 时域渲染 + ViCO 底层函数）
```

### 2.3 接口定义

```python
class BaseRenderer(ABC):
    @abstractmethod
    def render_batch(
        self,
        time_series: torch.Tensor,    # [B, T, F]
        att_mask: Optional[torch.Tensor] = None,  # [B, T]
        img_size: int = 224,
    ) -> torch.Tensor:                 # [B, 3, img_size, img_size]
        pass
```

### 2.4 注册表机制

```python
@RendererRegistry.register('vico')
class ViCORenderer(BaseRenderer):
    ...

@RendererRegistry.register('gaf')
class GAFRenderer(BaseRenderer):
    ...

# 获取渲染器
renderer = create_renderer('vico')
```

---

## 3. 配置集成

```yaml
# configs/base.yaml
model:
  vision_branch:
    vico_renderer: vico              # 渲染器类型
    renderer_params: {}              # 预留参数扩展
```

---

## 4. Trainer 改造

### 4.1 初始化

```python
class Trainer:
    def __init__(self, ...):
        # 新增：初始化渲染器
        renderer_name = self._get_renderer_name(cfg)
        self.vico_renderer = create_renderer(renderer_name)
```

### 4.2 调用点替换

```python
# 改造前
images_vico = render_vico_batch(time_series_raw, att_mask=att_mask)

# 改造后
images_vico = self.vico_renderer.render_batch(
    time_series_raw, att_mask=att_mask, img_size=self.img_size
)
```

### 4.3 改动范围

| 文件 | 改动 |
|------|------|
| `src/engines/trainer.py` | `__init__` 初始化渲染器；`train_epoch`/`_evaluate_split` 替换调用 |
| `configs/base.yaml` | 新增 `model.vision_branch` 配置块 |

---

## 5. 扩展流程（添加新渲染器）

1. 新建 `src/datasets/renderers/xxx.py`
2. 实现 `BaseRenderer` 接口
3. 添加 `@RendererRegistry.register('xxx')` 装饰器
4. 在 `__init__.py` 中导入
5. 配置文件设置 `vico_renderer: xxx`

---

## 6. 向后兼容

- 默认渲染器为 `vico`，行为与现有代码完全一致
- `pre_image.py` 中的底层函数（`vico_render_timeseries`, `find_period_fft` 等）保留不变
- 未配置 `vision_branch` 时自动使用 `vico` 渲染器

---

## 7. 实现计划

1. 创建 `src/datasets/renderers/` 目录及基础文件
2. 实现 `BaseRenderer` 和 `RendererRegistry`
3. 迁移 ViCO 渲染器（包装现有代码）
4. 改造 `Trainer` 集成渲染器
5. 更新配置文件
6. 编写单元测试
