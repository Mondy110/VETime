# 旧代码清理设计方案

## 概述

本文档描述了 VETime 项目旧代码清理的详细设计方案。项目已完成从高耦合状态到标准深度学习项目架构的重构，现在需要移除保留的旧代码，实现生产代码的完全迁移。

## 背景

### 项目重构历史
- 项目已从 `model/`, `loss/`, `dataset/` 结构重构为标准的 `src/` 模块化架构
- 重构后保留了旧代码以确保向后兼容
- 新的训练入口 `train_univariate_hydra()` 已验证可用

### 当前状态
- **新代码位置**: `src/` 目录下的模块化架构
- **旧代码位置**: `model/`, `loss/` 目录
- **训练入口**: `train.py` 中存在两个入口
  - `train_univariate()` - 使用旧路径导入
  - `train_univariate_hydra()` - 使用新路径导入

## 设计目标

### 成功标准
- 所有生产代码（train.py, src/*）只使用 `src.*` 导入路径
- 旧测试文件（Test_TSB.py, test_dual_branch_integration.py）仍然可以运行
- 项目没有重复的实现代码

### 约束条件
- 保留旧测试文件不变
- 保持模型权重 100% 兼容（nn.Module 命名不变）

## 设计方案

### 方案选择：重导出兼容层

**核心策略**：将 `model/` 和 `loss/` 目录转换为轻量的重导出模块，删除实现代码，保留兼容性。

#### 优点
1. 旧测试文件无需修改即可运行
2. 清理彻底，生产代码完全使用新架构
3. 兼容层代码量小，易于维护
4. 已有成功先例（dataset/dataloader.py）

## 详细设计

### 1. train.py 清理

**目标**: 删除旧的训练入口及相关代码

**删除内容**:
- 旧路径导入语句（29-39行）
- `create_model()` 函数（91-189行）
- `train_univariate()` 函数（完整删除）
- 仅被旧入口使用的辅助函数

**保留内容**:
- `train_univariate_hydra()` 函数及其依赖
- 共用的辅助函数（如 `set_seed()`, `seed_worker()`）
- `__main__` 入口

**预期结果**:
- train.py 变成简洁的入口文件
- 不再依赖 `model/`, `loss/` 路径

### 2. model/ 目录重构

**目标**: 从实现包转换为重导出兼容层

**删除内容**:
- `model/VETime.py`
- `model/VTS_module.py`
- `model/Vision_encoder/` 目录（包括所有子文件）
- `model/TS_encoder/` 目录（包括所有子文件）

**创建兼容层**:
- 保留 `model/__init__.py`，内容改为重导出：

```python
"""
DEPRECATED: This module now re-exports from src.models for backward compatibility.
New code should import directly from src.models:
    from src.models import VETIME, VTS_Module
"""

# Main models
from src.models.vetime import VETIME
from src.models.vts_module import VTS_Module

# Vision encoder
from src.models.vision_encoder.v_encoder import V_model
from src.models.vision_encoder.vit4ad import Vit4AD

# TS encoder
from src.models.ts_encoder.ts_encoder import TimeSeriesEncoder
from src.models.ts_encoder.ts_model import TS_Model
from src.models.ts_encoder.config import TimeSeriesConfig

__all__ = [
    'VETIME',
    'VTS_Module',
    'V_model',
    'Vit4AD',
    'TimeSeriesEncoder',
    'TS_Model',
    'TimeSeriesConfig'
]
```

**注意**: 需要确保 `src/models/` 中包含所有 `model/` 中导出的类和函数。

### 3. loss/ 目录重构

**目标**: 从实现包转换为重导出兼容层

**删除内容**:
- `loss/loss.py`

**创建兼容层**:
- 保留 `loss/__init__.py`，内容改为重导出：

```python
"""
DEPRECATED: This module now re-exports from src.losses for backward compatibility.
New code should import directly from src.losses:
    from src.losses import ContrastiveLoss, BalanceLoss
"""

from src.losses.contrastive import win_Contrastive_Loss
from src.losses.balance import load_balance_loss

__all__ = ['win_Contrastive_Loss', 'load_balance_loss']
```

**注意**: 需要先确认 `src/losses/` 中包含所有 `loss/loss.py` 中的功能。

### 4. dataset/ 目录

**现状**: `dataset/dataloader.py` 已经是重导出兼容层

**操作**: 保持不变，无需修改

### 5. 其他文件更新

**需要检查的文件**:
- `src/datasets/pre_image.py` - 检查是否引用了 `model.*` 或 `loss.*`
- `tests/test_datasets.py` - 检查导入路径

**更新策略**:
- 如果 `src/datasets/pre_image.py` 引用了旧路径，更新为 `src.*`
- `tests/test_datasets.py` 保持不变（用户要求保留旧测试文件）

## 验证计划

### 验证步骤
1. 运行新训练入口确认正常工作
2. 运行旧测试文件确认兼容层有效
3. 检查是否有其他文件引用旧路径
4. 确认项目没有重复的实现代码

### 验证命令
```bash
# 测试新训练入口
python train.py --config-name base

# 测试旧测试文件
python Test_TSB.py
python test_dual_branch_integration.py

# 检查旧路径引用
grep -r "from model\." --include="*.py" src/
grep -r "from loss\." --include="*.py" src/
```

## 风险与缓解

### 风险
1. **接口不兼容**: 新旧代码的接口可能不完全一致
2. **遗漏的依赖**: 可能存在未发现的旧路径引用
3. **测试失败**: 兼容层可能导致测试失败

### 缓解措施
1. **仔细比对接口**: 在删除前逐一确认接口兼容性
2. **全面搜索**: 使用 grep 搜索所有可能的旧路径引用
3. **逐步清理**: 先清理一个模块，验证后再清理下一个

## 实施顺序

1. **检查阶段**: 确认 src/ 中包含所有需要的功能
2. **train.py 清理**: 删除旧入口，验证新入口可用
3. **loss/ 重构**: 创建兼容层，测试旧文件
4. **model/ 重构**: 创建兼容层，测试旧文件
5. **最终验证**: 运行所有测试，确认清理完成

## 后续工作

清理完成后，未来可以考虑：
1. 逐步迁移旧测试文件到新路径
2. 最终删除兼容层（当所有代码都迁移到 src.* 后）
3. 更新文档和注释

## 参考资料

- 项目记忆文档: `/home/cjm/.claude/projects/-mnt-sda-cjmProject-VETime/memory/MEMORY.md`
- 项目架构说明: 2026-07-09 重构记录
