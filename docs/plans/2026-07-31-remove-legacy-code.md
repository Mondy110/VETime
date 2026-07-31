# Legacy Code Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove legacy code from model/, loss/ directories and train.py, converting them to re-export compatibility layers for production code migration.

**Architecture:** Delete implementation files, create lightweight re-export modules (model/__init__.py, loss/__init__.py), and remove old train_univariate() entry point from train.py.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf

---

## Task 1: Verify src/ Completeness

**Files:**
- Check: `model/VETime.py`, `src/models/vetime.py`
- Check: `model/VTS_module.py`, `src/models/vts_module.py`
- Check: `loss/loss.py`, `src/losses/*.py`

**Step 1: Compare model exports**

Run:
```bash
# 列出 model/ 中的主要类和函数
grep -E "^(class|def) " model/VETime.py model/VTS_module.py | head -20
```

Expected: 列出所有类和函数定义

**Step 2: Compare loss exports**

Run:
```bash
# 列出 loss/loss.py 中的类和函数
grep -E "^(class|def) " loss/loss.py | head -20
```

Expected: 列出所有损失函数类

**Step 3: Verify src.models completeness**

Run:
```bash
# 列出 src/models/ 中的主要导出
grep -E "^(class|def) " src/models/vetime.py src/models/vts_module.py | head -20
```

Expected: 应包含 model/ 中所有对应的类

**Step 4: Verify src.losses completeness**

Run:
```bash
# 列出 src/losses/ 中的导出
grep -E "^(class|def) " src/losses/*.py | head -20
```

Expected: 应包含 loss/loss.py 中所有对应的函数

**Step 5: Document gaps**

如果发现缺失的功能，记录下来：
- 缺失的类/函数
- 不兼容的接口签名

---

## Task 2: Create loss/ Re-export Layer

**Files:**
- Create: `loss/__init__.py` (backup old, create new)
- Backup: `loss/loss.py` → `loss/loss.py.bak`

**Step 1: Backup existing loss files**

Run:
```bash
cp loss/loss.py loss/loss.py.bak
cp loss/__init__.py loss/__init__.py.bak 2>/dev/null || echo "No __init__.py to backup"
```

Expected: 备份文件创建成功

**Step 2: Identify all exports from loss/loss.py**

Run:
```bash
# 提取所有类和公共函数
grep -E "^(class|def [^_])" loss/loss.py.bak
```

Expected: 列出需要重导出的类和函数

**Step 3: Write new loss/__init__.py**

```python
"""
DEPRECATED: This module now re-exports from src.losses for backward compatibility.
New code should import directly from src.losses:
    from src.losses import ContrastiveLoss, BalanceLoss

This module provides loss functions for VETime anomaly detection.
"""

# Backward compatibility: re-export from new location
# Note: Adjust imports based on actual function names in src/losses/
from src.losses.contrastive import win_Contrastive_Loss
from src.losses.balance import load_balance_loss

# Add other loss functions as needed
# from src.losses.anomaly import ...
# from src.losses.reconstruction import ...

__all__ = [
    'win_Contrastive_Loss',
    'load_balance_loss',
]
```

**Step 4: Test backward compatibility**

Run:
```bash
python -c "from loss.loss import win_Contrastive_Loss, load_balance_loss; print('OK')"
```

Expected: 应该报错（因为删除了 loss.py），用于验证下一步

**Step 5: Update import path for backward compat**

修改 `loss/__init__.py`，确保旧路径导入也能工作：

```python
# For backward compatibility with "from loss.loss import X"
import sys
import importlib.util

# Create a module alias for loss.loss that points to this module
# This allows both "from loss import X" and "from loss.loss import X"
```

实际上，更简单的方法是检查旧代码如何导入：

Run:
```bash
grep -n "from loss" train.py Test_TSB.py test_dual_branch_integration.py
```

Expected: 找出所有 loss 导入路径

**Step 6: Delete loss/loss.py**

Run:
```bash
rm loss/loss.py
```

**Step 7: Test old test files still work**

Run:
```bash
python -c "from loss import win_Contrastive_Loss, load_balance_loss; print('Direct import OK')"
python -c "from loss.loss import win_Contrastive_Loss; print('Submodule import OK')" 2>/dev/null || echo "Submodule import not needed"
```

Expected: 至少直接导入应该工作

**Step 8: Commit loss/ changes**

```bash
git add loss/
git commit -m "refactor: convert loss/ to re-export compatibility layer

- Remove loss/loss.py implementation
- Update loss/__init__.py to re-export from src.losses
- Maintain backward compatibility for old test files

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Analyze model/ Dependencies

**Files:**
- Check: `model/` directory structure
- Check: Test file imports

**Step 1: List model/ structure**

Run:
```bash
find model/ -name "*.py" -type f | sort
```

Expected: 列出所有 Python 文件

**Step 2: Find all imports from model/**

Run:
```bash
grep -rn "from model\." --include="*.py" . | grep -v ".bak" | grep -v ".git"
```

Expected: 列出所有依赖 model/ 的文件和行号

**Step 3: Identify Vision_encoder imports**

Run:
```bash
grep -n "from model.Vision_encoder" Test_TSB.py test_dual_branch_integration.py train.py
```

Expected: 找出视觉编码器导入

**Step 4: Identify TS_encoder imports**

Run:
```bash
grep -n "from model.TS_encoder" Test_TSB.py test_dual_branch_integration.py train.py
```

Expected: 找出时序编码器导入

**Step 5: Identify VETime imports**

Run:
```bash
grep -n "from model.VETime import\|from model import VETime" Test_TSB.py test_dual_branch_integration.py train.py
```

Expected: 找出 VETIME 模型导入

---

## Task 4: Create model/ Re-export Layer

**Files:**
- Create: `model/__init__.py` (backup and replace)
- Delete: `model/VETime.py`
- Delete: `model/VTS_module.py`
- Delete: `model/Vision_encoder/` directory
- Delete: `model/TS_encoder/` directory

**Step 1: Backup model/ files**

Run:
```bash
# 备份整个目录（以防需要回退）
tar -czf model_backup_$(date +%Y%m%d_%H%M%S).tar.gz model/
```

Expected: 创建备份压缩包

**Step 2: Write new model/__init__.py**

```python
"""
DEPRECATED: This module now re-exports from src.models for backward compatibility.
New code should import directly from src.models:
    from src.models import VETIME, VTS_Module
    from src.models.vision_encoder import V_model
    from src.models.ts_encoder import TS_Model

This module provides the VETime model and its components for
multimodal time series anomaly detection.
"""

# Main models
from src.models.vetime import VETIME
from src.models.vts_module import VTS_Module

# Vision encoder components
from src.models.vision_encoder.v_encoder import V_model
from src.models.vision_encoder.vit4ad import Vit4AD
# Note: models_mae may not need to be exported, check actual usage

# Time series encoder components
from src.models.ts_encoder.ts_encoder import TimeSeriesEncoder
from src.models.ts_encoder.ts_model import TS_Model
from src.models.ts_encoder.config import TimeSeriesConfig, default_config_t

__all__ = [
    'VETIME',
    'VTS_Module',
    'V_model',
    'Vit4AD',
    'TimeSeriesEncoder',
    'TS_Model',
    'TimeSeriesConfig',
    'default_config_t',
]
```

**Step 3: Test new __init__.py before deleting files**

Run:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from model import VETIME, V_model, TS_Model
print('model/__init__.py OK')
"
```

Expected: 打印成功消息

**Step 4: Delete model/VETime.py**

Run:
```bash
rm model/VETime.py
```

**Step 5: Delete model/VTS_module.py**

Run:
```bash
rm model/VTS_module.py
```

**Step 6: Delete model/Vision_encoder/ directory**

Run:
```bash
rm -rf model/Vision_encoder/
```

**Step 7: Delete model/TS_encoder/ directory**

Run:
```bash
rm -rf model/TS_encoder/
```

**Step 8: Verify only __init__.py remains**

Run:
```bash
find model/ -name "*.py" -type f
```

Expected: 只显示 `model/__init__.py`

**Step 9: Test old test files import**

Run:
```bash
# 测试常见的导入路径
python -c "
from model.VETime import VETIME
from model.Vision_encoder.V_encoder import V_model
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
print('All imports OK')
"
```

Expected: 所有导入应该成功

**Step 10: Commit model/ changes**

```bash
git add model/
git commit -m "refactor: convert model/ to re-export compatibility layer

- Remove model/VETime.py implementation
- Remove model/VTS_module.py implementation
- Remove model/Vision_encoder/ and model/TS_encoder/ directories
- Update model/__init__.py to re-export from src.models
- Maintain backward compatibility for old test files

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Clean train.py Old Entry Point

**Files:**
- Modify: `train.py` (remove old imports and train_univariate function)

**Step 1: Identify old imports to remove**

Run:
```bash
grep -n "^from model\.\|^from loss\.\|^from dataset\." train.py | head -20
```

Expected: 列出需要删除的旧导入行号

**Step 2: Identify train_univariate function location**

Run:
```bash
grep -n "^def train_univariate\|^def create_model" train.py
```

Expected: 显示函数定义的行号

**Step 3: Check what to preserve**

Run:
```bash
# 检查哪些函数被 train_univariate_hydra 使用
grep -n "set_seed\|seed_worker\|load_config" train.py
```

Expected: 找出共用函数

**Step 4: Remove old imports (lines 29-39)**

Edit `train.py`, delete these lines:
```python
from model.Vision_encoder.V_encoder import V_model
from loss.loss import load_balance_loss
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
from dataset.dataloader import AnomalyDataset, collate_fn, DynamicLengthBatchSampler
from dataset.pre_image import render_vico_batch
from model.VETime import VETIME
```

**Step 5: Remove create_model function**

找到 `create_model()` 函数（大约91-189行），完整删除。

**Step 6: Remove train_univariate function**

找到 `train_univariate()` 函数，完整删除整个函数（包括函数体）。

**Step 7: Update __main__ entry**

检查文件末尾的 `if __name__ == "__main__":` 部分，确保只调用 hydra 版本：

```python
if __name__ == "__main__":
    # Use Hydra for configuration
    from omegaconf import OmegaConf
    import hydra

    @hydra.main(config_path="configs", config_name="base", version_base=None)
    def main(cfg):
        train_univariate_hydra(cfg)

    main()
```

**Step 8: Verify train.py syntax**

Run:
```bash
python -m py_compile train.py && echo "Syntax OK"
```

Expected: 打印 "Syntax OK"

**Step 9: Verify imports are only from src/**

Run:
```bash
grep -n "^from model\.\|^from loss\.\|^import model\.\|^import loss\." train.py
```

Expected: 应该没有输出（所有旧导入已删除）

**Step 10: Commit train.py changes**

```bash
git add train.py
git commit -m "refactor: remove legacy train_univariate entry point

- Delete old imports from model/, loss/, dataset/
- Remove create_model() function (now in hydra entry)
- Remove train_univariate() function
- Keep only train_univariate_hydra() entry point
- Update __main__ to use Hydra decorator

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Check and Update src/datasets/pre_image.py

**Files:**
- Check: `src/datasets/pre_image.py`

**Step 1: Check for old imports in pre_image.py**

Run:
```bash
grep -n "from model\.\|from loss\.\|import model\.\|import loss\." src/datasets/pre_image.py
```

Expected: 应该没有输出，如果有则需要修复

**Step 2: Fix any old imports if found**

如果有旧导入，编辑文件替换为：
- `from model.X` → `from src.models.X`
- `from loss.X` → `from src.losses.X`

**Step 3: Commit if changes made**

如果有修改：
```bash
git add src/datasets/pre_image.py
git commit -m "fix: update imports in pre_image.py to use src.*

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Final Verification

**Files:**
- Test all production code imports
- Test old test files compatibility

**Step 1: Verify no old imports in src/**

Run:
```bash
grep -rn "from model\.\|from loss\.\|import model\.\|import loss\." src/ --include="*.py"
```

Expected: 应该没有输出

**Step 2: Verify train.py imports**

Run:
```bash
python -c "
import sys
sys.path.insert(0, '.')
# 检查 train.py 中的导入是否可用
from src.utils.seed import seed_everything
from src.engines.trainer import Trainer
from src.models.vetime import VETIME
from src.models.vision_encoder.v_encoder import V_model
print('All src imports OK')
"
```

Expected: 打印成功消息

**Step 3: Test old test files still work**

Run:
```bash
python -c "
# 模拟 Test_TSB.py 的导入
from model.VETime import VETIME
from model.Vision_encoder.V_encoder import V_model
from loss.loss import win_Contrastive_Loss
print('Old test file imports OK')
"
```

Expected: 打印成功消息（通过兼容层）

**Step 4: Check for any remaining old code**

Run:
```bash
# 检查是否有其他文件引用旧路径
grep -rn "from model\.\|from loss\." --include="*.py" . | grep -v "test_dual_branch\|Test_TSB\|\.bak\|\.git"
```

Expected: 应该没有输出（除了测试文件）

**Step 5: Run a quick smoke test**

Run:
```bash
# 尝试导入主要模块
python -c "
from src.models.vetime import VETIME
from src.models.vision_encoder.v_encoder import V_model
from src.models.ts_encoder.ts_model import TS_Model
from src.losses.contrastive import win_Contrastive_Loss
from src.losses.balance import load_balance_loss
print('All new path imports OK')

# 测试兼容层
from model import VETIME as VETIME_old
from loss import win_Contrastive_Loss as loss_old
print('Compatibility layer OK')
"
```

Expected: 所有导入成功

**Step 6: Final commit summary**

```bash
git status
```

Expected: 工作区干净，没有未提交的更改

---

## Task 8: Update Documentation

**Files:**
- Update: `docs/plans/2026-07-31-remove-legacy-code-design.md` (if needed)
- Update: `README.md` (if mentions old paths)

**Step 1: Check README for old paths**

Run:
```bash
grep -n "model/\|loss/\|dataset/" README.md | head -10
```

Expected: 检查是否有需要更新的说明

**Step 2: Update README if needed**

如果有旧路径说明，更新为新的 src/ 路径示例。

**Step 3: Commit doc updates**

如果有修改：
```bash
git add README.md docs/
git commit -m "docs: update documentation to reflect src/ migration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `model/` directory contains only `__init__.py` (re-export layer)
- [ ] `loss/` directory contains only `__init__.py` (re-export layer)
- [ ] `train.py` has no imports from `model.`, `loss.`, `dataset.`
- [ ] `train_univariate()` function removed from `train.py`
- [ ] `src/` modules have no imports from old paths
- [ ] Old test files (`Test_TSB.py`, `test_dual_branch_integration.py`) can still import from old paths
- [ ] All commits include proper Co-Authored-By line

---

## Rollback Plan

If issues arise, restore from backups:

```bash
# Restore model/
tar -xzf model_backup_*.tar.gz

# Restore loss/
cp loss/loss.py.bak loss/loss.py
cp loss/__init__.py.bak loss/__init__.py

# Revert train.py changes
git checkout HEAD -- train.py
```
