# VETime 项目重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 VETime 从高耦合状态重构为标准深度学习项目架构，数据-模型-训练三方解耦，配置与代码分离。

**Architecture:** 渐进式抽离 5 步走：先 utils → 再数据 → 再模型 → 再训练引擎 → 最后评估。每步完成后可验证，旧代码始终可用作参照。

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, Hydra + OmegaConf, logging, TensorBoard

**重要约束：**
- VETime.py 的 `__init__` 中 nn.Module/nn.Parameter 命名不变，权重 100% 兼容
- PCGrad 保持死代码不激活
- train_multivariate 暂不重构
- 旧 `train.py`、`Test_TSB.py` 保留到全部迁移完成后再删除

---

## Task 1: 创建项目基础设施与 src/utils/

**Files:**
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/seed.py`
- Create: `src/utils/logger.py`
- Create: `src/utils/checkpoint.py`
- Test: `tests/test_utils.py`

**Step 1: 创建目录结构和 __init__.py**

```bash
mkdir -p src/utils tests
touch src/__init__.py src/utils/__init__.py tests/__init__.py
```

**Step 2: 写 seed.py 的测试**

Create `tests/test_utils.py`:

```python
import torch
import numpy as np
import random


def test_seed_everything_reproducible():
    from src.utils.seed import seed_everything

    seed_everything(42)
    a = random.random()
    b = np.random.randn()
    c = torch.randn(3)

    seed_everything(42)
    d = random.random()
    e = np.random.randn()
    f = torch.randn(3)

    assert a == d, "Python random not reproducible"
    assert np.allclose(b, e), "NumPy random not reproducible"
    assert torch.allclose(c, f), "PyTorch random not reproducible"


def test_seed_worker():
    from src.utils.seed import seed_worker

    # seed_worker 不应抛异常
    seed_worker(0)
    seed_worker(3)
```

**Step 3: 运行测试确认失败**

```bash
cd /mnt/sda/cjmProject/VETime && python -m pytest tests/test_utils.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src'`

**Step 4: 实现 seed.py**

Create `src/utils/seed.py`:

```python
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """设置所有随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """DataLoader worker 的种子初始化函数。"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**Step 5: 写 logger.py 的测试**

Append to `tests/test_utils.py`:

```python
import logging


def test_get_logger_returns_logger():
    from src.utils.logger import get_logger

    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_has_handler():
    from src.utils.logger import get_logger

    logger = get_logger("test_handler")
    assert len(logger.handlers) > 0
```

**Step 6: 实现 logger.py**

Create `src/utils/logger.py`:

```python
import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取带统一格式的 logger。

    Args:
        name: logger 名称（通常用 __name__）
        level: 日志级别

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
```

**Step 7: 写 checkpoint.py 的测试**

Append to `tests/test_utils.py`:

```python
import tempfile


def test_save_load_checkpoint_roundtrip():
    from src.utils.checkpoint import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        data = {"epoch": 5, "loss": 0.123, "tags": ["a", "b"]}
        save_checkpoint(data, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert abs(loaded["loss"] - 0.123) < 1e-6
        assert loaded["tags"] == ["a", "b"]
```

需要在文件顶部加 `import os`。

**Step 8: 实现 checkpoint.py**

Create `src/utils/checkpoint.py`:

```python
import os
import torch
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(state: Dict[str, Any], path: str):
    """保存 checkpoint 到磁盘。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint 已保存: {path}")


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """从磁盘加载 checkpoint。"""
    assert os.path.isfile(path), f"Checkpoint 不存在: {path}"
    state = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"Checkpoint 已加载: {path}")
    return state
```

**Step 9: 运行所有测试确认通过**

```bash
cd /mnt/sda/cjmProject/VETime && python -m pytest tests/test_utils.py -v
```

Expected: 全部 PASS

**Step 10: 提交**

```bash
git add src/ tests/test_utils.py
git commit -m "feat: add src/utils/ with seed, logger, checkpoint utilities"
```

---

## Task 2: 迁移数据管线 → src/datasets/

**Files:**
- Create: `src/datasets/__init__.py`
- Create: `src/datasets/anomaly_dataset.py` (从 `dataset/dataloader.py` 迁移 AnomalyDataset)
- Create: `src/datasets/collate.py` (从 `dataset/dataloader.py` 迁移 collate_fn + DynamicLengthBatchSampler)
- Create: `src/datasets/masking.py` (从 `dataset/dataloader.py` 分离 create_random_mask)
- Create: `src/datasets/pre_image.py` (从 `dataset/pre_image.py` 迁移)
- Modify: `dataset/dataloader.py` (改为从 src 导入，保持向后兼容)
- Test: `tests/test_datasets.py`

**Step 1: 创建目录结构**

```bash
mkdir -p src/datasets
touch src/datasets/__init__.py
```

**Step 2: 迁移 pre_image.py**

```bash
cp dataset/pre_image.py src/datasets/pre_image.py
```

然后修改 `src/datasets/pre_image.py` 的 import：将 `from dataset.pre_image import ...` 类引用改为相对路径（实际代码中 pre_image.py 不导入自身模块，所以只需检查内部是否有 `dataset.` 前缀的 import，如有则移除）。

**Step 3: 分离 create_random_mask → masking.py**

Create `src/datasets/masking.py`:

从 `dataset/dataloader.py:346-410` 提取 `create_random_mask` 函数，保持代码完全不变，仅修改函数签名增加默认参数文档：

```python
"""自监督掩码生成模块。"""

import torch
import numpy as np


def create_random_mask(time_series, attention_mask, patch_size=14, mask_ratio=0.3):
    """
    生成 patch 级别的随机掩码，被掩码位置用高斯噪声替换。

    Args:
        time_series: [B, L, C] 时序数据
        attention_mask: [B, L] 注意力掩码（1=有效，0=padding）
        patch_size: patch 大小
        mask_ratio: 掩码比例

    Returns:
        mask_time_series: 掩码后的时序数据 [B, L, C]
        mask: 掩码标记 [B, L]（1=被掩码，0=可见）
    """
    # 以下是 dataset/dataloader.py:346-410 的完整代码，一字不改
    B, L, C = time_series.shape
    # ... (复制完整的 create_random_mask 函数体)
```

**Step 4: 迁移 AnomalyDataset → anomaly_dataset.py**

从 `dataset/dataloader.py:22-195` 提取 `AnomalyDataset` 类。

修改 import：
```python
# 旧
from dataset.pre_image import ts2image_1d, vico_render_timeseries
# 新
from src.datasets.pre_image import ts2image_1d, vico_render_timeseries
```

其余代码不变。

**Step 5: 迁移 collate_fn + DynamicLengthBatchSampler → collate.py**

从 `dataset/dataloader.py:198-576` 提取：
- `collate_fn` (line 198)
- `image_right_padding` (line 302)
- `DynamicLengthBatchSampler` (line 412)

修改 import：
```python
# 旧
from dataset.pre_image import ts2image_1d, vico_render_timeseries  # 如有
# 新
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_1d, vico_render_timeseries  # 如有
```

**Step 6: 更新 src/datasets/__init__.py**

```python
from src.datasets.anomaly_dataset import AnomalyDataset
from src.datasets.collate import collate_fn, DynamicLengthBatchSampler
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_1d, ts2image_Test, vico_render_timeseries
```

**Step 7: 写数据管线集成测试**

Create `tests/test_datasets.py`:

```python
import torch
import numpy as np


def test_create_random_mask_shape():
    from src.datasets.masking import create_random_mask

    B, L, C = 2, 128, 1
    ts = torch.randn(B, L, C)
    att_mask = torch.ones(B, L, dtype=torch.long)
    att_mask[0, 100:] = 0  # 模拟 padding

    masked_ts, mask = create_random_mask(ts, att_mask, patch_size=16, mask_ratio=0.3)
    assert masked_ts.shape == ts.shape, f"Expected {ts.shape}, got {masked_ts.shape}"
    assert mask.shape == (B, L), f"Expected {(B, L)}, got {mask.shape}"
    # padding 位置不应被掩码
    assert mask[0, 100:].sum() == 0, "Padding positions should not be masked"


def test_collate_fn_output_keys():
    from src.datasets.collate import collate_fn

    # 构造最小 batch（模拟 AnomalyDataset 的输出格式）
    batch = []
    for _ in range(2):
        ts = np.random.randn(200).astype(np.float32)
        normal_ts = ts.copy()
        img_vetime = np.random.randint(0, 255, (3, 64, 200), dtype=np.uint8)
        img_vico = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
        labels = np.zeros(200, dtype=np.float32)
        attribute = np.zeros(200, dtype=np.float32)
        period = 10
        padding_value = 0.0
        batch.append((ts, normal_ts, img_vetime, img_vico, labels, attribute, period, padding_value))

    result = collate_fn(batch, patch_size=16)
    expected_keys = {'time_series', 'normal_time_series', 'mask_time_series',
                     'image', 'image_vico', 'mask', 'labels', 'attention_mask',
                     'period', 'padding_value'}
    assert expected_keys.issubset(set(result.keys())), f"Missing keys: {expected_keys - set(result.keys())}"
```

**Step 8: 运行测试确认通过**

```bash
cd /mnt/sda/cjmProject/VETime && python -m pytest tests/test_datasets.py -v
```

Expected: PASS

**Step 9: 更新旧 dataset/dataloader.py 保持向后兼容**

在 `dataset/dataloader.py` 顶部添加：

```python
# 向后兼容：从新位置导入
from src.datasets.anomaly_dataset import AnomalyDataset
from src.datasets.collate import collate_fn, DynamicLengthBatchSampler, image_right_padding
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_1d, ts2image_Test, vico_render_timeseries
```

注释掉原有的类/函数定义，但保留文件存在。

**Step 10: 提交**

```bash
git add src/datasets/ dataset/dataloader.py tests/test_datasets.py
git commit -m "feat: migrate dataset pipeline to src/datasets/ with masking separation"
```

---

## Task 3: 迁移模型代码 → src/models/ + VETIME 统一接口

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/ts_encoder/__init__.py`
- Copy: `model/TS_encoder/config.py` → `src/models/ts_encoder/config.py`
- Copy: `model/TS_encoder/ts_model.py` → `src/models/ts_encoder/ts_model.py`
- Copy: `model/TS_encoder/ts_encoder.py` → `src/models/ts_encoder/ts_encoder.py`
- Copy: `model/TS_encoder/encoding_utils.py` → `src/models/ts_encoder/encoding_utils.py`
- Create: `src/models/vision_encoder/__init__.py`
- Copy: `model/Vision_encoder/V_encoder.py` → `src/models/vision_encoder/v_encoder.py`
- Copy: `model/Vision_encoder/models_mae.py` → `src/models/vision_encoder/models_mae.py`
- Copy: `model/Vision_encoder/Vit4AD.py` → `src/models/vision_encoder/vit4ad.py`
- Copy: `model/VTS_module.py` → `src/models/vts_module.py`
- Create: `src/models/vetime.py` (从 model/VETime.py 迁移，加统一接口)
- Test: `tests/test_models.py`

**Step 1: 创建目录结构并复制文件**

```bash
mkdir -p src/models/ts_encoder src/models/vision_encoder
touch src/models/__init__.py src/models/ts_encoder/__init__.py src/models/vision_encoder/__init__.py

# 复制 ts_encoder 文件（保持文件名不变以减少 diff）
cp model/TS_encoder/config.py src/models/ts_encoder/config.py
cp model/TS_encoder/ts_model.py src/models/ts_encoder/ts_model.py
cp model/TS_encoder/ts_encoder.py src/models/ts_encoder/ts_encoder.py
cp model/TS_encoder/encoding_utils.py src/models/ts_encoder/encoding_utils.py

# 复制 vision_encoder 文件
cp model/Vision_encoder/V_encoder.py src/models/vision_encoder/v_encoder.py
cp model/Vision_encoder/models_mae.py src/models/vision_encoder/models_mae.py
cp model/Vision_encoder/Vit4AD.py src/models/vision_encoder/vit4ad.py

# 复制 VTS_module
cp model/VTS_module.py src/models/vts_module.py
```

**Step 2: 修改所有内部 import 路径**

在 `src/models/` 下的所有文件中，将旧路径替换为新路径：

| 旧路径 | 新路径 |
|---|---|
| `from model.TS_encoder.ts_encoder import` | `from src.models.ts_encoder.ts_encoder import` |
| `from model.TS_encoder.encoding_utils import` | `from src.models.ts_encoder.encoding_utils import` |
| `from model.Vision_encoder.V_encoder import` | `from src.models.vision_encoder.v_encoder import` |
| `from model.Vision_encoder.models_mae import` | `from src.models.vision_encoder.models_mae import` |
| `from model.Vision_encoder.Vit4AD import` | `from src.models.vision_encoder.vit4ad import` |
| `from model.VTS_module import` | `from src.models.vts_module import` |
| `from loss.loss import` | `from src.losses.contrastive import win_Contrastive_Loss` |
| `from loss.loss import load_balance_loss` | `from src.losses.balance import load_balance_loss` |

**Step 3: 创建 src/losses/ 模块**

先创建损失函数模块（Task 3 的前置依赖）：

```bash
mkdir -p src/losses
touch src/losses/__init__.py
```

- `src/losses/contrastive.py`：从 `loss/loss.py` 复制 `win_ContrastiveLoss_init` 和 `win_Contrastive_Loss`
- `src/losses/balance.py`：从 `loss/loss.py` 复制 `load_balance_loss`
- `src/losses/anomaly.py`：从 `model/TS_encoder/ts_model.py:124-178` 提取 `anomaly_detection_loss` 作为独立函数
- `src/losses/reconstruction.py`：从 `model/TS_encoder/ts_model.py:74-122` 提取 `weighted_reconstruction_loss` 和 `masked_reconstruction_loss` 作为独立函数

注意：`ts_model.py` 上的方法暂时保留，内部改为调用 `src.losses` 中的函数，标注 `# Deprecated: use src.losses.xxx`。

**Step 4: 创建带统一接口的 src/models/vetime.py**

从 `model/VETime.py` 复制，修改 import 后添加 3 个新方法。

新增方法——`fold_images`：

```python
def fold_images(self, images, period, padding_value, **data_setting):
    """
    封装 vit_encoder.fold_image 调用。

    Args:
        images: VETime 时域图像 [B, C, H, W]
        period: 周期信息
        padding_value: 填充值
        **data_setting: 传给 fold_image 的额外参数（如 img_size, T_sqrt）

    Returns:
        images_folded: 折叠后的图像特征
        init_img_size: 原始图像尺寸
    """
    images_folded, init_img_size = self.vit_encoder.fold_image(
        images, period, padding_value, **data_setting
    )
    return images_folded, init_img_size
```

新增方法——`split_sequence`：

```python
def split_sequence(self, images, time_series, att_mask, labels):
    """
    封装 self.split_data 调用，用于长序列分块。

    Args:
        images: 折叠后的图像特征
        time_series: 时序数据 [B, L, C]
        att_mask: 注意力掩码 [B, L]
        labels: 标签 [B, L]

    Returns:
        list of (sub_images, sub_ts, sub_att_mask, sub_labels) chunks
    """
    return self.split_data(images, time_series, att_mask, labels)
```

新增方法——`compute_loss`：

```python
def compute_loss(self, outputs, time_series, att_mask, labels, stage,
                 alpha_recon=0.05, cl_weight=0.1, balance_weight=0.01,
                 focal_gamma=2.0, w_anomaly=1.2, w_normal=0.8):
    """
    统一损失计算入口。

    Args:
        outputs: self.forward() 的返回值 (local_embeddings1, m_w, loss_cl, local_embeddings2)
        time_series: 原始时序数据 [B, L, C]
        att_mask: 注意力掩码 [B, L]
        labels: 标签 [B, L]
        stage: 1=仅重构, 2=重构+分类
        alpha_recon: 重构损失缩放系数
        cl_weight: 对比损失权重
        balance_weight: 专家平衡损失权重
        focal_gamma: Focal Loss gamma
        w_anomaly: 异常点权重
        w_normal: 正常点权重

    Returns:
        dict: {
            'loss_total': Tensor,       # 总损失（可直接 backward）
            'loss_recon': float,        # 重构损失原始值
            'loss_anomaly': float,      # 分类损失原始值
            'loss_cl': float,           # 对比损失值
            'loss_balance': float,      # 平衡损失值
            'logits': Tensor,           # 分类 logits [B, L, 2]
            'reconstruction': Tensor,   # 重构输出
        }
    """
    from src.losses.balance import load_balance_loss

    local_embeddings1, m_w, loss_cl, local_embeddings2 = outputs
    device = local_embeddings1.device

    # 分类损失
    loss_anomaly, logits = self.anomaly_detection_loss(local_embeddings1, labels)
    if stage == 1:
        loss_anomaly = torch.tensor(0.0, device=device)

    # 重构损失
    loss_recon, rec = self.weighted_reconstruction_loss(
        local_embeddings2, time_series, att_mask, labels
    )

    # 专家平衡损失
    if stage == 1:
        loss_balance = balance_weight * load_balance_loss(m_w[0])
    else:
        loss_balance = balance_weight * 0.5 * (
            load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
        )

    # 对比损失
    loss_cl_val = cl_weight * loss_cl

    # 总损失
    loss_total = loss_anomaly + (alpha_recon * loss_recon) + loss_cl_val + loss_balance

    return {
        'loss_total': loss_total,
        'loss_recon': loss_recon.item(),
        'loss_anomaly': loss_anomaly.item() if isinstance(loss_anomaly, torch.Tensor) else 0.0,
        'loss_cl': loss_cl_val.item(),
        'loss_balance': loss_balance.item(),
        'logits': logits,
        'reconstruction': rec,
    }
```

**Step 5: 写模型验证测试**

Create `tests/test_models.py`:

```python
import torch
import pytest


def test_vetime_forward_output_shape():
    """验证 VETIME forward 输出形状正确。"""
    from src.models.ts_encoder.config import TimeSeriesConfig
    from src.models.ts_encoder.ts_model import TS_Model
    from src.models.vetime import VETIME

    config_t = TimeSeriesConfig(d_model=64, d_proj=32, patch_size=16,
                                 num_layers=1, num_heads=4, num_features=1)

    # 不加载视觉编码器权重，仅测试 forward 形状
    pytest.skip("需要预训练视觉编码器权重，标记为集成测试")


def test_vetime_compute_loss_returns_dict():
    """验证 compute_loss 返回正确格式的 dict。"""
    # 此测试用 mock 数据验证接口，不依赖实际模型权重
    pytest.skip("需要完整模型实例，标记为集成测试")


def test_vetime_fold_images_delegates():
    """验证 fold_images 正确代理到 vit_encoder.fold_image。"""
    pytest.skip("需要视觉编码器，标记为集成测试")
```

> **注意**：模型测试需要 GPU 和预训练权重，在 CI 中标记为 skip。实际验证在 Step 4 的端到端训练中进行。

**Step 6: 提交**

```bash
git add src/models/ src/losses/ tests/test_models.py
git commit -m "feat: migrate models to src/models/ with VETIME unified interface"
```

---

## Task 4: 配置管理 Hydra + OmegaConf

**Files:**
- Create: `configs/base.yaml`
- Create: `configs/univariate.yaml`
- Create: `configs/model/vetime.yaml`
- Create: `configs/model/vision.yaml`
- Modify: `requirements.txt` (添加 hydra-core)

**Step 1: 安装 hydra-core**

```bash
pip install hydra-core omegaconf
```

更新 `requirements.txt` 添加：
```
hydra-core>=1.3
omegaconf>=2.3
```

**Step 2: 创建 configs/base.yaml**

```yaml
# ==================== 基础配置 ====================
seed: 2024                          # 随机种子，保证实验可复现
device: cuda                        # 计算设备
mixed_precision: bf16               # 混合精度模式（bf16/fp16/no）

# ==================== 训练配置 ====================
training:
  stage1_epochs: 1                  # Stage 1 持续的 epoch 数（仅重构预训练）
  total_epochs: 30                  # 总 epoch 数
  cls_warmup_ratio: 0.5             # Stage 2 首个 epoch 的分类预热比例（0=跳过）

  optimizer:
    lr: 5e-4                        # 学习率
    weight_decay: 1e-5              # 权重衰减
  scheduler:
    type: cosine_with_warmup        # 学习率调度器类型
    warmup_ratio: 0.1              # 预热比例（占总训练步数的比例）
    eta_min: 1e-6                   # 余弦退火最小学习率
    start_factor: 0.1              # 预热起始因子
  early_stopping:
    patience: 4                     # 早停耐心值

# ==================== 损失配置 ====================
loss:
  alpha_recon: 0.05                 # 重构损失缩放系数，降低重构量级避免淹没分类梯度
  cl_weight: 0.1                    # 对比学习损失权重
  balance_weight: 0.01              # MoE 专家负载均衡损失权重
  focal_gamma: 2.0                  # Focal Loss gamma，增大对难样本的关注
  w_anomaly: 1.2                    # 异常点分类权重
  w_normal: 0.8                     # 正常点分类权重

# ==================== 数据配置 ====================
data:
  batch_size: 32                    # 批大小
  mask_ratio: 0.3                   # 自监督掩码比例
  gradient_accumulation_steps: 2    # 梯度累积步数
  max_seq_length: 5000              # 最大序列长度
  effective_batch_size: 256         # 动态批大小的有效批大小
  num_workers: 5                    # DataLoader 工作线程数
  val_ratio: 0.1                    # 验证集比例
  val_mode: tsb                     # 验证模式（tsb/split）
  min_seq_length: 100               # 最短序列长度过滤阈值

# ==================== 模型配置 ====================
model:
  model_name: VETime                # 模型名称
  vision_name: mae_nonparams        # 视觉编码器预训练权重名称
  ts_finetune_type: lora            # 时序编码器微调类型（lora/freeze）
  use_vectorized_fold: true         # 是否使用向量化 fold_image（约150倍加速）
  use_gradient_checkpointing: false  # 是否使用梯度检查点节省显存

# ==================== 路径配置 ====================
paths:
  dataset_path: ./dataset/univariate.pkl      # 训练数据集路径
  dataset_test_dir: ./dataset/TSB-AD          # 测试数据集目录
  ts_path: checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth  # TS Encoder 预训练权重
  save_dir: ./output                          # 输出目录
```

**Step 3: 创建 configs/model/vetime.yaml**

```yaml
# ==================== VETIME 时序编码器配置 ====================
d_model: 512                        # Transformer 隐藏维度
d_proj: 256                         # 投影维度
patch_size: 16                      # Patch 大小
num_layers: 8                       # Transformer 层数
num_heads: 8                        # 注意力头数
d_ff_dropout: 0.1                   # FFN dropout 率
max_total_tokens: 8192              # 最大 token 数
num_query_tokens: 1                 # 查询 token 数
use_rope: true                      # 是否使用旋转位置编码
activation: gelu                    # 激活函数
num_features: 1                     # 输入特征数
use_lora: true                      # 是否使用 LoRA 微调
lora_r: 8                           # LoRA 秩
lora_alpha: 16                      # LoRA alpha
```

**Step 4: 创建 configs/model/vision.yaml**

```yaml
# ==================== 视觉编码器配置 ====================
max_seq_length: 5000                # 最大序列长度
unpatch: true                       # 是否使用 unpatch 模式
finetune_type: none                 # 微调类型（none=完全冻结）
use_vectorized_fold: true           # 是否使用向量化 fold_image
img_size: 224                       # 输入图像尺寸
```

**Step 5: 写配置加载测试**

Append to `tests/test_utils.py`:

```python
from omegaconf import DictConfig, OmegaConf


def test_base_config_loads():
    cfg = OmegaConf.load("configs/base.yaml")
    assert cfg.seed == 2024
    assert cfg.training.stage1_epochs == 1
    assert cfg.loss.alpha_recon == 0.05


def test_vetime_config_loads():
    cfg = OmegaConf.load("configs/model/vetime.yaml")
    assert cfg.d_model == 512
    assert cfg.use_lora is True
```

**Step 6: 运行测试确认通过**

```bash
cd /mnt/sda/cjmProject/VETime && python -m pytest tests/test_utils.py::test_base_config_loads tests/test_utils.py::test_vetime_config_loads -v
```

Expected: PASS

**Step 7: 提交**

```bash
git add configs/ requirements.txt tests/test_utils.py
git commit -m "feat: add Hydra/OmegaConf configuration files"
```

---

## Task 5: 抽离训练引擎 → src/engines/trainer.py

**Files:**
- Create: `src/engines/__init__.py`
- Create: `src/engines/trainer.py`
- Create: `src/engines/hooks.py`
- Modify: `train.py` (改为薄壳入口)
- Test: `tests/test_trainer.py`

这是最大的任务。`train_univariate` (750 行) 将被拆分为：

| 组件 | 职责 | 原始行号 |
|---|---|---|
| `src/engines/hooks.py` | `freeze_for_cls_warmup` / `restore_requires_grad` | train.py:122-158 |
| `src/engines/trainer.py` | Trainer 类 | train.py:160-908 |
| `train.py` | 薄壳入口 | 新写 |

**Step 1: 创建目录结构**

```bash
mkdir -p src/engines
touch src/engines/__init__.py
```

**Step 2: 迁移 hooks.py**

Create `src/engines/hooks.py`，从 `train.py:122-158` 提取 `freeze_for_cls_warmup` 和 `restore_requires_grad`：

```python
"""训练钩子：分类预热冻结/解冻逻辑。"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 分类预热时可训练的参数前缀
CLS_WARMUP_TRAINABLE_PREFIXES = [
    "anomaly_head",
    "mm_w.task_proj.1.",   # task_id=1 的 task projection（分类）
    "mm_w.Router.",         # 路由器
]


def freeze_for_cls_warmup(model, accelerator):
    """
    分类预热冻结：仅保留分类相关参数可训练。

    Args:
        model: 模型实例
        accelerator: Accelerator 实例

    Returns:
        saved_requires_grad: dict，记录原始 requires_grad 状态，供恢复用
    """
    unwrapped = accelerator.unwrap_model(model)
    saved_requires_grad = {}

    for name, param in unwrapped.named_parameters():
        saved_requires_grad[name] = param.requires_grad
        should_train = any(prefix in name for prefix in CLS_WARMUP_TRAINABLE_PREFIXES)
        param.requires_grad = should_train

    trainable = sum(1 for p in unwrapped.parameters() if p.requires_grad)
    total = sum(1 for p in unwrapped.parameters())
    logger.info(f"[Cls Warmup] 可训练参数: {trainable}/{total}")

    return saved_requires_grad


def restore_requires_grad(model, accelerator, saved_requires_grad):
    """
    恢复分类预热前的 requires_grad 状态。

    Args:
        model: 模型实例
        accelerator: Accelerator 实例
        saved_requires_grad: freeze_for_cls_warmup 返回的 dict
    """
    unwrapped = accelerator.unwrap_model(model)
    for name, param in unwrapped.named_parameters():
        if name in saved_requires_grad:
            param.requires_grad = saved_requires_grad[name]
    logger.info("[Cls Warmup] 已恢复所有参数的 requires_grad 状态")
```

**Step 3: 创建 Trainer 类骨架**

Create `src/engines/trainer.py`。这是核心文件，将 `train_univariate` 的 750 行训练循环拆解为类方法。

```python
"""单变量训练引擎。"""

import os
import random
import gc
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from functools import partial

from accelerate import Accelerator

from src.utils.logger import get_logger
from src.utils.seed import seed_worker
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engines.hooks import freeze_for_cls_warmup, restore_requires_grad
from src.losses.balance import load_balance_loss
from src.datasets.collate import collate_fn
from src.models.vetime import VETIME

logger = get_logger(__name__)


class Trainer:
    """
    单变量训练引擎，承载两阶段课程训练循环。

    Stage 1 (epoch < stage1_epochs): 纯重构预训练，分类损失归零
    Stage 2 (epoch >= stage1_epochs): 多任务联合训练，含分类预热
    """

    def __init__(self, cfg, model, train_loader, val_loader, accelerator,
                 data_setting, vision_model=None):
        """
        Args:
            cfg: OmegaConf DictConfig，完整训练配置
            model: VETIME 模型实例
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
            accelerator: Accelerator 实例
            data_setting: dict，视觉编码器的数据设置（img_size, T_sqrt 等）
            vision_model: V_model 实例（用于 checkpoint 保存）
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.data_setting = data_setting
        self.vision_model = vision_model

        # 训练状态
        self.global_step = 0
        self.accumulated_samples = 0
        self.start_epoch = 0
        self.device = accelerator.device

    def setup(self):
        """构建 optimizer、scheduler、early_stopping。"""
        from Test_TSB import EarlyStopping

        unwrapped = self.accelerator.unwrap_model(self.model)
        trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in trainable_params)
        logger.info(f"可训练参数量: {param_count:,}")

        self.optimizer = AdamW(
            trainable_params,
            lr=self.cfg.training.optimizer.lr,
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )

        # Warmup + Cosine 调度器
        steps_per_epoch = len(self.train_loader)
        total_steps = (self.cfg.training.early_stopping.patience + 2) * steps_per_epoch
        warmup_steps = steps_per_epoch  # 1 epoch warmup

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.cfg.training.scheduler.start_factor,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.cfg.training.scheduler.eta_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        self.early_stopping = EarlyStopping(
            patience=self.cfg.training.early_stopping.patience,
            verbose=True,
            path=os.path.join(self.cfg.paths.save_dir, self.cfg.model.model_name, "best_model.pth"),
        )

        logger.info(f"Optimizer: AdamW(lr={self.cfg.training.optimizer.lr}, wd={self.cfg.training.optimizer.weight_decay})")
        logger.info(f"Scheduler: Warmup({warmup_steps} steps) + Cosine(T_max={total_steps - warmup_steps})")

    def train_epoch(self, epoch):
        """
        单个 epoch 训练。

        Args:
            epoch: 当前 epoch 编号（0-based）

        Returns:
            dict: epoch 级别的训练指标
        """
        is_stage_1 = epoch < self.cfg.training.stage1_epochs
        stage_name = "Stage 1" if is_stage_1 else "Stage 2"
        logger.info(f"[{stage_name}] Epoch {epoch+1}/{self.cfg.training.total_epochs}")

        # 分类预热检查
        is_cls_warmup = (
            (not is_stage_1)
            and (epoch == self.cfg.training.stage1_epochs)
            and (self.cfg.training.cls_warmup_ratio > 0)
        )
        cls_warmup_active = False
        saved_requires_grad = None
        cls_warmup_batches = 0

        if is_cls_warmup:
            cls_warmup_batches = max(1, int(len(self.train_loader) * self.cfg.training.cls_warmup_ratio))
            saved_requires_grad = freeze_for_cls_warmup(self.model, self.accelerator)
            cls_warmup_active = True
            logger.info(f"[Cls Warmup] 前 {cls_warmup_batches}/{len(self.train_loader)} 个 batch 仅训练分类网络")

        self.model.train()
        total_loss = 0
        total_loss_bce = 0
        total_loss_mse = 0
        total_loss_cl = 0
        total_loss_e = 0
        all_probs, all_preds, all_labels = [], [], []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}[Train]",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 分类预热解冻
            if cls_warmup_active and batch_idx == cls_warmup_batches:
                restore_requires_grad(self.model, self.accelerator, saved_requires_grad)
                self.optimizer.zero_grad()
                cls_warmup_active = False
                progress_bar.set_description(f"Epoch {epoch+1}[Train]")
                logger.info(f"[Cls Warmup] 分类预热完成 (batch {batch_idx}/{len(self.train_loader)})")

            # 解包 batch
            labels = batch["labels"]
            images = batch["image"]
            images_vico = batch.get("image_vico", None)
            time_series = batch["time_series"]
            att_mask = batch["attention_mask"]
            mask = batch["mask"]
            period = batch["period"]
            p_value = batch["padding_value"]

            stage = 1 if is_stage_1 else 2

            if labels.shape[1] > self.model.MAX_L:
                # 长序列分块
                batch_metrics = self._train_long_sequence(
                    images, images_vico, time_series, att_mask, labels,
                    period, p_value, stage
                )
            else:
                # 正常序列
                batch_metrics = self._train_single(
                    images, images_vico, time_series, att_mask, labels,
                    period, p_value, stage
                )

            # 反向传播
            self.accelerator.backward(batch_metrics["loss_total"])

            # 梯度累积与参数更新
            self._step_optimizer(labels.shape[0])

            # 记录指标
            total_loss += batch_metrics["loss_total"].item()
            total_loss_bce += batch_metrics["loss_anomaly"]
            total_loss_mse += batch_metrics["loss_recon"]
            total_loss_cl += batch_metrics["loss_cl"]
            total_loss_e += batch_metrics["loss_balance"]

            progress_bar.set_postfix({
                "Tot": f"{batch_metrics['loss_total'].item():.3f}",
                "BCE": f"{batch_metrics['loss_anomaly']:.3f}",
                "MSE": f"{batch_metrics['loss_recon']:.3f}",
            })

            # TensorBoard 日志
            if self.global_step > 0:
                unwrapped = self.accelerator.unwrap_model(self.model)
                self.accelerator.log({
                    "Loss/Total": batch_metrics["loss_total"].item(),
                    "Loss/BCE_Anomaly": batch_metrics["loss_anomaly"],
                    "Loss/MSE_Recon": batch_metrics["loss_recon"],
                    "Loss/CL_Contrastive": batch_metrics["loss_cl"],
                    "Loss/Balance": batch_metrics["loss_balance"],
                    "Train/LR": self.optimizer.param_groups[0]["lr"],
                    "Gate/alpha": torch.sigmoid(unwrapped.visual_cross_attn.alpha).item(),
                }, step=self.global_step)

            # 收集预测结果
            logits = batch_metrics.get("logits")
            if logits is not None:
                probs = torch.softmax(logits, dim=-1)[:, :, 1]
                preds = (probs > 0.5).float()
                probs, preds, labels_gathered = self.accelerator.gather_for_metrics(
                    (probs, preds, labels)
                )
                if self.global_step % 10 == 0:
                    for i in range(probs.shape[0]):
                        all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                        all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                        all_labels.append(labels_gathered[i].detach().cpu().numpy().reshape(-1))

            del batch_metrics

        # Epoch 级别日志
        n = len(self.train_loader)
        avg_loss = total_loss / n
        avg_bce = total_loss_bce / n
        avg_mse = total_loss_mse / n
        avg_cl = total_loss_cl / n
        avg_e = total_loss_e / n

        logger.info(
            f"Epoch {epoch+1} Avg Train Loss: {avg_loss:.4f} "
            f"(BCE: {avg_bce:.4f}, MSE: {avg_mse:.4f}, CL: {avg_cl:.4f}, Bal: {avg_e:.4f})"
        )

        return {
            "avg_loss": avg_loss,
            "avg_loss_bce": avg_bce,
            "avg_loss_mse": avg_mse,
            "avg_loss_cl": avg_cl,
            "avg_loss_e": avg_e,
        }

    def _train_single(self, images, images_vico, time_series, att_mask, labels,
                      period, p_value, stage):
        """处理正常长度序列的单 batch 训练。"""
        images_folded, init_img_size = self.model.fold_images(
            images, period, p_value, **self.data_setting
        )

        outputs = self.model(
            hidden_states=images_folded,
            hidden_states_vico=images_vico,
            time_series=time_series,
            att_mask=att_mask,
            init_img_size=init_img_size,
            labels=labels,
        )

        return self.model.compute_loss(
            outputs, time_series, att_mask, labels, stage,
            alpha_recon=self.cfg.loss.alpha_recon,
            cl_weight=self.cfg.loss.cl_weight,
            balance_weight=self.cfg.loss.balance_weight,
        )

    def _train_long_sequence(self, images, images_vico, time_series, att_mask, labels,
                             period, p_value, stage):
        """处理长序列分块训练。"""
        chunks = self.model.split_sequence(images, time_series, att_mask, labels)

        total_loss_total = 0
        total_loss_bce = 0
        total_loss_mse = 0
        total_loss_cl = 0
        total_loss_e = 0
        logits_list = []

        for img_part, ts_part, att_mask_part, label_part in chunks:
            images_folded, init_img_size = self.model.fold_images(
                img_part, period, p_value, **self.data_setting
            )

            outputs = self.model(
                hidden_states=images_folded,
                hidden_states_vico=images_vico,
                time_series=ts_part,
                att_mask=att_mask_part,
                init_img_size=init_img_size,
                labels=label_part,
            )

            metrics = self.model.compute_loss(
                outputs, ts_part, att_mask_part, label_part, stage,
                alpha_recon=self.cfg.loss.alpha_recon,
                cl_weight=self.cfg.loss.cl_weight,
                balance_weight=self.cfg.loss.balance_weight,
            )

            total_loss_total += metrics["loss_total"]
            total_loss_bce += metrics["loss_anomaly"]
            total_loss_mse += metrics["loss_recon"]
            total_loss_cl += metrics["loss_cl"]
            total_loss_e += metrics["loss_balance"]
            logits_list.append(metrics["logits"])

        n = len(chunks)
        if n > 0:
            total_loss_total = total_loss_total / n
            total_loss_bce /= n
            total_loss_mse /= n
            total_loss_cl /= n
            total_loss_e /= n

        logits = torch.cat(logits_list, dim=1) if logits_list else None

        return {
            "loss_total": total_loss_total,
            "loss_anomaly": total_loss_bce,
            "loss_recon": total_loss_mse,
            "loss_cl": total_loss_cl,
            "loss_balance": total_loss_e,
            "logits": logits,
        }

    def _step_optimizer(self, current_batch_size):
        """梯度累积与参数更新。"""
        if self.cfg.data.get("dynamic_batch", False):
            self.accumulated_samples += current_batch_size
            if self.accumulated_samples >= self.cfg.data.effective_batch_size:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self.accumulated_samples = 0
        else:
            self.global_step += 1
            if self.global_step % self.accelerator.gradient_accumulation_steps == 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    def validate(self, epoch):
        """
        验证循环。

        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                labels = batch["labels"]
                images = batch["image"]
                images_vico = batch.get("image_vico", None)
                time_series = batch["time_series"]
                att_mask = batch["attention_mask"]
                period = batch["period"]
                p_value = batch["padding_value"]

                images_folded, init_img_size = self.model.fold_images(
                    images, period, p_value, **self.data_setting
                )

                outputs = self.model(
                    hidden_states=images_folded,
                    hidden_states_vico=images_vico,
                    time_series=time_series,
                    att_mask=att_mask,
                    init_img_size=init_img_size,
                    labels=labels,
                )

                metrics = self.model.compute_loss(
                    outputs, time_series, att_mask, labels, stage=2,
                    alpha_recon=self.cfg.loss.alpha_recon,
                    cl_weight=self.cfg.loss.cl_weight,
                    balance_weight=self.cfg.loss.balance_weight,
                )
                total_val_loss += metrics["loss_total"].item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} Avg Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def run(self):
        """
        完整训练流程。

        遍历所有 epoch，自动切换 stage，早停时退出。
        """
        self.setup()

        # 准备模型和数据
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )

        for epoch in range(self.start_epoch, self.cfg.training.total_epochs):
            train_metrics = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            if self.early_stopping(val_loss, self.accelerator.unwrap_model(self.model)):
                logger.info(f"早停触发，最佳验证损失: {self.early_stopping.val_loss_min:.6f}")
                break

        logger.info("训练完成")
```

**Step 4: 写 Trainer 单元测试**

Create `tests/test_trainer.py`:

```python
import pytest


def test_trainer_init():
    """Trainer 可以被实例化（需要完整依赖，标记为集成测试）。"""
    pytest.skip("需要 GPU 和预训练权重，标记为集成测试")
```

**Step 5: 修改 train.py 为薄壳入口**

在 train.py 中添加新入口函数（保留旧 `train_univariate` 不删）：

```python
def train_univariate_new(cfg):
    """新入口：基于 src/ 的训练流程。"""
    from src.utils.seed import seed_everything
    from src.utils.logger import get_logger
    from src.engines.trainer import Trainer
    from src.models.vision_encoder.v_encoder import V_model
    from src.models.ts_encoder.config import TimeSeriesConfig
    from src.models.ts_encoder.ts_model import TS_Model
    from src.models.vetime import VETIME
    from src.datasets.anomaly_dataset import AnomalyDataset
    from src.datasets.collate import collate_fn, DynamicLengthBatchSampler
    from omegaconf import OmegaConf

    log = get_logger(__name__)
    seed_everything(cfg.seed)

    # Accelerator
    gradient_accumulation_steps = max(1, cfg.data.effective_batch_size // cfg.data.batch_size)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir="./output/logs",
    )

    # Vision Encoder
    log.info(f"加载 Vision Encoder: {cfg.model.vision_name}")
    vision_model = V_model(
        cfg.model.vision_name,
        MAX_L=cfg.data.max_seq_length,
        unpatch=True,
        finetune_type='none',
        use_vectorized_fold=cfg.model.use_vectorized_fold,
    )
    config_v = vision_model.config

    # TS Encoder
    config_t = TimeSeriesConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    if cfg.model.ts_finetune_type == 'lora':
        config_t.use_lora = True
    else:
        config_t.use_lora = False

    ts_model = TS_Model(config_t)
    if cfg.paths.ts_path:
        state = torch.load(cfg.paths.ts_path, map_location='cpu')['model_state_dict']
        if cfg.model.ts_finetune_type == 'lora':
            new_state = {}
            for key, value in state.items():
                if any(x in key for x in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                                           'out_proj.weight', 'gate_proj.weight', 'gate_proj.bias',
                                           'up_proj.weight', 'up_proj.bias', 'down_proj.weight',
                                           'down_proj.bias']):
                    parts = key.rsplit('.', 1)
                    new_key = f"{parts[0]}.original_linear.{parts[1]}"
                    new_state[new_key] = value
                else:
                    new_state[key] = value
            ts_model.load_state_dict(new_state, strict=False)
        else:
            ts_model.load_state_dict(state, strict=False)

    # VETIME Model
    model = VETIME(config_v, vision_model, config_t, ts_model, cfg.model.model_name)

    # Data
    patch_size = config_v['patch_size'] if isinstance(config_v, dict) else config_v.patch_size
    data_setting = {"img_size": 224, "T_sqrt": False}

    # ... DataLoader 构建逻辑（从旧 train_univariate 对应段迁移）

    # Trainer
    trainer = Trainer(cfg, model, train_loader, val_loader, accelerator,
                      data_setting, vision_model)
    trainer.run()
```

**Step 6: 提交**

```bash
git add src/engines/ train.py tests/test_trainer.py
git commit -m "feat: extract training engine to src/engines/trainer.py"
```

---

## Task 6: 抽离评估引擎 → src/engines/evaluator.py + evaluate.py

**Files:**
- Create: `src/engines/evaluator.py`
- Create: `evaluate.py`
- Modify: `Test_TSB.py` (保留但添加兼容注释)

**Step 1: 创建 evaluator.py**

从 `Test_TSB.py:189-540` 的 `TSB_test` 函数提取为 `Evaluator` 类：

```python
"""TSB-AD 基准测试引擎。"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_Test
from evaluation.metrics import get_metrics

logger = get_logger(__name__)


class Evaluator:
    """TSB-AD 基准测试引擎。"""

    def __init__(self, cfg, model, accelerator=None):
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator
        self.device = next(model.parameters()).device

    def evaluate_dataset(self, dataset_name, test_data, test_labels,
                         data_setting, patch_size):
        """
        单个数据集推理 + 异常分数计算。

        Args:
            dataset_name: 数据集名称
            test_data: 测试时序数据
            test_labels: 测试标签
            data_setting: 视觉编码器数据设置
            patch_size: patch 大小

        Returns:
            dict: 包含异常分数和评估指标
        """
        self.model.eval()
        # ... 从 TSB_test 中提取单数据集推理逻辑
        # 关键改动：使用 model.fold_images() 和 model.compute_loss() 而非直接访问模型内部

    def evaluate_benchmark(self, dataset_dir, dataset_list, data_setting, patch_size):
        """
        遍历 TSB-AD 所有数据集，汇总结果。

        Returns:
            float: 1 - avg_f1（越小越好）
        """
        results = []
        for ds_name in tqdm(dataset_list, desc="Evaluating"):
            # 加载数据集
            # 调用 evaluate_dataset
            # 收集结果
            pass

        avg_f1 = np.mean([r['F1'] for r in results])
        return 1.0 - avg_f1

    def compute_metrics(self, score, labels, sliding_window=100):
        """调用 evaluation.metrics 计算指标。"""
        return get_metrics(score, labels, slidingWindow=sliding_window)
```

> **注意**：Evaluator 的完整实现需要仔细对照 `Test_TSB.py:189-540` 逐行迁移，确保行为一致。特别要注意 `dataloader_TSB` 中的标准化逻辑（`std + 1e-2`）需要保持与原代码一致。

**Step 2: 创建 evaluate.py 薄壳入口**

```python
"""统一评估/推理启动入口。"""

import hydra
from omegaconf import DictConfig
from src.utils.seed import seed_everything
from src.engines.evaluator import Evaluator


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    # 构建模型、加载权重、创建 Evaluator、运行评估
    ...


if __name__ == "__main__":
    main()
```

**Step 3: 提交**

```bash
git add src/engines/evaluator.py evaluate.py
git commit -m "feat: extract evaluation engine to src/engines/evaluator.py"
```

---

## Task 7: 端到端集成验证

**Files:**
- Modify: `train.py` (最终版薄壳)
- Modify: `evaluate.py` (最终版薄壳)

**Step 1: 用新架构跑 1 epoch 训练**

```bash
cd /mnt/sda/cjmProject/VETime
python train.py training.total_epochs=1 training.stage1_epochs=1
```

**验证项**：
- 训练正常启动，无 import 错误
- Loss 数值与旧 `train_univariate` 在相同数据上对齐
- TensorBoard 日志正常写入
- Checkpoint 正常保存

**Step 2: 用新架构跑评估**

```bash
python evaluate.py paths.ts_path=checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth
```

**验证项**：
- 指标数值与旧 `Test_TSB.py` 对齐
- 无 import 错误

**Step 3: 对比新旧结果**

如果数值差异 > 1%，检查差异来源（通常由随机种子、数据加载顺序、浮点精度引起）。

**Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: complete project refactoring - data/model/training/evaluation decoupled"
```

---

## 执行顺序与依赖图

```
Task 1 (src/utils/)
  ↓
Task 2 (src/datasets/)
  ↓
Task 3 (src/models/ + src/losses/)
  ↓
Task 4 (configs/ Hydra)
  ↓
Task 5 (src/engines/trainer.py)
  ↓
Task 6 (src/engines/evaluator.py)
  ↓
Task 7 (端到端集成验证)
```

每个 Task 完成后独立提交，确保可回滚。
