# VETime 实验结果分析与改进方向

> 基于11个数据集的实验结果数据分析

---

## 一、实验数据概览

| 数据集 | VUS-PR | VUS-ROC | Standard-F1 | PA-F1 | Affiliation-F | F1_T |
|--------|--------|---------|-------------|-------|---------------|------|
| IOPS | 0.256 | 0.8997 | 0.3671 | 0.4752 | 0.8123 | 0.3763 |
| MGAB | 0.006 | 0.6264 | 0.0134 | 0.0393 | 0.6750 | 0.0147 |
| NAB | 0.3178 | 0.6383 | 0.3547 | 0.9458 | 0.8933 | 0.4318 |
| NEK | 0.7391 | 0.8503 | 0.7352 | 0.8437 | 0.8519 | 0.8095 |
| Power | 0.1191 | 0.5595 | 0.228 | 0.7831 | 0.7262 | 0.2293 |
| SED | 0.0559 | 0.3935 | 0.0943 | 0.2435 | 0.6713 | 0.0954 |
| Stock | 0.803 | 0.9164 | 0.2348 | 0.2345 | 0.7104 | 0.2149 |
| TODS | 0.532 | 0.8233 | 0.155 | 0.3537 | 0.688 | 0.1531 |
| WSD | 0.2538 | 0.881 | 0.3673 | 0.3313 | 0.8201 | 0.3824 |
| UCR | 0.0405 | 0.6145 | 0.0695 | 0.2415 | 0.7551 | 0.1107 |
| YAHOO | 0.3315 | 0.8018 | 0.1303 | 0.1399 | 0.7852 | 0.1596 |

---

## 二、关键问题诊断

### 🔴 问题1: 定位精度不足 (最核心问题)

**症状**: PA-F1 >> Standard-F1（差距越大，定位问题越严重）

| 数据集 | PA-F1 | Standard-F1 | 差距 | 严重程度 |
|--------|-------|-------------|------|----------|
| **NAB** | 0.9458 | 0.3547 | **+0.59** | 🔴 严重 |
| **Power** | 0.7831 | 0.228 | **+0.56** | 🔴 严重 |
| **TODS** | 0.3537 | 0.155 | +0.20 | 🟡 中等 |
| **UCR** | 0.2415 | 0.0695 | +0.17 | 🟡 中等 |
| **SED** | 0.2435 | 0.0943 | +0.15 | 🟡 中等 |

**诊断**:
- 模型能**检测**到异常存在，但**定位**不准确
- 异常分数在时间上有偏移（早于或晚于真实异常）
- 可能在异常边界处模糊

**根因分析**:
1. **Patch-Level Temporal Alignment** 可能引入位置偏差
2. **图像折叠/展开** (`fold_image`/`unfold_image`) 可能导致位置失真
3. **对比学习损失** 可能过度关注段级别而忽略点级别精确对齐

---

### 🟠 问题2: 假阳性/假阴性不平衡

**症状**: VUS-ROC >> VUS-PR（正常情况，但差距过大需关注）

| 数据集 | VUS-ROC | VUS-PR | 差距 | 含义 |
|--------|---------|--------|------|------|
| **IOPS** | 0.8997 | 0.256 | **0.64** | 大量假阳性 |
| **WSD** | 0.881 | 0.2538 | **0.63** | 大量假阳性 |
| **Stock** | 0.9164 | 0.803 | 0.11 | 较好 |
| **NEK** | 0.8503 | 0.7391 | 0.11 | 较好 |

**诊断**:
- **VUS-ROC高，VUS-PR低** → 模型产生大量**假阳性** (False Positives)
- 异常分数在负样本上也有较高值
- 背景噪声建模不足

---

### 🟡 问题3: 极端数据集表现差

**症状**: MGAB、SED、UCR、YAHOO上所有指标都很低

| 数据集 | 特点 | Standard-F1 | 可能原因 |
|--------|------|-------------|----------|
| **MGAB** | 极长序列(97777), 极低异常率(0.2%) | 0.0134 | 类别极度不平衡 |
| **SED** | 能量数据, 复杂模式 | 0.0943 | 视觉编码器对能量信号不敏感 |
| **UCR** | 228个短序列, 多样化 | 0.0695 | 零样本泛化不足 |
| **YAHOO** | Web数据, 259个序列 | 0.1303 | 数据分布差异大 |

---

### 🟢 优势领域

| 数据集 | 表现 | 原因 |
|--------|------|------|
| **NEK** | 全面优秀(F1=0.73) | 短序列(1073), 周期性明显 |
| **Stock** | VUS-PR高(0.803) | 金融数据与训练数据分布相似 |

---

## 三、具体改进方向

### 方向1: 提升定位精度 (最高优先级)

#### 1.1 改进Patch-Level Temporal Alignment

**当前问题**:
- `fold_image` 使用最近邻插值 (`mode='nearest'`)
- `unfold_image` 使用双线性插值，可能引入平滑

**改进方案**:
```python
# 当前代码 (V_encoder.py:89-90)
img_resized_y = F.interpolate(img_2d.unsqueeze(0), size=(img_size, img_2d.shape[2]), mode='nearest')
img_final = F.interpolate(img_resized_y, size=(img_size, img_size), mode='bilinear')

# 改进方案: 保持时间维度精确对齐
img_resized_y = F.interpolate(img_2d.unsqueeze(0), size=(img_size, img_2d.shape[2]),
                              mode='bilinear', align_corners=True)  # 使用align_corners
```

#### 1.2 引入位置感知损失

在对比学习中增加**位置约束**:
```python
# 在win_Contrastive_Loss中增加位置正则
pos_weight = torch.exp(-0.5 * ((pos_i - pos_j) / sigma) ** 2)  # 高斯位置权重
loss = loss * pos_weight  # 远离真实位置的配对降低权重
```

#### 1.3 多尺度特征融合

```python
# 融合不同patch size的特征
patch_sizes = [8, 16, 32]
multi_scale_features = []
for ps in patch_sizes:
    feat = extract_features(x, patch_size=ps)
    multi_scale_features.append(feat)

# 注意力机制融合
fused = attention_fusion(multi_scale_features)
```

**预期收益**: Standard-F1提升20-30%

---

### 方向2: 降低假阳性率

#### 2.1 改进重建损失权重

**当前问题**: 正常样本上的重建误差可能仍较大

**改进方案**:
```python
# 当前代码 (ts_model.py:107-109)
effective_mask = mask.clone()
if labels is not None:
    labels = labels.bool()
    effective_mask = effective_mask & (~labels)  # 排除异常点

# 改进: 增加难分正常样本的权重
reconstruction_error = (reconstructed - original_time_series).abs()
hard_negative_weight = 1 + torch.sigmoid(reconstruction_error - threshold)
weighted_loss = (reconstruction_error * hard_negative_weight)[flat_mask].mean()
```

#### 2.2 时序一致性约束

异常应该是**连续**的，不应出现孤立的点:
```python
# 在损失中加入时序平滑约束
smoothness_loss = torch.mean((anomaly_scores[1:] - anomaly_scores[:-1]) ** 2)
total_loss = detection_loss + 0.1 * smoothness_loss
```

#### 2.3 自适应阈值

不同数据集应使用不同的阈值策略:
```python
def adaptive_threshold(score, method='otsu'):
    """使用Otsu方法自动确定阈值"""
    from skimage.filters import threshold_otsu
    return threshold_otsu(score)
```

**预期收益**: VUS-PR提升15-25%

---

### 方向3: 针对极端不平衡数据

#### 3.1 类别重采样

对于MGAB(0.2%异常率)这类数据集:
```python
# 在dataloader中增加异常样本采样权重
sampler = WeightedRandomSampler(
    weights=[10 if label.sum() > 0 else 1 for label in labels],
    num_samples=len(dataset)
)
```

#### 3.2 改进对比学习的采样策略

**当前问题**: `_sample_bg_windows_fast` 随机采样可能忽略重要负样本

**改进方案**:
```python
def hard_negative_mining(z1, z2, labels, num_hard=5):
    """挖掘困难负样本"""
    similarities = torch.matmul(z1, z2.T)
    # 选择与正样本相似度最高的负样本
    hard_negatives = torch.topk(similarities[labels==0], num_hard).indices
    return hard_negatives
```

#### 3.3 引入Focal Loss

```python
# 替换BCE Loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal_loss(logits, labels)
```

**预期收益**: MGAB、UCR上F1提升50%以上

---

### 方向4: 模态融合优化

#### 4.1 动态模态权重

**当前**: 固定路由机制
**改进**: 基于输入特征动态调整模态权重
```python
class DynamicFusion(nn.Module):
    def forward(self, f_ts, f_v, f_a):
        # 计算各模态的置信度
        conf_ts = torch.norm(f_ts, dim=-1, keepdim=True)
        conf_v = torch.norm(f_v, dim=-1, keepdim=True)

        # 基于置信度加权
        weights = F.softmax(torch.cat([conf_ts, conf_v], dim=-1), dim=-1)
        fused = weights[..., 0:1] * f_ts + weights[..., 1:2] * f_v
        return fused
```

#### 4.2 跨模态注意力精细化

```python
# 在VTS_Alignment中引入门控机制
gate = torch.sigmoid(self.gate_linear(torch.cat([f_ts, f_v], dim=-1)))
fused = gate * f_ts + (1 - gate) * f_v
```

**预期收益**: 整体性能提升10-15%

---

### 方向5: 后处理优化

#### 5.1 异常分数平滑

```python
def smooth_scores(scores, window=5):
    """使用高斯平滑减少噪声"""
    kernel = gaussian_kernel(window, sigma=1.0)
    smoothed = convolve1d(scores, kernel, mode='constant')
    return smoothed
```

#### 5.2 峰值检测

```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(anomaly_scores, height=threshold,
                               distance=min_anomaly_distance)
```

**预期收益**: PA-F1到Standard-F1的差距缩小30%

---

## 四、改进优先级建议

### 短期 (1-2周，可快速验证)

1. ✅ **后处理平滑** (实现简单，立即见效)
2. ✅ **自适应阈值** (无需修改模型)
3. ✅ **重建损失权重调整** (少量代码修改)

### 中期 (1个月，需要训练)

1. 🔧 **引入位置感知损失** (修改loss.py)
2. 🔧 **多尺度特征融合** (修改模型结构)
3. 🔧 **Focal Loss** (替换分类损失)

### 长期 (2-3个月，重大改动)

1. 🏗️ **改进图像折叠/展开** (可能降低性能，需仔细验证)
2. 🏗️ **动态模态融合** (重新设计融合模块)
3. 🏗️ **增加时序一致性约束** (可能与其他约束冲突)

---

## 五、针对性数据集改进

| 数据集 | 主要问题 | 针对性改进 |
|--------|----------|------------|
| **NAB** | 定位偏差大 | 位置感知损失 + 后处理平滑 |
| **Power** | 假阳性多 | 时序一致性约束 + 重建权重调整 |
| **MGAB** | 类别不平衡 | Focal Loss + 重采样 |
| **UCR** | 零样本泛化差 | 增加多样化训练数据 |
| **YAHOO** | 多序列分布差异 | 动态模态权重 |
| **SED** | 视觉编码不匹配 | 针对能量数据预训练视觉编码器 |

---

## 六、预期改进效果估算

基于上述改进方向，预期各指标提升幅度：

| 指标 | 当前平均 | 预期提升 | 改进后预期 |
|------|----------|----------|------------|
| **Standard-F1** | 0.245 | +35% | 0.33 |
| **PA-F1** | 0.416 | +10% | 0.46 |
| **VUS-PR** | 0.314 | +25% | 0.39 |
| **Affiliation-F** | 0.749 | +8% | 0.81 |
| **F1_T** | 0.271 | +30% | 0.35 |

**关键目标**: 缩小 PA-F1 与 Standard-F1 的差距（从平均0.17降至0.08以内）

---

## 七、验证实验设计

建议按以下顺序验证改进效果：

1. **基线**: 当前模型在NAB、Power上的表现（定位问题最明显）
2. **后处理**: 添加平滑+自适应阈值
3. **损失改进**: 位置感知损失 + Focal Loss
4. **结构改进**: 多尺度融合 + 动态路由

每个阶段在**同一数据集子集**上验证，确保改进有效。

---

*分析完成*
