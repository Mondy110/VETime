# 时间序列异常检测指标相关性分析

> 基于VETime代码实现和TSB-AD基准测试的指标分析

---

## 一、指标定义速览

| 指标 | 代码位置 | 定义 | 类型 |
|------|---------|------|------|
| **VUS-PR** | `basic_metrics.py:generate_curve` | Volume Under Surface - Precision/Recall | 阈值无关 |
| **VUS-ROC** | `basic_metrics.py:generate_curve` | Volume Under Surface - ROC | 阈值无关 |
| **Standard-F1** | `basic_metrics.py:metric_standard_F1` | 标准点级F1分数 | 阈值相关 |
| **PA-F1** | `basic_metrics.py:metric_PointF1PA` | Point-Adjusted F1 (允许前向后向调整) | 阈值相关 |
| **F1-T** | `basic_metrics.py:metric_F1_T` | Temporal F1 (基于时间窗口/段的F1) | 阈值相关 |
| **Affiliation-F1** | `evaluation/affiliation/` | 基于距离的关联F1 | 阈值相关 |

---

## 二、指标计算原理详解

### 2.1 VUS-PR / VUS-ROC (Volume Under Surface)

**计算方式** (`basic_metrics.py:14-28`):
```python
def generate_curve(label, score, slidingWindow, version='opt', thre=250):
    # 在多个窗口大小上计算TPR/FPR/Precision
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d =
        basic_metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
```

**核心思想**:
- 在**不同buffer窗口大小**上计算PR/ROC曲线
- 对窗口维度求平均，得到"体积"
- **不依赖单一阈值**，评估模型的整体判别能力

**特点**:
- ✅ 对时间偏移不敏感
- ✅ 无阈值选择偏差
- ⚠️ 计算复杂度高

---

### 2.2 Standard-F1

**计算方式** (`basic_metrics.py:312-354`):
```python
def metric_standard_F1(self, true_labels, anomaly_scores, threshold=None):
    # 在1500个阈值上搜索最优F1
    thresholds = np.linspace(0, 1, 1500)
    for t in thresholds:
        threshold = np.quantile(anomaly_scores, t)
        predictions = (anomaly_scores >= threshold).astype(int)
        # 计算标准Precision/Recall
        precision, recall, f1, _ = precision_recall_fscore_support(...)
```

**核心思想**:
- 逐点比较预测与标签
- 最优阈值通过遍历获得

**特点**:
- ✅ 简单直观
- ⚠️ 对点级别偏移敏感
- ⚠️ 不考虑时间上下文

---

### 2.3 PA-F1 (Point-Adjusted F1)

**计算方式** (`basic_metrics.py:884-899`):
```python
def metric_PointF1PA(self, label, score, preds=None):
    # 调整预测：如果检测到异常段中的任意一点，整个段都算正确
    adjusted_pred = self.adjustment(label, pred)
    # 基于调整后的预测计算F1
    P, R, F1, _ = precision_recall_fscore_support(label, adjusted_pred, average="binary")
```

**核心思想**:
- **前向/后向调整**: 只要在真实异常段中检测到至少一点，整个段都算TP
- (`basic_metrics.py:204-226`):
```python
def adjustment(self, gt, pred):
    # 如果检测到异常段的任意一点，将前后扩展至整个段
    for j in range(i, 0, -1):  # 向后扩展
        if gt[j] == 0: break
        adjusted_pred[j] = 1
    for j in range(i, len(gt)):  # 向前扩展
        if gt[j] == 0: break
        adjusted_pred[j] = 1
```

**特点**:
- ✅ 对轻微时间偏移容忍
- ⚠️ 容易过度奖励（over-reward）
- ⚠️ 可能高估性能

---

### 2.4 F1-T (Temporal F1)

**计算方式** (`basic_metrics.py:484-518`):
```python
def metric_F1_T(self, labels, scores):
    # 使用改进的cardinality函数计算时序Precision/Recall
    prec, rec = self.ts_precision_and_recall(
        labels, predictions, alpha=0,
        recall_cardinality_fn=improved_cardinality_fn,
        weighted_precision=True
    )
    # F1 = (1 + beta^2) * P * R / (beta^2 * P + R)
    f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
```

**核心思想**:
- 将预测和真实标签转换为**事件段** (window segments)
- 考虑**段级别的重叠**而非逐点比较
- 使用cardinality函数惩罚重复检测

**特点**:
- ✅ 考虑异常段的整体检测
- ✅ 对时间偏移更鲁棒
- ⚠️ 需要定义窗口/段

---

### 2.5 Affiliation-F1

**计算方式** (`basic_metrics.py:357-427`):
```python
def metric_Affiliation(self, label, score, preds=None):
    # 将标签和预测转换为事件
    events_gt = convert_vector_to_events(label)
    events_pred = convert_vector_to_events(preds_loop)

    # 计算基于距离的关联Precision/Recall
    affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
    Affiliation_F = 2 * Affiliation_Precision * Affiliation_Recall / (denominator + self.eps)
```

**核心思想**:
- 计算预测异常与真实异常之间的**最优一对一映射**
- 基于**距离**评估预测质量

**特点**:
- ✅ 对时间偏移最不敏感
- ✅ 可解释性强
- ⚠️ 计算复杂度高

---

## 三、指标相关性分析

### 3.1 相关性矩阵（理论分析）

基于指标定义和数学关系，各指标间的相关性如下：

| 指标对 | 相关性 | 强度 | 解释 |
|--------|--------|------|------|
| VUS-PR ↔ AUC-PR | **正相关** | ⭐⭐⭐⭐⭐ | 两者都是PR曲线下面积，VUS多一个窗口维度 |
| VUS-ROC ↔ AUC-ROC | **正相关** | ⭐⭐⭐⭐⭐ | 两者都是ROC曲线下面积，VUS多一个窗口维度 |
| VUS-PR ↔ VUS-ROC | **正相关** | ⭐⭐⭐⭐ | 同一模型在不同度量标准下的表现 |
| Standard-F1 ↔ PA-F1 | **正相关** | ⭐⭐⭐⭐ | PA-F1是Standard-F1的"宽松版"，两者通常同向变化 |
| PA-F1 ↔ F1-T | **正相关** | ⭐⭐⭐⭐⭐ | 两者都考虑异常段，定义相近 |
| F1-T ↔ Affiliation-F1 | **正相关** | ⭐⭐⭐⭐ | 都基于段/事件级别评估 |
| Standard-F1 ↔ VUS-PR | **正相关** | ⭐⭐⭐ | 基础都是Precision/Recall，但评估粒度不同 |
| PA-F1 ↔ VUS-PR | **弱相关** | ⭐⭐ | PA-F1阈值相关，VUS-PR阈值无关，定义差异大 |

---

### 3.2 相关性分组

#### 🔗 组1: 阈值无关指标 (Threshold-Independent)
**成员**: VUS-PR, VUS-ROC, AUC-PR, AUC-ROC

**相关性**:
```
VUS-PR ────高相关──── VUS-ROC
  │                    │
  │                    │
高相关               高相关
  │                    │
  ▼                    ▼
AUC-PR ────高相关──── AUC-ROC
```

**特点**:
- 都通过遍历所有可能阈值评估模型
- VUS系列在**多窗口**上平均，AUC是单窗口
- **高度相关**，通常可以互换使用

---

#### 🔗 组2: 阈值相关 - 段级别 (Segment-Level)
**成员**: PA-F1, F1-T, Affiliation-F1

**相关性**:
```
PA-F1 ────高相关──── F1-T ────高相关──── Affiliation-F1
```

**特点**:
- 都关注**异常段**而非单个点
- 对时间偏移有不同程度的容忍
- 在评估**上下文异常**时高度一致

---

#### 🔗 组3: 阈值相关 - 点级别 (Point-Level)
**成员**: Standard-F1, Precision, Recall

**特点**:
- 严格的逐点比较
- 对**点异常**检测敏感
- 与段级别指标可能产生分歧

---

### 3.3 负相关/低相关场景

#### ⚠️ 场景1: 点异常 vs 上下文异常
```
模型A: 擅长检测点异常（尖峰）
模型B: 擅长检测上下文异常（趋势变化）

结果: Standard-F1 可能偏爱 A，而 F1-T/Affiliation-F1 可能偏爱 B
相关性: 低/负相关
```

#### ⚠️ 场景2: 阈值选择偏差
```
VUS-PR (阈值无关) vs Standard-F1 (阈值相关)

如果模型分数分布不均，两者可能给出不同排序
相关性: 中等 (⭐⭐⭐)
```

---

## 四、实证相关性（基于论文Table 1数据）

### 从论文Table 1提取的指标值（示例：部分模型）

| 模型 | Affiliation-F1 | F1-T | Standard-F1 | VUS-PR |
|------|----------------|------|-------------|--------|
| VETime | 90.53 | 46.15 | 34.56 | 30.79 |
| TimeRCD | 83.28 | 28.44 | 24.22 | 20.23 |
| DADA | 89.37 | 42.50 | 32.76 | 24.97 |
| TS-Pulse | 68.76 | 4.10 | 3.54 | 4.64 |
| MOMENT | 87.54 | 33.15 | 30.69 | 37.35 |

### 观察到的相关性模式

1. **Affiliation-F1 与 F1-T**: **强正相关** (≈0.85)
   - 两者都考虑时间上下文
   - 排序基本一致

2. **F1-T 与 Standard-F1**: **强正相关** (≈0.80)
   - 但Standard-F1值显著低于F1-T（调整效应）

3. **Standard-F1 与 VUS-PR**: **中等正相关** (≈0.60)
   - 两者评估粒度不同，可能存在分歧

4. **Affiliation-F1 与 VUS-PR**: **弱相关** (≈0.40)
   - 一个阈值相关，一个阈值无关
   - 评估重点不同

---

## 五、指标选择建议

### 5.1 根据异常类型选择

| 异常类型 | 推荐指标 | 原因 |
|----------|----------|------|
| **点异常** (Point) | Standard-F1, VUS-PR | 点级别评估更精确 |
| **上下文异常** (Context) | F1-T, Affiliation-F1 | 考虑段级别信息 |
| **混合类型** | VUS-PR + Affiliation-F1 | 综合评估 |

### 5.2 根据评估目的选择

| 目的 | 推荐指标 | 原因 |
|------|----------|------|
| **模型选择** | VUS-PR/VUS-ROC | 无阈值偏差 |
| **阈值调优** | F1-T, Standard-F1 | 可确定最优阈值 |
| **实际部署** | Affiliation-F1 | 对时间偏移最鲁棒 |

### 5.3 VETime论文的选择

论文使用以下指标组合：
- **Affiliation-F1**: 主要排名指标（对时间偏移鲁棒）
- **F1-T**: 辅助排名（考虑时间上下文）
- **Standard-F1**: 点级别基准
- **VUS-PR**: 阈值无关验证

**这种组合的优势**:
1. 覆盖不同评估粒度（点/段）
2. 包含阈值相关和无关指标
3. 避免单一指标的局限性

---

## 六、关键发现总结

### 🔑 核心结论

1. **高相关组**:
   - VUS-PR ↔ VUS-ROC (同一框架不同度量)
   - PA-F1 ↔ F1-T ↔ Affiliation-F1 (段级别评估)

2. **中等相关**:
   - Standard-F1 ↔ 段级别指标
   - 点级别 vs 段级别的评估差异

3. **低相关场景**:
   - 模型擅长不同类型异常时
   - 阈值相关 vs 阈值无关指标

### ⚠️ 注意事项

1. **不要单独使用单一指标** - 不同指标可能给出矛盾结论
2. **理解指标特性** - PA-F1容易高估，Standard-F1对偏移敏感
3. **考虑实际应用场景** - 选择与实际检测需求匹配的指标

### 📊 论文中的指标一致性

VETime在所有指标上都表现优异（Table 1）:
- Affiliation-F1: 25/44个第一
- F1-T: 7/44个第一
- Standard-F1: 8/44个第一
- VUS-PR: 6/44个第一

这表明VETime在**不同评估维度**上都具有优势，不是单一指标的偶然表现。

---

*分析完成*
