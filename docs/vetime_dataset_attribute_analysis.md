# VETime 数据集 Attribute 字段统计分析报告

> 数据集文件: `dataset/vetime_train_all_150000.pkl`  
> 样本总数: **150,000**  
> 分析日期: 2026-06-09

---

## 目录

1. [字段结构概述](#1-字段结构概述)
2. [Seasonal 季节性特征](#2-seasonal-季节性特征)
3. [Trend 趋势特征](#3-trend-趋势特征)
4. [Frequency 频率特征](#4-frequency-频率特征)
5. [Noise 噪声特征](#5-noise-噪声特征)
6. [Anomalies 异常类型](#6-anomalies-异常类型)
7. [异常数量分布](#7-异常数量分布)
8. [Full Attribute Pool 详细信息](#8-full-attribute-pool-详细信息)
9. [示例样本](#9-示例样本)

---

## 1. 字段结构概述

每个样本的 `attribute` 字段是一个字典，包含以下 **6 个顶层字段**：

```python
{
    'seasonal': str,              # 季节性特征类型
    'trend': str,                 # 趋势特征类型
    'frequency': str,             # 频率特征类型
    'noise': str,                 # 噪声特征类型
    'anomalies': dict,            # 异常类型及位置信息
    'full_attribute_pool': dict   # 完整的详细信息
}
```

### 字段说明表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `seasonal` | string | 描述时间序列的周期性波动模式 |
| `trend` | string | 描述时间序列的整体趋势走向 |
| `frequency` | string | 描述周期性波动的频率高低 |
| `noise` | string | 描述时间序列的噪声水平 |
| `anomalies` | dict | 记录异常的类型及其位置区间 |
| `full_attribute_pool` | dict | 包含振幅、周期、统计量等完整信息 |

---

## 2. Seasonal 季节性特征

季节性特征描述时间序列的周期性波动模式，共 **5 种类型**。

### 统计分布

| 类型 | 英文名称 | 数量 | 占比 |
|------|----------|------|------|
| 小波周期性波动 | wavelet periodic fluctuation | 47,134 | 31.42% |
| 正弦周期性波动 | sin periodic fluctuation | 46,436 | 30.96% |
| 无周期性波动 | no periodic fluctuation | 41,048 | 27.37% |
| 三角波周期性波动 | triangle periodic fluctuation | 7,788 | 5.19% |
| 方波周期性波动 | square periodic fluctuation | 7,594 | 5.06% |

### 类型说明

- **小波周期性波动 (wavelet periodic fluctuation)**: 使用小波函数生成的周期性模式，具有多尺度特征
- **正弦周期性波动 (sin periodic fluctuation)**: 标准的正弦波模式，平滑且规律
- **无周期性波动 (no periodic fluctuation)**: 时间序列没有明显的周期性特征
- **三角波周期性波动 (triangle periodic fluctuation)**: 三角波形周期性变化，线性上升和下降
- **方波周期性波动 (square periodic fluctuation)**: 方波形周期性变化，突变式的高低切换

---

## 3. Trend 趋势特征

趋势特征描述时间序列的整体走向，共 **5 种类型**。

### 统计分布

| 类型 | 英文名称 | 数量 | 占比 |
|------|----------|------|------|
| 复合趋势 | multiple | 41,066 | 27.38% |
| 保持平稳 | keep steady | 27,646 | 18.43% |
| 下降趋势 | decrease | 27,451 | 18.30% |
| 上升趋势 | increase | 27,400 | 18.27% |
| ARIMA趋势 | arima | 26,437 | 17.62% |

### 类型说明

- **复合趋势 (multiple)**: 包含多种趋势变化的组合，如先升后降等复杂模式
- **保持平稳 (keep steady)**: 整体趋势平稳，无明显上升或下降
- **下降趋势 (decrease)**: 整体呈现下降走向
- **上升趋势 (increase)**: 整体呈现上升走向
- **ARIMA趋势 (arima)**: 由ARIMA模型生成的趋势模式

---

## 4. Frequency 频率特征

频率特征描述周期性波动的频率高低，共 **4 种类型**。

### 统计分布

| 类型 | 英文名称 | 数量 | 占比 |
|------|----------|------|------|
| 低频波动 | low frequency | 43,426 | 28.95% |
| 无周期性 | no periodicity | 41,048 | 27.37% |
| 高频波动 | high frequency | 32,941 | 21.96% |
| 中频波动 | moderate frequency | 32,585 | 21.72% |

### 类型说明

- **低频波动 (low frequency)**: 周期较长，波动频率低
- **无周期性 (no periodicity)**: 没有周期性波动特征
- **高频波动 (high frequency)**: 周期较短，波动频率高
- **中频波动 (moderate frequency)**: 周期适中，波动频率中等

---

## 5. Noise 噪声特征

噪声特征描述时间序列中的噪声水平，共 **4 种类型**。

### 统计分布

| 类型 | 英文名称 | 数量 | 占比 |
|------|----------|------|------|
| 几乎无噪声 | almost no noise | 84,676 | 56.45% |
| 低噪声 | low noise | 58,003 | 38.67% |
| 中等噪声 | moderate noise | 3,667 | 2.44% |
| 高噪声 | high noise | 3,654 | 2.44% |

### 类型说明

- **几乎无噪声 (almost no noise)**: 曲线非常平滑，噪声标准差极小（约0.07以下）
- **低噪声 (low noise)**: 存在轻微噪声，但不影响主要模式
- **中等噪声 (moderate noise)**: 噪声水平适中，对曲线有一定影响
- **高噪声 (high noise)**: 噪声较大，可能干扰模式的识别

---

## 6. Anomalies 异常类型

### 6.1 异常记录格式

异常信息以字典形式存储，格式为：

```python
{
    '编号_异常类型': (起始位置, 结束位置),
    '0_outlier': (511, 512),
    '1_upward spike': (120, 125),
    ...
}
```

其中编号用于区分同一类型的不同异常实例。

### 6.2 异常类型分类统计

异常类型共分为 **6 大类别**，总计 **180+ 种**具体类型。

---

#### 6.2.1 点异常类 (Point Anomalies)

单点或多点的离群值异常，最常见的异常类型。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_outlier` | 51,900 | 34.60% |
| `1_outlier` | 38,075 | 25.38% |
| `2_outlier` | 22,797 | 15.20% |
| `3_outlier` | 4,284 | 2.86% |
| `4_outlier` | 4,098 | 2.73% |
| `5_outlier` | 3,887 | 2.59% |
| `6_outlier` | 3,571 | 2.38% |
| `7_outlier` | 3,250 | 2.17% |
| `8_outlier` | 2,948 | 1.97% |
| `9_outlier` | 2,654 | 1.77% |
| `10_outlier` | 2,395 | 1.60% |
| `11_outlier` | 2,139 | 1.43% |
| `12_outlier` | 1,887 | 1.26% |
| `13_outlier` | 1,699 | 1.13% |
| `14_outlier` | 1,515 | 1.01% |
| `15_outlier` | 1,336 | 0.89% |
| `16_outlier` | 1,194 | 0.80% |
| `17_outlier` | 1,040 | 0.69% |
| `18_outlier` | 906 | 0.60% |
| `19_outlier` | 779 | 0.52% |
| `20_outlier` | 662 | 0.44% |
| `21_outlier` | 576 | 0.38% |
| `22_outlier` | 497 | 0.33% |
| `23_outlier` | 423 | 0.28% |
| `24_outlier` | 348 | 0.23% |
| `25_outlier` | 281 | 0.19% |
| `26_outlier` | 214 | 0.14% |
| `27_outlier` | 157 | 0.10% |
| `28_outlier` | 115 | 0.08% |
| `29_outlier` | 77 | 0.05% |
| `30_outlier` | 39 | 0.03% |
| `31_outlier` | 11 | 0.01% |
| `32_outlier` | 2 | 0.00% |

**说明**: 编号表示同一样本中该类型异常的序号，如 `0_outlier` 表示第一个离群点，`1_outlier` 表示第二个离群点。

---

#### 6.2.2 尖峰异常类 (Spike Anomalies)

向上或向下的尖峰状异常，包括单点尖峰、宽尖峰和连续尖峰。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_upward spike` | 15,590 | 10.39% |
| `0_downward spike` | 13,057 | 8.70% |
| `1_upward spike` | 10,460 | 6.97% |
| `1_downward spike` | 8,945 | 5.96% |
| `2_upward spike` | 5,586 | 3.72% |
| `0_continuous upward spike` | 5,198 | 3.47% |
| `2_downward spike` | 4,647 | 3.10% |
| `0_wide downward spike` | 4,189 | 2.79% |
| `0_wide upward spike` | 4,127 | 2.75% |
| `0_increase after upward spike` | 4,036 | 2.69% |
| `0_increase after downward spike` | 4,019 | 2.68% |
| `0_decrease after downward spike` | 4,001 | 2.67% |
| `0_decrease after upward spike` | 3,943 | 2.63% |
| `1_continuous upward spike` | 3,545 | 2.36% |
| `1_wide upward spike` | 2,496 | 1.66% |
| `1_wide downward spike` | 2,467 | 1.64% |
| `1_increase after upward spike` | 2,687 | 1.79% |
| `1_decrease after downward spike` | 2,638 | 1.76% |
| `1_decrease after upward spike` | 2,634 | 1.76% |
| `0_continuous downward spike` | 2,649 | 1.77% |
| `1_increase after downward spike` | 2,513 | 1.68% |
| `2_continuous upward spike` | 1,794 | 1.20% |
| `1_continuous downward spike` | 1,718 | 1.15% |
| `2_increase after upward spike` | 1,314 | 0.88% |
| `2_increase after downward spike` | 1,258 | 0.84% |
| `2_decrease after downward spike` | 1,250 | 0.83% |
| `2_decrease after upward spike` | 1,200 | 0.80% |
| `2_wide upward spike` | 868 | 0.58% |
| `2_wide downward spike` | 923 | 0.62% |
| `2_continuous downward spike` | 878 | 0.59% |

**说明**:
- `upward spike` / `downward spike`: 单点向上/向下尖峰
- `wide upward spike` / `wide downward spike`: 宽向上/向下尖峰（跨度多个点）
- `continuous upward spike` / `continuous downward spike`: 连续向上/向下尖峰序列
- `increase after spike` / `decrease after spike`: 尖峰后的趋势变化

---

#### 6.2.3 趋势变化类 (Trend Change Anomalies)

趋势突变或异常转折模式。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_slow decline followed by rapid rise` | 2,728 | 1.82% |
| `0_downward convex` | 2,723 | 1.82% |
| `0_rapid rise followed by slow decline` | 2,700 | 1.80% |
| `0_sudden increase` | 2,681 | 1.79% |
| `0_slow rise followed by rapid decline` | 2,660 | 1.77% |
| `0_rapid decline followed by slow rise` | 2,652 | 1.77% |
| `0_sudden decrease` | 2,574 | 1.72% |
| `1_slow rise followed by rapid decline` | 1,698 | 1.13% |
| `1_slow decline followed by rapid rise` | 1,689 | 1.13% |
| `1_sudden decrease` | 1,781 | 1.19% |
| `1_sudden increase` | 1,742 | 1.16% |
| `1_rapid decline followed by slow rise` | 1,632 | 1.09% |
| `1_rapid rise followed by slow decline` | 1,620 | 1.08% |
| `2_sudden increase` | 941 | 0.63% |
| `2_sudden decrease` | 901 | 0.60% |
| `2_rapid rise followed by slow decline` | 633 | 0.42% |
| `2_rapid decline followed by slow rise` | 618 | 0.41% |
| `2_slow decline followed by rapid rise` | 607 | 0.40% |
| `2_slow rise followed by rapid decline` | 592 | 0.39% |

**说明**:
- `sudden increase` / `sudden decrease`: 突然增加/减少
- `slow rise followed by rapid decline`: 缓慢上升后快速下降
- `slow decline followed by rapid rise`: 缓慢下降后快速上升
- `rapid rise followed by slow decline`: 快速上升后缓慢下降
- `rapid decline followed by slow rise`: 快速下降后缓慢上升

---

#### 6.2.4 形态变化类 (Shape Anomalies)

波形形态的改变，包括振幅、频率、相位等变化。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_shake` | 5,542 | 3.69% |
| `1_shake` | 3,354 | 2.24% |
| `0_upward convex` | 2,615 | 1.74% |
| `0_downward convex` | 2,723 | 1.82% |
| `0_waveform_inversion` | 1,232 | 0.82% |
| `0_amplitude_scaling` | 1,238 | 0.83% |
| `2_shake` | 1,238 | 0.83% |
| `0_noise_injection` | 1,207 | 0.80% |
| `0_frequency_change` | 1,174 | 0.78% |
| `1_amplitude_scaling` | 1,094 | 0.73% |
| `1_waveform_inversion` | 1,062 | 0.71% |
| `1_noise_injection` | 1,049 | 0.70% |
| `1_frequency_change` | 1,035 | 0.69% |
| `2_waveform_inversion` | 985 | 0.66% |
| `2_noise_injection` | 981 | 0.65% |
| `2_frequency_change` | 980 | 0.65% |
| `2_amplitude_scaling` | 932 | 0.62% |
| `3_waveform_inversion` | 839 | 0.56% |
| `3_noise_injection` | 894 | 0.60% |
| `3_frequency_change` | 886 | 0.59% |
| `3_amplitude_scaling` | 888 | 0.59% |
| `0_waveform_change` | 501 | 0.33% |
| `0_amplitude_change` | 487 | 0.32% |
| `0_phase_shift` | 478 | 0.32% |
| `0_shift_change` | 468 | 0.31% |
| `0_scale_change` | 465 | 0.31% |
| `0_family_change` | 464 | 0.31% |
| `1_shift_change` | 453 | 0.30% |
| `1_amplitude_change` | 438 | 0.29% |
| `1_family_change` | 438 | 0.29% |
| `1_waveform_change` | 436 | 0.29% |
| `2_scale_change` | 434 | 0.29% |
| `1_phase_shift` | 396 | 0.26% |
| `1_scale_change` | 396 | 0.26% |
| `2_family_change` | 403 | 0.27% |
| `2_waveform_change` | 401 | 0.27% |
| `1_upward convex` | 1,570 | 1.05% |
| `1_downward convex` | 1,637 | 1.09% |
| `2_phase_shift` | 377 | 0.25% |
| `2_amplitude_change` | 376 | 0.25% |
| `2_shift_change` | 363 | 0.24% |
| `2_upward convex` | 505 | 0.34% |
| `2_downward convex` | 482 | 0.32% |
| `3_family_change` | 373 | 0.25% |
| `3_amplitude_change` | 343 | 0.23% |
| `3_scale_change` | 342 | 0.23% |
| `3_shift_change` | 339 | 0.23% |
| `3_phase_shift` | 335 | 0.22% |
| `3_waveform_change` | 334 | 0.22% |

**说明**:
- `shake`: 抖动异常，波形出现不规则抖动
- `upward convex` / `downward convex`: 向上/向下凸起
- `amplitude_scaling`: 振幅缩放，振幅比例发生变化
- `amplitude_change`: 振幅变化
- `waveform_inversion`: 波形反转
- `waveform_change`: 波形变化
- `frequency_change`: 频率变化
- `phase_shift`: 相位偏移
- `noise_injection`: 噪声注入
- `shift_change`: 偏移变化
- `scale_change`: 尺度变化
- `family_change`: 族变化

---

#### 6.2.5 谐波变化类 (Harmonic Anomalies)

谐波的添加、移除或修改，涉及频率成分的变化。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_modify_harmonic_mod_phase` | 292 | 0.19% |
| `0_add_harmonic` | 291 | 0.19% |
| `0_modify_harmonic_mod_freq` | 287 | 0.19% |
| `0_modify_harmonic_amp_mod` | 268 | 0.18% |
| `0_modify_harmonic_phase` | 265 | 0.18% |
| `0_remove_harmonic` | 255 | 0.17% |
| `1_remove_harmonic` | 262 | 0.17% |
| `1_add_harmonic` | 252 | 0.17% |
| `2_add_wavelet` | 259 | 0.17% |
| `1_remove_wavelet` | 262 | 0.17% |
| `2_remove_harmonic` | 223 | 0.15% |
| `1_modify_harmonic_mod_phase` | 237 | 0.16% |
| `1_modify_harmonic_mod_freq` | 233 | 0.16% |
| `1_modify_harmonic_phase` | 224 | 0.15% |
| `1_modify_harmonic_amp_mod` | 224 | 0.15% |
| `2_modify_harmonic_mod_phase` | 224 | 0.15% |
| `2_modify_harmonic_mod_freq` | 226 | 0.15% |
| `2_modify_harmonic_phase` | 218 | 0.15% |
| `2_add_harmonic` | 207 | 0.14% |
| `2_modify_harmonic_amp_mod` | 206 | 0.14% |
| `3_modify_harmonic_mod_phase` | 213 | 0.14% |
| `3_add_harmonic` | 203 | 0.14% |
| `3_modify_harmonic_mod_freq` | 199 | 0.13% |
| `3_modify_harmonic_phase` | 196 | 0.13% |
| `3_modify_harmonic_amp_mod` | 196 | 0.13% |
| `3_remove_harmonic` | 180 | 0.12% |

**说明**:
- `add_harmonic`: 添加谐波成分
- `remove_harmonic`: 移除谐波成分
- `modify_harmonic_phase`: 修改谐波相位
- `modify_harmonic_mod_phase`: 修改谐波调制相位
- `modify_harmonic_mod_freq`: 修改谐波调制频率
- `modify_harmonic_amp_mod`: 修改谐波振幅调制

---

#### 6.2.6 小波变化类 (Wavelet Anomalies)

小波相关的变化，包括脉冲偏移、脉宽调制等。

| 异常类型 | 数量 | 占比 |
|----------|------|------|
| `0_remove_wavelet` | 295 | 0.20% |
| `0_pulse_width_modulation` | 231 | 0.15% |
| `0_add_wavelet` | 267 | 0.18% |
| `0_pulse_shift` | 251 | 0.17% |
| `1_add_wavelet` | 284 | 0.19% |
| `2_add_wavelet` | 259 | 0.17% |
| `2_remove_wavelet` | 226 | 0.15% |
| `1_remove_wavelet` | 262 | 0.17% |
| `1_pulse_width_modulation` | 222 | 0.15% |
| `2_pulse_width_modulation` | 202 | 0.13% |
| `3_add_wavelet` | 233 | 0.16% |
| `3_remove_wavelet` | 205 | 0.14% |
| `2_pulse_shift` | 190 | 0.13% |
| `1_pulse_shift` | 200 | 0.13% |
| `3_pulse_width_modulation` | 170 | 0.11% |
| `3_pulse_shift` | 163 | 0.11% |

**说明**:
- `add_wavelet`: 添加小波成分
- `remove_wavelet`: 移除小波成分
- `pulse_shift`: 脉冲偏移
- `pulse_width_modulation`: 脉宽调制

---

### 6.3 异常类型完整索引表

| 异常类型 | 中文说明 | 类别 |
|----------|----------|------|
| `outlier` | 离群点异常 | 点异常 |
| `upward spike` | 向上尖峰 | 尖峰异常 |
| `downward spike` | 向下尖峰 | 尖峰异常 |
| `wide upward spike` | 宽向上尖峰 | 尖峰异常 |
| `wide downward spike` | 宽向下尖峰 | 尖峰异常 |
| `continuous upward spike` | 连续向上尖峰 | 尖峰异常 |
| `continuous downward spike` | 连续向下尖峰 | 尖峰异常 |
| `shake` | 抖动异常 | 形态变化 |
| `sudden increase` | 突然增加 | 趋势变化 |
| `sudden decrease` | 突然减少 | 趋势变化 |
| `increase after upward spike` | 向上尖峰后增加 | 尖峰异常 |
| `increase after downward spike` | 向下尖峰后增加 | 尖峰异常 |
| `decrease after upward spike` | 向上尖峰后减少 | 尖峰异常 |
| `decrease after downward spike` | 向下尖峰后减少 | 尖峰异常 |
| `slow rise followed by rapid decline` | 缓升后急降 | 趋势变化 |
| `slow decline followed by rapid rise` | 缓降后急升 | 趋势变化 |
| `rapid rise followed by slow decline` | 急升后缓降 | 趋势变化 |
| `rapid decline followed by slow rise` | 急降后缓升 | 趋势变化 |
| `upward convex` | 向上凸起 | 形态变化 |
| `downward convex` | 向下凸起 | 形态变化 |
| `amplitude_scaling` | 振幅缩放 | 形态变化 |
| `amplitude_change` | 振幅变化 | 形态变化 |
| `waveform_inversion` | 波形反转 | 形态变化 |
| `waveform_change` | 波形变化 | 形态变化 |
| `frequency_change` | 频率变化 | 形态变化 |
| `phase_shift` | 相位偏移 | 形态变化 |
| `noise_injection` | 噪声注入 | 形态变化 |
| `shift_change` | 偏移变化 | 形态变化 |
| `scale_change` | 尺度变化 | 形态变化 |
| `family_change` | 族变化 | 形态变化 |
| `add_harmonic` | 添加谐波 | 谐波变化 |
| `remove_harmonic` | 移除谐波 | 谐波变化 |
| `modify_harmonic_phase` | 修改谐波相位 | 谐波变化 |
| `modify_harmonic_mod_phase` | 修改谐波调制相位 | 谐波变化 |
| `modify_harmonic_mod_freq` | 修改谐波调制频率 | 谐波变化 |
| `modify_harmonic_amp_mod` | 修改谐波振幅调制 | 谐波变化 |
| `add_wavelet` | 添加小波 | 小波变化 |
| `remove_wavelet` | 移除小波 | 小波变化 |
| `pulse_shift` | 脉冲偏移 | 小波变化 |
| `pulse_width_modulation` | 脉宽调制 | 小波变化 |

---

## 7. 异常数量分布

每个样本包含的异常数量分布统计：

| 异常数量 | 样本数 | 占比 |
|----------|--------|------|
| 1 个 | 45,902 | 30.60% |
| 2 个 | 46,653 | 31.10% |
| 3 个 | 45,630 | 30.42% |
| 4 个 | 7,717 | 5.14% |
| 5 个 | 211 | 0.14% |
| 6 个 | 316 | 0.21% |
| 7 个 | 321 | 0.21% |
| 8 个 | 302 | 0.20% |
| 9 个 | 294 | 0.20% |
| 10 个 | 259 | 0.17% |
| 11 个 | 256 | 0.17% |
| 12 个 | 252 | 0.17% |
| 13 个 | 188 | 0.13% |
| 14 个 | 184 | 0.12% |
| 15 个 | 179 | 0.12% |
| 16 个 | 142 | 0.09% |
| 17 个 | 154 | 0.10% |
| 18 个 | 134 | 0.09% |
| 19 个 | 127 | 0.08% |
| 20 个 | 117 | 0.08% |
| 21 个 | 86 | 0.06% |
| 22 个 | 79 | 0.05% |
| 23 个 | 74 | 0.05% |
| 24 个 | 75 | 0.05% |
| 25 个 | 67 | 0.04% |
| 26 个 | 67 | 0.04% |
| 27 个 | 57 | 0.04% |
| 28 个 | 42 | 0.03% |
| 29 个 | 38 | 0.03% |
| 30 个 | 38 | 0.03% |
| 31 个 | 28 | 0.02% |
| 32 个 | 9 | 0.01% |
| 33 个 | 2 | 0.00% |

### 异常数量统计摘要

| 统计指标 | 数值 |
|----------|------|
| **最小异常数** | 1 个 |
| **最大异常数** | 33 个 |
| **平均异常数** | 约 2.5 个 |
| **中位数** | 2 个 |
| **主要分布区间** | 1-3 个（占92.12%） |

### 分布特征分析

1. **集中分布**: 大部分样本（约92%）包含1-3个异常，这符合时间序列异常检测任务的典型场景
2. **长尾分布**: 少数样本包含较多异常（最多达33个），这些样本可能代表更复杂的异常模式
3. **平衡性**: 1个、2个、3个异常的样本数量相近，数据分布较为均衡

---

## 8. Full Attribute Pool 详细信息

`full_attribute_pool` 字段包含完整的属性信息，提供了更丰富的细节。

### 8.1 字段结构

```python
full_attribute_pool = {
    'seasonal': {
        'type': str,           # 季节性类型
        'amplitude': float,    # 振幅
        'detail': str,         # 详细描述
        'segments': [          # 分段信息列表
            {
                'amplitude': float,
                'position_start': int,
                'position_end': int,
                'description': str
            }
        ]
    },
    'trend': {
        'type': str,           # 趋势类型
        'start': float,        # 起始值
        'amplitude': float,    # 振幅
        'detail': str,         # 详细描述
        'trend_list': [(type, start, end), ...]  # 趋势分段列表
    },
    'local': [                 # 局部异常列表
        {
            'type': str,
            'position_start': int,
            'position_end': int,
            'amplitude': float,
            'detail': str
        }
    ],
    'frequency': {
        'type': str,           # 频率类型
        'period': float,       # 周期（点数）
        'detail': str
    },
    'noise': {
        'type': str,           # 噪声类型
        'std': float,          # 噪声标准差
        'detail': str
    },
    'num_seasonal_anomalies': int,        # 季节性异常数量
    'seasonal_anomalies': [],              # 季节性异常列表
    'overall_amplitude': float,            # 整体振幅
    'overall_bias': float,                 # 整体偏置
    'background_periodic_spike': {
        'enabled': bool,
        'count': int
    },
    'background_periodic_noise_modulation': {
        'enabled': bool,
        'count': int
    },
    'statistics': {
        'mean': float,        # 均值
        'std': float,         # 标准差
        'max': float,         # 最大值
        'min': float,         # 最小值
        'max_pos': int,       # 最大值位置
        'min_pos': int        # 最小值位置
    }
}
```

### 8.2 各字段详细说明

| 字段路径 | 类型 | 说明 |
|----------|------|------|
| `seasonal.type` | string | 季节性波动类型 |
| `seasonal.amplitude` | float | 季节性波动的振幅 |
| `seasonal.detail` | string | 季节性特征的详细文字描述 |
| `seasonal.segments` | list | 季节性波动的分段信息 |
| `trend.type` | string | 趋势类型 |
| `trend.start` | float | 时间序列的起始值 |
| `trend.amplitude` | float | 趋势变化的振幅 |
| `trend.detail` | string | 趋势特征的详细描述 |
| `trend.trend_list` | list | 趋势的分段列表，每段包含(类型, 起点, 终点) |
| `local` | list | 局部异常的详细列表 |
| `frequency.type` | string | 频率类型 |
| `frequency.period` | float | 波动周期（以点数计） |
| `noise.type` | string | 噪声类型 |
| `noise.std` | float | 噪声标准差 |
| `overall_amplitude` | float | 整体振幅 |
| `overall_bias` | float | 整体偏置 |
| `statistics.mean` | float | 时间序列均值 |
| `statistics.std` | float | 时间序列标准差 |
| `statistics.max` | float | 最大值 |
| `statistics.min` | float | 最小值 |
| `statistics.max_pos` | int | 最大值所在位置 |
| `statistics.min_pos` | int | 最小值所在位置 |

---

## 9. 示例样本

以下是一个完整的 attribute 字段示例：

### 9.1 简化视图

```python
attribute = {
    'seasonal': 'sin periodic fluctuation',
    'trend': 'keep steady',
    'frequency': 'high frequency',
    'noise': 'almost no noise',
    'anomalies': {
        '0_outlier': (511, 512)
    },
    'full_attribute_pool': { ... }  # 详细信息见下文
}
```

### 9.2 完整示例

```python
attribute = {
    'seasonal': 'sin periodic fluctuation',
    'trend': 'keep steady',
    'frequency': 'high frequency',
    'noise': 'almost no noise',
    'anomalies': {
        '0_outlier': (511, 512)
    },
    'full_attribute_pool': {
        'seasonal': {
            'type': 'sin periodic fluctuation',
            'amplitude': 4.158174073283624,
            'detail': 'Time series shows sin periodic fluctuation: '
                      'Periodic fluctuation with amplitude 4.2, '
                      'from point 0 to point 550.',
            'segments': [
                {
                    'amplitude': 4.16,
                    'position_start': 0,
                    'position_end': 550,
                    'description': 'Periodic fluctuation with amplitude 4.2, '
                                   'from point 0 to point 550'
                }
            ]
        },
        'trend': {
            'type': 'keep steady',
            'start': 1.39,
            'amplitude': 6.897702515807523,
            'detail': 'From the perspective of the slope, the overall trend is steady. '
                      'The value of time series starts from around 1.39 and ends at around -0.70, '
                      'with an overall amplitude of -2.09.',
            'trend_list': [('keep steady', 0, 549)]
        },
        'local': [
            {
                'type': 'outlier',
                'position_start': 511,
                'position_end': 512,
                'amplitude': 21.336062374870448,
                'detail': 'A single point outlier with positive amplitude 21.34'
            }
        ],
        'frequency': {
            'type': 'high frequency',
            'period': 10.7,
            'detail': 'Each fluctuation period is approximately 10.7 points, '
                      'thus the overall fluctuation is high frequency.'
        },
        'noise': {
            'type': 'almost no noise',
            'std': 0.069,
            'detail': 'The overall noise standard deviation is around 0.07, '
                      'very small compared to the overall change of the curve. '
                      'The curve is overall smooth with almost no noise.'
        },
        'num_seasonal_anomalies': 0,
        'overall_amplitude': 9.106371531685301,
        'overall_bias': -3.0616059465525236,
        'seasonal_anomalies': [],
        'background_periodic_spike': {
            'enabled': False,
            'count': 0
        },
        'background_periodic_noise_modulation': {
            'enabled': False,
            'count': 0
        },
        'statistics': {
            'mean': -0.64,
            'std': 1.5,
            'max': 19.23,
            'min': -2.89,
            'max_pos': 511,
            'min_pos': 435
        }
    }
}
```

### 9.3 示例解读

该样本的特征解读：

| 特征维度 | 值 | 解读 |
|----------|-----|------|
| 季节性 | 正弦周期性波动 | 时间序列呈现标准的正弦波模式 |
| 趋势 | 保持平稳 | 整体趋势稳定，无明显的上升或下降 |
| 频率 | 高频 | 周期约10.7个点，属于高频波动 |
| 噪声 | 几乎无噪声 | 噪声标准差仅0.07，曲线非常平滑 |
| 异常 | 1个离群点 | 在位置511-512处有一个正离群点（幅度21.34） |
| 整体振幅 | 9.11 | 时间序列的整体变化幅度 |
| 整体偏置 | -3.06 | 时间序列相对于零点的偏移量 |

---

## 10. 数据集特征总结

### 10.1 整体特征概览

| 特征维度 | 类别数量 | 主要类型分布 |
|----------|----------|--------------|
| 季节性 (Seasonal) | 5 | 小波波动(31.4%)、正弦波动(31.0%)、无周期(27.4%)、三角波(5.2%)、方波(5.1%) |
| 趋势 (Trend) | 5 | 复合趋势(27.4%)、平稳(18.4%)、下降(18.3%)、上升(18.3%)、ARIMA(17.6%) |
| 频率 (Frequency) | 4 | 低频(29.0%)、无周期(27.4%)、高频(22.0%)、中频(21.7%) |
| 噪声 (Noise) | 4 | 几乎无噪声(56.5%)、低噪声(38.7%)、中等噪声(2.4%)、高噪声(2.4%) |
| 异常类型 | 180+ | 点异常(34.6%)、尖峰异常(约25%)、趋势变化(约15%)、形态变化(约12%)等 |

### 10.2 数据集设计特点

1. **多样性**: 涵盖多种时间序列模式和异常类型
2. **层次性**: 提供简化的顶层字段和详细的完整信息
3. **精确标注**: 每个异常都有精确的位置区间
4. **统计完整**: 包含丰富的统计信息和特征描述

### 10.3 适用任务

该数据集适用于以下研究和应用场景：

| 任务类型 | 说明 |
|----------|------|
| **时间序列异常检测** | 识别和定位各类异常模式 |
| **异常分类** | 区分不同类型的异常（点异常、尖峰、趋势变化等） |
| **时间序列分类** | 基于特征进行时间序列的模式分类 |
| **时间序列生成** | 学习时间序列的生成模式 |
| **特征工程** | 研究时间序列特征的提取和表示 |
| **模型鲁棒性测试** | 测试模型在不同噪声水平下的表现 |

### 10.4 数据集统计摘要

```
┌─────────────────────────────────────────────────────────────┐
│                   VETime 数据集统计摘要                      │
├─────────────────────────────────────────────────────────────┤
│  样本总数:          150,000                                 │
│  季节性类型:        5 种                                     │
│  趋势类型:          5 种                                     │
│  频率类型:          4 种                                     │
│  噪声类型:          4 种                                     │
│  异常类型:          180+ 种                                  │
│  平均异常数/样本:   约 2.5 个                                │
│  主要异常分布:      1-3 个异常 (92.12%)                      │
│  无噪声样本占比:    56.45%                                   │
│  有周期性样本占比:  72.63%                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. 附录

### 11.1 异常类型编号说明

异常类型字段格式为 `{编号}_{类型名称}`，其中：

- **编号**: 从0开始，表示该类型异常在当前样本中的序号
- **类型名称**: 异常的具体类型

例如：
- `0_outlier`: 当前样本的第1个离群点
- `1_outlier`: 当前样本的第2个离群点
- `0_upward spike`: 当前样本的第1个向上尖峰

### 11.2 位置区间说明

异常的位置使用 `(起始位置, 结束位置)` 的元组格式表示：

- 单点异常: `(511, 512)` - 表示位置511的单点异常
- 区间异常: `(100, 120)` - 表示从位置100到120的区间异常

位置索引从0开始。

---

> **文档生成时间**: 2026-06-09  
> **数据集版本**: vetime_train_all_150000.pkl  
> **文档版本**: v1.0
