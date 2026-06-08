# 时间序列异常检测可视化项目设计文档

**日期**: 2026-06-08
**状态**: 已批准

## 1. 项目概述

创建一个独立的 Web 可视化项目，用于读取和可视化 15 万条时间序列异常检测数据集。

### 1.1 技术栈
- **后端**: FastAPI + Uvicorn
- **前端**: HTML 原生 + TailwindCSS (CDN) + ECharts 5.4.3
- **数据处理**: Numpy, Pickle

### 1.2 数据集
- **路径**: `dataset/vetime_train_all_150000.pkl`
- **大小**: 约 13GB
- **样本数**: 150,000
- **异常类型**: 30+ 种

## 2. 项目结构

```
VETime/
└── visualization/
    ├── main.py              # FastAPI 后端主程序
    ├── templates/
    │   └── index.html       # 前端页面
    ├── static/
    │   └── app.js           # 前端可视化逻辑
    └── requirements.txt     # 依赖清单
```

## 3. 后端设计

### 3.1 数据加载策略：按需懒加载

```python
# 全局变量存储加载状态
_data_cache = None        # 缓存的数据
_loading = False          # 加载锁

def get_data():
    """懒加载函数：首次调用时加载数据，后续调用直接返回缓存"""
    global _data_cache, _loading

    if _data_cache is not None:
        return _data_cache

    if _loading:
        raise HTTPException(503, "数据正在加载中，请稍后重试")

    _loading = True
    try:
        with open('dataset/vetime_train_all_150000.pkl', 'rb') as f:
            _data_cache = pickle.load(f)
        return _data_cache
    finally:
        _loading = False
```

### 3.2 API 路由

| 路由 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/` | GET | 无 | 返回前端页面 |
| `/api/status` | GET | 无 | 检查数据加载状态 |
| `/api/anomaly_types` | GET | 无 | 返回所有异常类型列表 |
| `/api/samples` | GET | `type`, `limit=10` | 按类型筛选样本 |

### 3.3 异常类型提取

```python
def extract_anomaly_type(anomalies_dict):
    """从 anomalies 字典中提取异常类型"""
    # anomalies 格式: {'0_outlier': (511, 512), '1_upward spike': (100, 105)}
    types = []
    for key in anomalies_dict.keys():
        parts = key.split('_', 1)
        if len(parts) > 1:
            types.append(parts[1])
    return types
```

### 3.4 NumPy 序列化处理

**关键点**: 必须使用 `.tolist()` 将 numpy.ndarray 转换为 Python list，否则 FastAPI 无法序列化为 JSON。

```python
sample_data = {
    'normal_time_series': sample['normal_time_series'].tolist(),
    'time_series': sample['time_series'].tolist(),
    'labels': sample['labels'].tolist(),
    'attribute': sample['attribute']
}
```

## 4. 前端设计

### 4.1 页面布局

```
┌─────────────────────────────────────────────────────────────┐
│  时间序列异常检测可视化                                      │
│  ─────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 异常类型: [下拉选择框 ▼]    数量: [10 ▼]    [加载数据]  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    样本图表                              ││
│  │  上半区: Normal Time Series (纯净序列)                   ││
│  │  下半区: Time Series (含异常序列) + 异常标记             ││
│  │  缩放滑块: dataZoom                                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 4.2 TailwindCSS 样式

| 组件 | 样式类 |
|------|--------|
| 页面容器 | `min-h-screen bg-gray-100 p-6` |
| 控制面板 | `bg-white rounded-lg shadow-md p-4 mb-6` |
| 下拉框 | `border rounded px-4 py-2` |
| 按钮 | `bg-blue-500 text-white px-6 py-2 rounded` |
| 图表容器 | `w-full h-[500px] bg-white rounded-lg shadow` |

## 5. ECharts 图表设计

### 5.1 双网格布局

- **上方网格 (gridIndex: 0)**: 显示 normal_time_series（蓝色线条）
- **下方网格 (gridIndex: 1)**: 显示 time_series（绿色线条）+ 异常标记

### 5.2 异常标记方式

| 标记类型 | ECharts 组件 | 样式 |
|----------|-------------|------|
| 异常区间 | `markArea` | 半透明红色 `rgba(239, 68, 68, 0.3)` |
| 异常点 | `markPoint` | 红色圆点 `#ef4444` |

### 5.3 异常区间计算

```javascript
function computeAnomalyRegions(labels) {
    const regions = [];
    let start = -1;

    for (let i = 0; i < labels.length; i++) {
        if (labels[i] === 1 && start === -1) {
            start = i;
        } else if (labels[i] === 0 && start !== -1) {
            regions.push({ start, end: i - 1 });
            start = -1;
        }
    }
    if (start !== -1) regions.push({ start, end: labels.length - 1 });

    return regions;
}
```

### 5.4 缩放功能

使用 `dataZoom` 组件支持图表缩放：
```javascript
dataZoom: [
    { type: 'slider', xAxisIndex: [0, 1], bottom: 20 }
]
```

## 6. 依赖清单

```
fastapi
uvicorn
numpy
jinja2
python-multipart
```

## 7. 关键设计决策

1. **懒加载而非启动时加载**: 13GB 数据首次请求时加载，避免启动延迟
2. **完整异常类型列表**: 前端下拉框展示全部 30+ 种类型
3. **单文件后端架构**: 便于教学和理解
4. **CDN 引入前端框架**: 无需构建工具，开箱即用
