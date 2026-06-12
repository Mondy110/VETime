# 异常检测分析看板实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建高性能交互式异常检测分析看板，支持 700+ 样本、最大 15 万数据点的流畅可视化。

**Architecture:** FastAPI 后端提供 REST API，静态文件托管前端页面。前端使用 ECharts 的 large 模式渲染大数据量图表，通过 progressive 渐进渲染确保流畅交互。

**Tech Stack:** Python FastAPI, HTML5, Tailwind CSS CDN, Apache ECharts CDN

---

## Task 1: 创建项目目录结构

**Files:**
- Create: `static/` 目录

**Step 1: 创建 static 目录**

Run: `mkdir -p static`
Expected: 目录创建成功

---

## Task 2: 实现后端基础框架

**Files:**
- Create: `app.py`

**Step 1: 创建 FastAPI 应用骨架**

```python
#!/usr/bin/env python3
"""异常检测分析看板 - FastAPI 后端"""

import argparse
import os
import re
import pickle
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="异常检测分析看板")

# 全局变量存储数据目录路径
DATA_DIR: Path = None
DATASETS: Dict[str, List[str]] = {}  # {dataset_name: [sample_ids]}


def parse_filename(filename: str) -> tuple:
    """
    解析文件名，提取数据集和样本ID

    格式: {序号}_{数据集}_id_{样本号}_{类别}_tr_{长度}_1st_{编号}_results.pkl
    返回: (dataset, sample_id) 或 None
    """
    match = re.match(r'(\d+)_([A-Z]+)_(id_\d+)', filename)
    if match:
        dataset = match.group(2)
        sample_id = match.group(3)
        return dataset, sample_id
    return None


def scan_data_directory(data_dir: Path) -> Dict[str, List[str]]:
    """扫描数据目录，返回数据集-样本映射"""
    datasets = {}

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    pkl_files = list(data_dir.glob("*_results.pkl"))
    print(f"扫描到 {len(pkl_files)} 个 pkl 文件")

    for pkl_file in pkl_files:
        result = parse_filename(pkl_file.name)
        if result:
            dataset, sample_id = result
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(sample_id)
        else:
            print(f"警告: 无法解析文件名: {pkl_file.name}")

    # 打印统计信息
    print("\n数据集统计:")
    for ds, samples in sorted(datasets.items()):
        print(f"  {ds}: {len(samples)} 个样本")

    return datasets


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse("static/index.html")


@app.get("/api/datasets")
async def get_datasets():
    """获取数据集列表及样本"""
    return DATASETS


@app.get("/api/data/{dataset}/{sample_id}")
async def get_sample_data(dataset: str, sample_id: str):
    """获取样本数据"""
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset}")

    if sample_id not in DATASETS[dataset]:
        raise HTTPException(status_code=404, detail=f"样本不存在: {sample_id}")

    # 查找对应的 pkl 文件
    pattern = f"*_{dataset}_{sample_id}_*_results.pkl"
    matching_files = list(DATA_DIR.glob(pattern))

    if not matching_files:
        raise HTTPException(status_code=404, detail=f"找不到数据文件: {pattern}")

    pkl_file = matching_files[0]

    try:
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)

        return {
            "value": df["value"].tolist(),
            "label": df["label"].tolist(),
            "anomaly_score": df["anomaly_score"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取数据失败: {str(e)}")


# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    global DATA_DIR, DATASETS

    parser = argparse.ArgumentParser(description="异常检测分析看板")
    parser.add_argument("--data-dir", type=str, default="./output/VETime",
                        help="数据目录路径 (默认: ./output/VETime)")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务器端口 (默认: 8000)")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    print(f"数据目录: {DATA_DIR}")

    # 扫描数据目录
    DATASETS = scan_data_directory(DATA_DIR)

    print(f"\n服务启动: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
```

**Step 2: 测试后端启动**

Run: `python app.py --data-dir ./output/VETime --port 8000`
Expected: 打印数据集统计，服务启动成功

---

## Task 3: 创建前端 HTML 页面

**Files:**
- Create: `static/index.html`

**Step 1: 创建 HTML 页面结构**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>异常检测分析看板</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b border-gray-200">
        <div class="px-6 py-4">
            <h1 class="text-xl font-semibold text-gray-800">异常检测分析看板</h1>
        </div>
    </header>

    <div class="flex">
        <!-- Sidebar -->
        <aside class="w-60 bg-white border-r border-gray-200 min-h-screen p-4">
            <div class="mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-2">数据集</label>
                <select id="datasetSelect" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option value="">请选择数据集</option>
                </select>
            </div>

            <div class="mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-2">样本</label>
                <select id="sampleSelect" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500" disabled>
                    <option value="">请先选择数据集</option>
                </select>
            </div>

            <!-- Loading Spinner -->
            <div id="loadingSpinner" class="hidden flex items-center justify-center py-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <span class="ml-2 text-gray-600">加载中...</span>
            </div>

            <!-- Error Message -->
            <div id="errorMessage" class="hidden bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md text-sm">
            </div>

            <!-- Sample Info -->
            <div id="sampleInfo" class="hidden mt-6 p-4 bg-gray-50 rounded-md">
                <h3 class="text-sm font-medium text-gray-700 mb-2">样本信息</h3>
                <div class="text-sm text-gray-600">
                    <p>数据点: <span id="dataLength">-</span></p>
                    <p>异常点: <span id="anomalyCount">-</span></p>
                </div>
            </div>
        </aside>

        <!-- Main Dashboard -->
        <main class="flex-1 p-6">
            <!-- Chart 1: Time-Series Alignment -->
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
                <div class="px-4 py-3 border-b border-gray-200">
                    <h2 class="text-lg font-medium text-gray-800">时序与异常分数对齐图</h2>
                </div>
                <div id="chart1" class="chart-container"></div>
            </div>

            <!-- Chart 2: Distribution Histogram -->
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
                <div class="px-4 py-3 border-b border-gray-200">
                    <h2 class="text-lg font-medium text-gray-800">区分度分析直方图</h2>
                </div>
                <div id="chart2" class="chart-container"></div>
            </div>

            <!-- Chart 3: Correlation Scatter -->
            <div class="bg-white rounded-lg shadow-sm border border-gray-200">
                <div class="px-4 py-3 border-b border-gray-200">
                    <h2 class="text-lg font-medium text-gray-800">关联性散点图</h2>
                </div>
                <div id="chart3" class="chart-container"></div>
            </div>
        </main>
    </div>

    <!-- Toast Container -->
    <div id="toastContainer" class="fixed bottom-4 right-4 z-50"></div>

    <script src="/static/app.js"></script>
</body>
</html>
```

---

## Task 4: 创建 CSS 样式文件

**Files:**
- Create: `static/style.css`

**Step 1: 创建样式文件**

```css
/* 图表容器高度 */
.chart-container {
    width: 100%;
    height: 350px;
    padding: 10px;
}

/* 加载动画 */
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

.animate-spin {
    animation: spin 1s linear infinite;
}

/* Toast 通知样式 */
.toast {
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    opacity: 0;
    transform: translateY(20px);
    transition: all 0.3s ease;
}

.toast.show {
    opacity: 1;
    transform: translateY(0);
}

.toast.error {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
}

.toast.success {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #16a34a;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .chart-container {
        height: 280px;
    }
}
```

---

## Task 5: 创建前端 JavaScript 逻辑（第一部分 - 初始化与数据获取）

**Files:**
- Create: `static/app.js`

**Step 1: 创建 app.js 基础结构和数据获取函数**

```javascript
// 全局变量
let datasets = {};
let chart1, chart2, chart3;
let currentData = null;

// DOM 元素
const datasetSelect = document.getElementById('datasetSelect');
const sampleSelect = document.getElementById('sampleSelect');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const sampleInfo = document.getElementById('sampleInfo');
const dataLengthSpan = document.getElementById('dataLength');
const anomalyCountSpan = document.getElementById('anomalyCount');

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    await loadDatasets();
    setupEventListeners();
});

// 初始化图表
function initCharts() {
    chart1 = echarts.init(document.getElementById('chart1'));
    chart2 = echarts.init(document.getElementById('chart2'));
    chart3 = echarts.init(document.getElementById('chart3'));

    // 窗口大小改变时重绘图表
    window.addEventListener('resize', () => {
        chart1.resize();
        chart2.resize();
        chart3.resize();
    });

    // 显示空状态
    showEmptyCharts();
}

// 显示空状态图表
function showEmptyCharts() {
    const emptyOption = {
        title: {
            text: '请选择数据集和样本',
            left: 'center',
            top: 'center',
            textStyle: {
                color: '#999',
                fontSize: 14
            }
        }
    };
    chart1.setOption(emptyOption);
    chart2.setOption(emptyOption);
    chart3.setOption(emptyOption);
}

// 加载数据集列表
async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets');
        datasets = await response.json();

        // 填充数据集下拉框
        Object.keys(datasets).sort().forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset;
            option.textContent = `${dataset} (${datasets[dataset].length})`;
            datasetSelect.appendChild(option);
        });
    } catch (error) {
        showError('加载数据集列表失败: ' + error.message);
        datasetSelect.disabled = true;
    }
}

// 设置事件监听器
function setupEventListeners() {
    datasetSelect.addEventListener('change', onDatasetChange);
    sampleSelect.addEventListener('change', onSampleChange);
}

// 数据集选择变化
function onDatasetChange() {
    const dataset = datasetSelect.value;

    // 重置样本下拉框
    sampleSelect.innerHTML = '<option value="">请选择样本</option>';

    if (!dataset) {
        sampleSelect.disabled = true;
        sampleInfo.classList.add('hidden');
        showEmptyCharts();
        return;
    }

    // 填充样本下拉框
    const samples = datasets[dataset];
    samples.sort().forEach(sample => {
        const option = document.createElement('option');
        option.value = sample;
        option.textContent = sample;
        sampleSelect.appendChild(option);
    });

    sampleSelect.disabled = false;
}

// 样本选择变化
async function onSampleChange() {
    const dataset = datasetSelect.value;
    const sampleId = sampleSelect.value;

    if (!sampleId) {
        sampleInfo.classList.add('hidden');
        showEmptyCharts();
        return;
    }

    showLoading(true);
    hideError();

    try {
        const response = await fetch(`/api/data/${dataset}/${sampleId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        currentData = await response.json();
        updateCharts();
        updateSampleInfo();
    } catch (error) {
        showToast('加载数据失败: ' + error.message, 'error');
        showEmptyCharts();
    } finally {
        showLoading(false);
    }
}

// 更新样本信息
function updateSampleInfo() {
    if (!currentData) return;

    dataLengthSpan.textContent = currentData.value.length.toLocaleString();
    anomalyCountSpan.textContent = currentData.label.filter(l => l === 1).length.toLocaleString();
    sampleInfo.classList.remove('hidden');
}

// 显示/隐藏加载状态
function showLoading(show) {
    loadingSpinner.classList.toggle('hidden', !show);
}

// 显示错误信息
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

// 隐藏错误信息
function hideError() {
    errorMessage.classList.add('hidden');
}

// 显示 Toast 通知
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    // 触发动画
    setTimeout(() => toast.classList.add('show'), 10);

    // 3秒后移除
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
```

**Step 2: 添加图表渲染函数（追加到 app.js）**

```javascript
// 更新所有图表
function updateCharts() {
    if (!currentData) return;

    updateChart1();
    updateChart2();
    updateChart3();
}

// 图一：时序与异常分数对齐图
function updateChart1() {
    const { value, label, anomaly_score } = currentData;
    const xData = Array.from({ length: value.length }, (_, i) => i);

    // 计算 markArea：找出 label=1 的连续区间（包括单个异常点）
    const markAreas = [];
    let start = -1;
    for (let i = 0; i < label.length; i++) {
        if (label[i] === 1 && start === -1) {
            start = i;
        } else if (label[i] === 0 && start !== -1) {
            // 如果是单个异常点，给它一个最小宽度（前后各扩展0.5）
            if (start === i - 1) {
                markAreas.push([{ xAxis: start - 0.5 }, { xAxis: i - 0.5 }]);
            } else {
                markAreas.push([{ xAxis: start }, { xAxis: i - 1 }]);
            }
            start = -1;
        }
    }
    // 处理末尾的异常区间
    if (start !== -1) {
        if (start === label.length - 1) {
            markAreas.push([{ xAxis: start - 0.5 }, { xAxis: label.length - 0.5 }]);
        } else {
            markAreas.push([{ xAxis: start }, { xAxis: label.length - 1 }]);
        }
    }

    const option = {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },
        legend: {
            data: ['原始信号', '异常分数'],
            top: 10
        },
        grid: {
            left: 60,
            right: 60,
            top: 50,
            bottom: 80
        },
        xAxis: {
            type: 'category',
            data: xData,
            name: '时间步',
            nameLocation: 'middle',
            nameGap: 30
        },
        yAxis: [
            {
                type: 'value',
                name: '原始信号',
                position: 'left',
                nameLocation: 'middle',
                nameGap: 40
            },
            {
                type: 'value',
                name: '异常分数',
                position: 'right',
                nameLocation: 'middle',
                nameGap: 40,
                min: 0,
                max: 1
            }
        ],
        dataZoom: [
            {
                type: 'slider',
                xAxisIndex: 0,
                start: 0,
                end: 100,
                height: 20,
                bottom: 10
            },
            {
                type: 'inside',
                xAxisIndex: 0,
                zoomOnMouseWheel: true,
                moveOnMouseMove: true
            }
        ],
        series: [
            {
                name: '原始信号',
                type: 'line',
                yAxisIndex: 0,
                data: value,
                large: true,
                progressive: 3000,
                lineStyle: { width: 1 },
                itemStyle: { color: '#3b82f6' }
            },
            {
                name: '异常分数',
                type: 'line',
                yAxisIndex: 1,
                data: anomaly_score,
                large: true,
                progressive: 3000,
                lineStyle: { width: 1 },
                areaStyle: { opacity: 0.3 },
                itemStyle: { color: '#ef4444' }
            }
        ]
    };

    // 添加 markArea
    if (markAreas.length > 0) {
        option.series[0].markArea = {
            silent: true,
            itemStyle: {
                color: 'rgba(239, 68, 68, 0.15)'
            },
            data: markAreas
        };
    }

    chart1.setOption(option, true);
}

// 图二：区分度分析直方图
function updateChart2() {
    const { label, anomaly_score } = currentData;

    // 分离正常和异常分数
    const normalScores = [];
    const anomalyScores = [];

    for (let i = 0; i < label.length; i++) {
        if (label[i] === 0) {
            normalScores.push(anomaly_score[i]);
        } else {
            anomalyScores.push(anomaly_score[i]);
        }
    }

    // 计算直方图分箱
    const bins = 50;
    const allScores = anomaly_score;
    const minScore = Math.min(...allScores);
    const maxScore = Math.max(...allScores);
    const binWidth = (maxScore - minScore) / bins;

    // 初始化分箱
    const normalCounts = new Array(bins).fill(0);
    const anomalyCounts = new Array(bins).fill(0);
    const xLabels = [];

    for (let i = 0; i < bins; i++) {
        xLabels.push((minScore + i * binWidth).toFixed(3));
    }

    // 统计频数
    normalScores.forEach(score => {
        const binIndex = Math.min(Math.floor((score - minScore) / binWidth), bins - 1);
        normalCounts[binIndex]++;
    });

    anomalyScores.forEach(score => {
        const binIndex = Math.min(Math.floor((score - minScore) / binWidth), bins - 1);
        anomalyCounts[binIndex]++;
    });

    const option = {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' }
        },
        legend: {
            data: [`正常 (${normalScores.length})`, `异常 (${anomalyScores.length})`],
            top: 10
        },
        grid: {
            left: 60,
            right: 30,
            top: 50,
            bottom: 60
        },
        xAxis: {
            type: 'category',
            data: xLabels,
            name: '异常分数',
            nameLocation: 'middle',
            nameGap: 30,
            axisLabel: {
                rotate: 45,
                interval: Math.floor(bins / 10)
            }
        },
        yAxis: {
            type: 'value',
            name: '频数',
            nameLocation: 'middle',
            nameGap: 40
        },
        series: [
            {
                name: `正常 (${normalScores.length})`,
                type: 'bar',
                data: normalCounts,
                itemStyle: {
                    color: 'rgba(59, 130, 246, 0.5)',
                    borderColor: '#3b82f6'
                },
                barWidth: '60%'
            },
            {
                name: `异常 (${anomalyScores.length})`,
                type: 'bar',
                data: anomalyCounts,
                itemStyle: {
                    color: 'rgba(239, 68, 68, 0.5)',
                    borderColor: '#ef4444'
                },
                barWidth: '60%'
            }
        ]
    };

    chart2.setOption(option, true);
}

// 图三：关联性散点图
function updateChart3() {
    const { value, anomaly_score } = currentData;

    // 计算散点数据 [abs(value), anomaly_score]
    const scatterData = [];
    for (let i = 0; i < value.length; i++) {
        scatterData.push([Math.abs(value[i]), anomaly_score[i]]);
    }

    const option = {
        animation: false,
        tooltip: {
            trigger: 'item',
            formatter: params => `|value|: ${params.value[0].toFixed(4)}<br/>score: ${params.value[1].toFixed(4)}`
        },
        grid: {
            left: 70,
            right: 30,
            top: 30,
            bottom: 80
        },
        xAxis: {
            type: 'value',
            name: '|value|',
            nameLocation: 'middle',
            nameGap: 40
        },
        yAxis: {
            type: 'value',
            name: '异常分数',
            nameLocation: 'middle',
            nameGap: 50,
            min: 0,
            max: 1
        },
        dataZoom: [
            {
                type: 'slider',
                xAxisIndex: 0,
                start: 0,
                end: 100,
                height: 20,
                bottom: 10
            },
            {
                type: 'slider',
                yAxisIndex: 0,
                start: 0,
                end: 100,
                width: 20,
                right: 10
            },
            {
                type: 'inside',
                xAxisIndex: 0,
                yAxisIndex: 0
            }
        ],
        series: [{
            type: 'scatter',
            data: scatterData,
            large: true,
            largeThreshold: 2000,
            symbolSize: 3,
            itemStyle: {
                color: 'rgba(59, 130, 246, 0.5)'
            }
        }]
    };

    chart3.setOption(option, true);
}
```

---

## Task 6: 测试与验证

**Step 1: 安装依赖**

Run: `pip install fastapi uvicorn`
Expected: 依赖安装成功

**Step 2: 启动服务**

Run: `python app.py --data-dir ./output/VETime --port 8000`
Expected:
```
数据目录: output/VETime
扫描到 700 个 pkl 文件

数据集统计:
  IOPS: 17 个样本
  MGAB: 9 个样本
  NAB: 28 个样本
  ...

服务启动: http://localhost:8000
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Step 3: 访问页面**

在浏览器打开 `http://localhost:8000`
Expected:
- 页面正常加载
- 侧边栏显示数据集下拉框
- 选择数据集后显示样本列表
- 选择样本后显示三个图表

**Step 4: 验证大数据量性能**

选择 UCR 数据集中的样本（部分样本超过 10 万数据点）
Expected:
- 图表加载时间 < 3 秒
- 缩放、拖拽操作流畅，无明显卡顿
- dataZoom 滑块正常工作

---

## 实现总结

**文件清单：**
```
/mnt/sda/cjmProject/VETime/
├── app.py              # FastAPI 后端 (~120 行)
└── static/
    ├── index.html      # 页面结构 (~90 行)
    ├── style.css       # 样式 (~40 行)
    └── app.js          # 图表逻辑 (~280 行)
```

**启动命令：**
```bash
python app.py --data-dir ./output/VETime --port 8000
```

**访问地址：**
```
http://localhost:8000
```
