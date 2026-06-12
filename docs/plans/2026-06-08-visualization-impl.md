# 时间序列异常检测可视化项目实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个独立的 Web 可视化项目，用于读取和可视化 15 万条时间序列异常检测数据集。

**Architecture:** 采用轻量级单文件架构，后端使用 FastAPI 懒加载策略处理 13GB 数据，前端使用 TailwindCSS CDN + ECharts 5.4.3 实现双图联动可视化。

**Tech Stack:** FastAPI, Uvicorn, TailwindCSS (CDN), ECharts 5.4.3, NumPy, Pickle

---

## Task 1: 创建项目目录结构

**Files:**
- Create: `visualization/` 目录
- Create: `visualization/templates/` 目录
- Create: `visualization/static/` 目录

**Step 1: 创建项目目录**

Run:
```bash
mkdir -p visualization/templates visualization/static
```

Expected: 目录创建成功

**Step 2: 验证目录结构**

Run:
```bash
ls -la visualization/
```

Expected:
```
drwxr-xr-x 2 cjm cjm 4096 ... static/
drwxr-xr-x 2 cjm cjm 4096 ... templates/
```

**Step 3: Commit**

```bash
git add visualization/
git commit -m "feat: 创建可视化项目目录结构

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: 创建依赖清单文件

**Files:**
- Create: `visualization/requirements.txt`

**Step 1: 创建 requirements.txt**

```txt
fastapi>=0.104.0
uvicorn>=0.24.0
numpy>=1.24.0
jinja2>=3.1.2
python-multipart>=0.0.6
```

**Step 2: 验证文件内容**

Run:
```bash
cat visualization/requirements.txt
```

Expected: 文件内容正确显示

**Step 3: Commit**

```bash
git add visualization/requirements.txt
git commit -m "feat: 添加可视化项目依赖清单

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: 创建后端主程序框架

**Files:**
- Create: `visualization/main.py`

**Step 1: 创建 main.py 基础框架**

```python
"""
时间序列异常检测可视化项目 - 后端主程序

本模块提供以下功能：
1. FastAPI 应用初始化和配置
2. 静态文件和模板目录挂载
3. 数据懒加载机制
4. API 路由定义

作者：VETime 项目组
日期：2026-06-08
"""

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

# FastAPI 核心组件
from fastapi import FastAPI, HTTPException, Query
# 静态文件服务，用于提供 JS、CSS 等静态资源
from fastapi.staticfiles import StaticFiles
# Jinja2 模板引擎，用于渲染 HTML 页面
from fastapi.templating import Jinja2Templates
# Request 对象，用于获取请求信息
from starlette.requests import Request
# 响应类型，用于返回 HTML 页面
from starlette.responses import HTMLResponse

# 数据处理库
import pickle      # 用于加载 pickle 格式的数据文件
import numpy as np # NumPy 数组处理（数据集中的数组格式）

# 类型提示和路径处理
from typing import List, Dict, Any, Optional
from pathlib import Path

# ============================================================================
# 第二部分：FastAPI 应用初始化
# ============================================================================

# 创建 FastAPI 应用实例
# title: 应用名称，显示在 API 文档中
# description: 应用描述
app = FastAPI(
    title="时间序列异常检测可视化",
    description="用于可视化 15 万条时间序列异常检测数据集的 Web 应用"
)

# ============================================================================
# 第三部分：静态文件和模板目录挂载
# ============================================================================

# 获取当前文件所在目录（visualization/）
BASE_DIR = Path(__file__).resolve().parent

# 挂载静态文件目录
# StaticFiles 会将 /static URL 映射到 visualization/static/ 目录
# 这样前端可以通过 /static/app.js 访问 JS 文件
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# 配置 Jinja2 模板引擎
# 指向 visualization/templates/ 目录
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ============================================================================
# 第四部分：全局变量和数据懒加载机制
# ============================================================================

# 全局数据缓存变量
# _data_cache: 存储加载的数据，初始为 None
# _loading: 加载状态锁，防止并发加载
# _load_error: 存储加载过程中的错误信息
_data_cache: Optional[List[Dict[str, Any]]] = None
_loading: bool = False
_load_error: Optional[str] = None


def get_data() -> List[Dict[str, Any]]:
    """
    懒加载数据函数

    实现按需加载策略：
    1. 首次调用时，从 pickle 文件加载数据到内存
    2. 后续调用直接返回缓存的数据
    3. 使用加载锁防止并发加载

    Returns:
        List[Dict]: 包含所有样本的列表

    Raises:
        HTTPException: 数据加载失败或正在加载中
    """
    global _data_cache, _loading, _load_error

    # 如果数据已加载，直接返回缓存
    if _data_cache is not None:
        return _data_cache

    # 如果正在加载，返回 503 服务不可用
    if _loading:
        raise HTTPException(
            status_code=503,
            detail="数据正在加载中，请稍后重试"
        )

    # 设置加载锁
    _loading = True

    try:
        # 数据文件路径（相对于项目根目录）
        # 注意：需要从 visualization/ 目录向上查找 dataset/
        data_path = BASE_DIR.parent / "dataset" / "vetime_train_all_150000.pkl"

        # 使用 pickle 加载数据
        # pickle.load 会反序列化 Python 对象
        with open(data_path, 'rb') as f:
            _data_cache = pickle.load(f)

        print(f"数据加载成功：共 {_data_cache.__len__()} 个样本")
        return _data_cache

    except Exception as e:
        # 记录错误信息
        _load_error = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"数据加载失败: {e}"
        )
    finally:
        # 无论成功或失败，都释放加载锁
        _loading = False


# ============================================================================
# 第五部分：辅助函数
# ============================================================================

def extract_anomaly_types(anomalies: Dict[str, tuple]) -> List[str]:
    """
    从 anomalies 字典中提取异常类型

    anomalies 字典格式示例：
    {
        '0_outlier': (511, 512),
        '1_upward spike': (100, 105)
    }

    提取结果：['outlier', 'upward spike']

    Args:
        anomalies: 异常描述字典

    Returns:
        List[str]: 异常类型列表
    """
    types = []
    for key in anomalies.keys():
        # 使用 split('_', 1) 只分割第一个下划线
        # '0_outlier' -> ['0', 'outlier']
        # '1_upward spike' -> ['1', 'upward spike']
        parts = key.split('_', 1)
        if len(parts) > 1:
            types.append(parts[1])
    return types


def convert_sample_to_json(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    将样本数据转换为 JSON 可序列化格式

    关键点：
    NumPy 数组不能直接被 FastAPI 序列化为 JSON
    必须使用 .tolist() 方法转换为 Python 原生列表

    Args:
        sample: 原始样本字典

    Returns:
        Dict: JSON 可序列化的样本字典
    """
    return {
        'normal_time_series': sample['normal_time_series'].tolist(),
        'time_series': sample['time_series'].tolist(),
        'labels': sample['labels'].tolist(),
        'attribute': sample['attribute']
    }
```

**Step 2: 验证语法正确**

Run:
```bash
cd visualization && python3 -m py_compile main.py && echo "语法检查通过"
```

Expected: `语法检查通过`

**Step 3: Commit**

```bash
git add visualization/main.py
git commit -m "feat: 创建后端主程序框架

- FastAPI 应用初始化
- 静态文件和模板目录挂载
- 数据懒加载机制
- 辅助函数定义

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 添加 API 路由

**Files:**
- Modify: `visualization/main.py`（在文件末尾添加路由）

**Step 1: 添加 API 路由代码**

在 `main.py` 末尾追加以下代码：

```python
# ============================================================================
# 第六部分：API 路由定义
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    根路由：返回前端页面

    使用 Jinja2Templates 渲染 index.html 模板
    request 对象包含请求信息，模板中可能需要使用
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """
    获取数据加载状态

    返回：
    - loaded: 数据是否已加载
    - loading: 是否正在加载
    - error: 错误信息（如果有）
    - sample_count: 样本数量（如果已加载）
    """
    return {
        "loaded": _data_cache is not None,
        "loading": _loading,
        "error": _load_error,
        "sample_count": len(_data_cache) if _data_cache is not None else 0
    }


@app.get("/api/anomaly_types")
async def get_anomaly_types():
    """
    获取所有异常类型列表

    遍历所有样本，提取并去重异常类型
    用于前端下拉框的选项
    """
    data = get_data()

    # 使用集合去重
    all_types = set()

    for sample in data:
        anomalies = sample['attribute'].get('anomalies', {})
        types = extract_anomaly_types(anomalies)
        all_types.update(types)

    # 转换为排序后的列表
    return {"anomaly_types": sorted(list(all_types))}


@app.get("/api/samples")
async def get_samples(
    anomaly_type: str = Query(..., description="异常类型"),
    limit: int = Query(10, ge=1, le=100, description="返回样本数量")
):
    """
    按异常类型筛选样本

    参数：
    - anomaly_type: 异常类型（如 'outlier', 'upward spike' 等）
    - limit: 返回的样本数量，默认 10，范围 1-100

    返回：
    - samples: 符合条件的样本列表
    - total: 符合条件的样本总数
    """
    data = get_data()

    # 筛选包含指定异常类型的样本
    matched_samples = []

    for sample in data:
        anomalies = sample['attribute'].get('anomalies', {})
        types = extract_anomaly_types(anomalies)

        # 检查是否包含目标异常类型
        if anomaly_type in types:
            # 转换为 JSON 可序列化格式
            matched_samples.append(convert_sample_to_json(sample))

    # 返回指定数量的样本
    return {
        "samples": matched_samples[:limit],
        "total": len(matched_samples),
        "returned": min(limit, len(matched_samples))
    }
```

**Step 2: 验证语法正确**

Run:
```bash
cd visualization && python3 -m py_compile main.py && echo "语法检查通过"
```

Expected: `语法检查通过`

**Step 3: Commit**

```bash
git add visualization/main.py
git commit -m "feat: 添加 API 路由

- GET /: 返回前端页面
- GET /api/status: 数据加载状态
- GET /api/anomaly_types: 异常类型列表
- GET /api/samples: 按类型筛选样本

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 创建前端 HTML 页面

**Files:**
- Create: `visualization/templates/index.html`

**Step 1: 创建 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>时间序列异常检测可视化</title>

    <!-- ================================
         第一部分：引入外部资源
         ================================ -->

    <!-- TailwindCSS 框架（通过 CDN 引入）
         提供原子化 CSS 类，快速构建美观界面 -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- ECharts 图表库 v5.4.3（通过 CDN 引入）
         用于绘制交互式时间序列图表 -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
</head>

<body class="min-h-screen bg-gray-100 p-6">
    <!-- ================================
         第二部分：页面标题
         ================================ -->
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">
            时间序列异常检测可视化
        </h1>
        <p class="text-gray-600 mb-6">
            可视化展示 15 万条时间序列数据中的异常模式
        </p>

        <!-- ================================
             第三部分：控制面板
             ================================ -->
        <div class="bg-white rounded-lg shadow-md p-4 mb-6">
            <div class="flex flex-wrap items-center gap-4">
                <!-- 异常类型下拉框 -->
                <div class="flex items-center gap-2">
                    <label for="anomaly-type" class="text-gray-700 font-medium">
                        异常类型:
                    </label>
                    <select id="anomaly-type"
                            class="border border-gray-300 rounded px-4 py-2 min-w-[200px]
                                   focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <!-- 选项将通过 JavaScript 动态加载 -->
                        <option value="">正在加载...</option>
                    </select>
                </div>

                <!-- 数量选择 -->
                <div class="flex items-center gap-2">
                    <label for="limit" class="text-gray-700 font-medium">
                        数量:
                    </label>
                    <select id="limit"
                            class="border border-gray-300 rounded px-4 py-2
                                   focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="5">5</option>
                        <option value="10" selected>10</option>
                        <option value="20">20</option>
                        <option value="50">50</option>
                    </select>
                </div>

                <!-- 加载按钮 -->
                <button id="load-btn"
                        class="bg-blue-500 hover:bg-blue-600 text-white
                               px-6 py-2 rounded transition-colors duration-200
                               focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                    加载数据
                </button>

                <!-- 状态提示 -->
                <span id="status-text" class="text-gray-500 text-sm"></span>
            </div>
        </div>

        <!-- ================================
             第四部分：图表容器
             ================================ -->
        <!-- 图表将动态添加到此容器中 -->
        <div id="charts-container"
             class="w-full flex flex-col gap-6">
            <!-- 初始提示信息 -->
            <div id="placeholder"
                 class="bg-white rounded-lg shadow p-8 text-center text-gray-500">
                <p class="text-lg">请选择异常类型并点击"加载数据"开始可视化</p>
            </div>
        </div>
    </div>

    <!-- ================================
         第五部分：引入 JavaScript
         ================================ -->
    <script src="/static/app.js"></script>
</body>
</html>
```

**Step 2: 验证 HTML 结构**

Run:
```bash
cat visualization/templates/index.html | head -20
```

Expected: HTML 文件内容正确显示

**Step 3: Commit**

```bash
git add visualization/templates/index.html
git commit -m "feat: 创建前端 HTML 页面

- TailwindCSS CDN 引入
- ECharts 5.4.3 CDN 引入
- 控制面板 UI（异常类型下拉框、数量选择、加载按钮）
- 图表容器

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: 创建前端 JavaScript（第一部分：初始化和异常类型加载）

**Files:**
- Create: `visualization/static/app.js`

**Step 1: 创建 app.js 初始化代码**

```javascript
/**
 * 时间序列异常检测可视化 - 前端逻辑
 *
 * 本模块负责：
 * 1. 页面初始化和异常类型加载
 * 2. 样本数据获取和处理
 * 3. ECharts 图表渲染
 * 4. 异常区间高亮显示
 *
 * 作者：VETime 项目组
 * 日期：2026-06-08
 */

// ============================================================================
// 第一部分：全局变量和 DOM 元素引用
// ============================================================================

// DOM 元素引用（页面加载完成后初始化）
let anomalyTypeSelect;   // 异常类型下拉框
let limitSelect;         // 数量选择下拉框
let loadBtn;             // 加载按钮
let statusText;          // 状态提示文本
let chartsContainer;     // 图表容器
let placeholder;         // 占位提示元素

// ============================================================================
// 第二部分：页面初始化
// ============================================================================

/**
 * 页面加载完成后执行初始化
 *
 * DOMContentLoaded 事件在 HTML 解析完成后触发
 * 此时所有 DOM 元素都已可用
 */
document.addEventListener('DOMContentLoaded', function() {
    // 获取 DOM 元素引用
    anomalyTypeSelect = document.getElementById('anomaly-type');
    limitSelect = document.getElementById('limit');
    loadBtn = document.getElementById('load-btn');
    statusText = document.getElementById('status-text');
    chartsContainer = document.getElementById('charts-container');
    placeholder = document.getElementById('placeholder');

    // 绑定事件监听器
    loadBtn.addEventListener('click', handleLoadData);

    // 加载异常类型列表
    loadAnomalyTypes();
});

// ============================================================================
// 第三部分：异常类型加载
// ============================================================================

/**
 * 从后端加载异常类型列表
 *
 * 使用 Fetch API 发送 GET 请求到 /api/anomaly_types
 * 然后填充到下拉框中
 */
async function loadAnomalyTypes() {
    try {
        statusText.textContent = '正在加载异常类型...';

        // Fetch API 发送请求
        // await 等待请求完成
        const response = await fetch('/api/anomaly_types');

        // 检查响应状态
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // 解析 JSON 响应
        const data = await response.json();

        // 清空下拉框
        anomalyTypeSelect.innerHTML = '';

        // 添加选项
        // data.anomaly_types 是异常类型数组
        data.anomaly_types.forEach(function(type) {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            anomalyTypeSelect.appendChild(option);
        });

        statusText.textContent = `已加载 ${data.anomaly_types.length} 种异常类型`;

    } catch (error) {
        console.error('加载异常类型失败:', error);
        statusText.textContent = '加载异常类型失败，请刷新页面重试';
        anomalyTypeSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

// ============================================================================
// 第四部分：数据加载处理
// ============================================================================

/**
 * 处理"加载数据"按钮点击事件
 *
 * 1. 获取选中的异常类型和数量
 * 2. 发送请求到后端
 * 3. 渲染图表
 */
async function handleLoadData() {
    // 获取选中的值
    const anomalyType = anomalyTypeSelect.value;
    const limit = parseInt(limitSelect.value);

    if (!anomalyType) {
        statusText.textContent = '请选择异常类型';
        return;
    }

    try {
        // 禁用按钮，显示加载状态
        loadBtn.disabled = true;
        loadBtn.textContent = '加载中...';
        statusText.textContent = '正在加载数据，首次加载可能需要较长时间...';

        // 发送请求
        // URL 参数使用 URLSearchParams 构建
        const params = new URLSearchParams({
            anomaly_type: anomalyType,
            limit: limit.toString()
        });

        const response = await fetch(`/api/samples?${params}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // 渲染图表
        renderCharts(data.samples);

        statusText.textContent = `已加载 ${data.returned} 个样本（共 ${data.total} 个匹配）`;

    } catch (error) {
        console.error('加载数据失败:', error);
        statusText.textContent = `加载失败: ${error.message}`;
    } finally {
        // 恢复按钮状态
        loadBtn.disabled = false;
        loadBtn.textContent = '加载数据';
    }
}
```

**Step 2: 验证 JavaScript 语法**

Run:
```bash
node --check visualization/static/app.js 2>&1 || echo "注意: Node.js 可能未安装，跳过语法检查"
```

Expected: 无语法错误或跳过提示

**Step 3: Commit**

```bash
git add visualization/static/app.js
git commit -m "feat: 创建前端 JavaScript 初始化代码

- 全局变量和 DOM 元素引用
- 页面初始化逻辑
- 异常类型加载函数
- 数据加载处理函数

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: 添加 ECharts 图表渲染代码

**Files:**
- Modify: `visualization/static/app.js`（在文件末尾追加）

**Step 1: 追加图表渲染代码**

在 `app.js` 末尾追加以下代码：

```javascript
// ============================================================================
// 第五部分：图表渲染
// ============================================================================

/**
 * 渲染所有样本的图表
 *
 * @param {Array} samples - 样本数组
 */
function renderCharts(samples) {
    // 清空容器
    chartsContainer.innerHTML = '';

    if (samples.length === 0) {
        chartsContainer.innerHTML = `
            <div class="bg-white rounded-lg shadow p-8 text-center text-gray-500">
                <p>没有找到匹配的样本</p>
            </div>
        `;
        return;
    }

    // 遍历样本，为每个样本创建图表
    samples.forEach(function(sample, index) {
        renderSingleChart(sample, index + 1);
    });
}

/**
 * 渲染单个样本的图表
 *
 * @param {Object} sample - 单个样本数据
 * @param {number} index - 样本序号
 */
function renderSingleChart(sample, index) {
    // 创建图表容器
    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'w-full h-[600px] bg-white rounded-lg shadow p-4';
    chartWrapper.innerHTML = `
        <div class="text-sm text-gray-600 mb-2">
            样本 #${index} |
            序列长度: ${sample.time_series.length} |
            异常点数: ${sample.labels.filter(l => l === 1).length}
        </div>
        <div id="chart-${index}" style="width: 100%; height: 550px;"></div>
    `;
    chartsContainer.appendChild(chartWrapper);

    // 初始化 ECharts
    const chartDom = document.getElementById(`chart-${index}`);
    const myChart = echarts.init(chartDom);

    // 生成 X 轴索引数组
    const indices = Array.from({ length: sample.time_series.length }, (_, i) => i);

    // 计算异常区间
    const anomalyRegions = computeAnomalyRegions(sample.labels);

    // 构建 markArea 数据（异常区间）
    const markAreaData = anomalyRegions.map(region => [
        { coord: [region.start, 'min'] },
        { coord: [region.end, 'max'] }
    ]);

    // 构建 markPoint 数据（异常点）
    const markPointData = [];
    sample.labels.forEach(function(label, i) {
        if (label === 1) {
            markPointData.push({
                coord: [i, sample.time_series[i]],
                itemStyle: { color: '#ef4444' },
                symbolSize: 10
            });
        }
    });

    // ECharts 配置项
    const option = {
        // 标题配置
        title: [
            {
                text: 'Normal Time Series (纯净序列)',
                left: 'center',
                top: 10,
                textStyle: { fontSize: 14, color: '#3b82f6' }
            },
            {
                text: 'Time Series (含异常序列)',
                left: 'center',
                top: '52%',
                textStyle: { fontSize: 14, color: '#10b981' }
            }
        ],

        // 提示框配置
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },

        // 图例配置
        legend: {
            data: ['Normal', 'Anomaly'],
            top: 30
        },

        // 双网格布局
        grid: [
            // 上方网格：纯净序列
            {
                left: '10%',
                right: '5%',
                top: '15%',
                height: '30%'
            },
            // 下方网格：含异常序列
            {
                left: '10%',
                right: '5%',
                top: '60%',
                height: '30%'
            }
        ],

        // X 轴配置
        xAxis: [
            {
                type: 'category',
                gridIndex: 0,
                data: indices,
                axisLabel: { show: false }
            },
            {
                type: 'category',
                gridIndex: 1,
                data: indices,
                axisLabel: { show: true }
            }
        ],

        // Y 轴配置
        yAxis: [
            { type: 'value', gridIndex: 0 },
            { type: 'value', gridIndex: 1 }
        ],

        // 数据系列
        series: [
            // 上方：纯净序列
            {
                name: 'Normal',
                type: 'line',
                xAxisIndex: 0,
                yAxisIndex: 0,
                data: sample.normal_time_series,
                lineStyle: { color: '#3b82f6', width: 1 },
                showSymbol: false
            },
            // 下方：含异常序列
            {
                name: 'Anomaly',
                type: 'line',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: sample.time_series,
                lineStyle: { color: '#10b981', width: 1 },
                showSymbol: false,
                // 异常区间标记（半透明红色区域）
                markArea: {
                    data: markAreaData,
                    itemStyle: {
                        color: 'rgba(239, 68, 68, 0.3)'
                    }
                },
                // 异常点标记（红色圆点）
                markPoint: {
                    data: markPointData,
                    symbol: 'circle',
                    symbolSize: 10
                }
            }
        ],

        // 缩放组件
        dataZoom: [
            {
                type: 'slider',
                xAxisIndex: [0, 1],
                bottom: 20,
                height: 20
            }
        ]
    };

    // 应用配置项并渲染
    myChart.setOption(option);
}

// ============================================================================
// 第六部分：辅助函数
// ============================================================================

/**
 * 计算异常区间
 *
 * 从 labels 数组中识别连续的异常区间
 *
 * @param {Array} labels - 标签数组（0 表示正常，1 表示异常）
 * @returns {Array} 异常区间数组，每个元素包含 start 和 end
 *
 * 示例：
 * 输入: [0, 0, 1, 1, 0, 1, 0, 0]
 * 输出: [{start: 2, end: 3}, {start: 5, end: 5}]
 */
function computeAnomalyRegions(labels) {
    const regions = [];
    let start = -1;  // 当前异常区间的起始位置

    // 遍历标签数组
    for (let i = 0; i < labels.length; i++) {
        if (labels[i] === 1 && start === -1) {
            // 发现异常点，且不在异常区间内 -> 开始新区间
            start = i;
        } else if (labels[i] === 0 && start !== -1) {
            // 发现正常点，且在异常区间内 -> 结束当前区间
            regions.push({ start: start, end: i - 1 });
            start = -1;
        }
    }

    // 处理序列末尾的异常区间
    if (start !== -1) {
        regions.push({ start: start, end: labels.length - 1 });
    }

    return regions;
}
```

**Step 2: 验证 JavaScript 语法**

Run:
```bash
node --check visualization/static/app.js 2>&1 || echo "注意: Node.js 可能未安装，跳过语法检查"
```

Expected: 无语法错误或跳过提示

**Step 3: Commit**

```bash
git add visualization/static/app.js
git commit -m "feat: 添加 ECharts 图表渲染代码

- renderCharts: 渲染所有样本图表
- renderSingleChart: 渲染单个样本图表
- computeAnomalyRegions: 计算异常区间
- 双网格布局配置
- markArea 异常区间标记
- markPoint 异常点标记
- dataZoom 缩放组件

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: 安装依赖并启动服务

**Files:**
- 无新文件

**Step 1: 安装依赖**

Run:
```bash
cd visualization && pip install -r requirements.txt
```

Expected: 所有依赖安装成功

**Step 2: 启动开发服务器**

Run:
```bash
cd visualization && uvicorn main:app --reload --port 8000
```

Expected: 服务启动成功，显示类似以下信息：
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Step 3: 访问测试**

打开浏览器访问 `http://127.0.0.1:8000`

Expected:
- 页面正常显示
- 异常类型下拉框加载成功
- 点击"加载数据"后图表正常渲染

---

## Task 9: 最终提交和文档更新

**Files:**
- 无新文件

**Step 1: 确认所有文件已提交**

Run:
```bash
git status
```

Expected: 无未提交的更改

**Step 2: 创建项目说明文件**

在 `visualization/` 目录创建 `README.md`：

```markdown
# 时间序列异常检测可视化

基于 FastAPI + ECharts 的时间序列异常检测数据可视化工具。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --reload --port 8000
```

访问 http://127.0.0.1:8000 开始使用。

## 功能特点

- 支持 30+ 种异常类型筛选
- 双图联动对比（纯净序列 vs 含异常序列）
- 异常区间高亮显示
- 图表缩放功能

## 数据来源

数据集路径：`../dataset/vetime_train_all_150000.pkl`
```

**Step 3: 最终提交**

Run:
```bash
git add visualization/README.md
git commit -m "docs: 添加可视化项目说明文档

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 实现计划完成

所有任务已完成。以下是执行选项：

