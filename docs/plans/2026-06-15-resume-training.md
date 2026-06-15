# Resume Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为多变量训练添加checkpoint恢复和预训练权重加载功能，支持从OOM中断恢复和使用单变量权重启动多变量训练。

**Architecture:** 扩展现有train.py中的train_multivariate函数，新增三个辅助函数处理checkpoint的完整保存和加载。保持与现有checkpoint机制的兼容性。

**Tech Stack:** PyTorch, PyYAML, Accelerate

---

## Task 1: 添加命令行参数和配置支持

**Files:**
- Modify: `train.py:896-921` (argparse部分)

**Step 1: 添加新的命令行参数**

在 `train.py` 的 argparse 部分添加两个新参数：

```python
parser.add_argument('--resume', type=str, default=None,
                    help='从checkpoint继续训练的路径（完整状态恢复）')
parser.add_argument('--pretrain_from', type=str, default=None,
                    help='预训练权重路径（仅模型权重），用于启动多变量训练')
```

位置：在 `--ts_finetune_type` 参数之后添加（约第920行）

**Step 2: 验证参数添加成功**

```bash
python train.py --help | grep -E "(resume|pretrain_from)"
```

预期输出：
```
  --resume RESUME       从checkpoint继续训练的路径（完整状态恢复）
  --pretrain_from PRETRAIN_FROM
                        预训练权重路径（仅模型权重），用于启动多变量训练
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: add --resume and --pretrain_from arguments"
```

---

## Task 2: 更新配置文件格式

**Files:**
- Modify: `config/multivariate_config.yaml:61-64`

**Step 1: 添加配置字段**

在 `multivariate_config.yaml` 的 `multivariate` 部分末尾添加：

```yaml
  # 继续训练配置
  resume_from: null       # 从checkpoint恢复的路径
  pretrain_from: null     # 预训练权重路径（如单变量权重）
```

**Step 2: 验证YAML格式正确**

```bash
python -c "import yaml; yaml.safe_load(open('config/multivariate_config.yaml'))"
```

预期：无输出（表示解析成功）

**Step 3: 提交**

```bash
git add config/multivariate_config.yaml
git commit -m "feat: add resume_from and pretrain_from config options"
```

---

## Task 3: 实现save_full_checkpoint函数

**Files:**
- Modify: `train.py` (在train_multivariate函数之前添加)

**Step 1: 添加save_full_checkpoint函数**

在 `train_multivariate` 函数定义之前（约第500行附近），添加：

```python
def save_full_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    dataset_idx: int,
    current_dim: int,
    prev_checkpoint_path: Optional[str],
    best_val_loss: float,
    patience_counter: int,
    save_path: str,
    accelerator
):
    """
    保存完整的训练状态checkpoint

    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前epoch（已完成）
        global_step: 全局步数
        dataset_idx: 当前数据集索引
        current_dim: 当前维度
        prev_checkpoint_path: 上一维度的checkpoint路径
        best_val_loss: 最佳验证损失
        patience_counter: 早停计数器
        save_path: 保存路径
        accelerator: Accelerator实例
    """
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    checkpoint = {
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'dataset_idx': dataset_idx,
        'current_dim': current_dim,
        'prev_checkpoint_path': prev_checkpoint_path,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'random_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }

    if accelerator.is_main_process:
        torch.save(checkpoint, save_path)
        print(f"[INFO] 完整Checkpoint已保存: {save_path}")
```

**Step 2: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: add save_full_checkpoint function"
```

---

## Task 4: 实现load_full_checkpoint函数

**Files:**
- Modify: `train.py` (在save_full_checkpoint之后添加)

**Step 1: 添加load_full_checkpoint函数**

```python
def load_full_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    accelerator
) -> dict:
    """
    加载完整的训练状态checkpoint

    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型实例
        optimizer: 优化器实例
        accelerator: Accelerator实例

    Returns:
        包含恢复状态的字典: {
            'start_epoch': int,
            'global_step': int,
            'dataset_idx': int,
            'current_dim': int,
            'prev_checkpoint_path': str,
            'best_val_loss': float,
            'patience_counter': int
        }
    """
    print(f"[INFO] 正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检查是否为完整checkpoint
    if 'model_state_dict' not in checkpoint:
        # 兼容旧格式（仅模型权重）
        print("[INFO] 检测到旧格式checkpoint（仅模型权重），将作为预训练权重加载")
        unwrapped_model = accelerator.unwrap_model(model)
        missing, unexpected = unwrapped_model.load_state_dict(checkpoint, strict=False)
        print(f"[INFO] 模型权重加载完成 (strict=False)")
        return None

    # 加载模型权重
    unwrapped_model = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"[INFO] 模型权重加载完成")
    if missing:
        print(f"  缺失的参数: {len(missing)} 个")
    if unexpected:
        print(f"  未预期的参数: {len(unexpected)} 个")

    # 加载optimizer状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"[INFO] Optimizer状态已恢复")

    # 恢复随机状态
    random_state = checkpoint.get('random_state', {})
    if 'python' in random_state:
        random.setstate(random_state['python'])
    if 'numpy' in random_state:
        np.random.set_state(random_state['numpy'])
    if 'torch' in random_state:
        torch.set_rng_state(random_state['torch'])
    if 'cuda' in random_state and random_state['cuda'] is not None:
        torch.cuda.set_rng_state_all(random_state['cuda'])
    print(f"[INFO] 随机状态已恢复")

    return {
        'start_epoch': checkpoint['epoch'] + 1,  # 从下一个epoch开始
        'global_step': checkpoint['global_step'],
        'dataset_idx': checkpoint['dataset_idx'],
        'current_dim': checkpoint['current_dim'],
        'prev_checkpoint_path': checkpoint.get('prev_checkpoint_path'),
        'best_val_loss': checkpoint['best_val_loss'],
        'patience_counter': checkpoint['patience_counter']
    }
```

**Step 2: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: add load_full_checkpoint function with legacy support"
```

---

## Task 5: 实现load_pretrain_weights函数

**Files:**
- Modify: `train.py` (在load_full_checkpoint之后添加)

**Step 1: 添加load_pretrain_weights函数**

```python
def load_pretrain_weights(pretrain_path: str, model, accelerator):
    """
    加载预训练权重（仅模型权重，用于启动新训练）

    Args:
        pretrain_path: 预训练权重路径
        model: 模型实例
        accelerator: Accelerator实例
    """
    print(f"[INFO] 正在加载预训练权重: {pretrain_path}")
    state_dict = torch.load(pretrain_path, map_location='cpu')

    # 处理可能的格式差异
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    unwrapped_model = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] 预训练权重加载完成 (strict=False)")
    if missing:
        print(f"  缺失的参数（新维度组件，使用初始化值）: {len(missing)} 个")
    if unexpected:
        print(f"  未预期的参数: {len(unexpected)} 个")
```

**Step 2: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: add load_pretrain_weights function"
```

---

## Task 6: 修改train_multivariate - 解析resume和pretrain参数

**Files:**
- Modify: `train.py:518-587` (train_multivariate函数开头部分)

**Step 1: 在train_multivariate开头添加参数解析逻辑**

在 `mv_config = config['multivariate']` 之后，添加：

```python
    # 解析 resume 和 pretrain 参数（命令行优先于配置文件）
    resume_path = args.resume or mv_config.get('resume_from')
    pretrain_path = args.pretrain_from or mv_config.get('pretrain_from')

    if resume_path and pretrain_path:
        print("[WARNING] 同时指定了 --resume 和 --pretrain_from，将使用 --resume")
        pretrain_path = None

    if resume_path:
        print(f"[INFO] 继续训练模式: 从 {resume_path} 恢复")
    elif pretrain_path:
        print(f"[INFO] 预训练模式: 使用 {pretrain_path} 初始化")
```

**Step 2: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: parse resume and pretrain arguments in train_multivariate"
```

---

## Task 7: 修改train_multivariate - 处理resume恢复

**Files:**
- Modify: `train.py:591-630` (维度循环开始部分)

**Step 1: 在维度循环开始前添加resume恢复逻辑**

在 `# ========== 外层维度循环 ==========` 之前添加：

```python
    # ========== 处理resume恢复 ==========
    resume_state = None
    start_dataset_idx = 0

    if resume_path:
        # 先创建一个临时模型用于加载checkpoint检查维度
        temp_model, _ = create_model(args, datasets[0]['dim'], vision_model, config_v)
        temp_optimizer = torch.optim.AdamW([p for p in temp_model.parameters() if p.requires_grad])

        resume_state = load_full_checkpoint(resume_path, temp_model, temp_optimizer, accelerator)

        if resume_state is not None:
            start_dataset_idx = resume_state['dataset_idx']
            print(f"[INFO] 将从数据集索引 {start_dataset_idx} (维度 {resume_state['current_dim']}) 继续")

        del temp_model, temp_optimizer
        torch.cuda.empty_cache()
        gc.collect()
```

**Step 2: 修改维度循环的起始逻辑**

将维度循环修改为支持跳过已完成的维度：

```python
    # ========== 外层维度循环 ==========
    for dataset_idx, dataset_info in enumerate(datasets):
        # 跳过已完成的维度（resume场景）
        if resume_state is not None and dataset_idx < start_dataset_idx:
            print(f"[INFO] 跳过已完成的数据集 {dataset_idx+1}/{len(datasets)}: 维度 {dataset_info['dim']}")
            continue
```

**Step 3: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 4: 提交**

```bash
git add train.py
git commit -m "feat: add resume state restoration logic in train_multivariate"
```

---

## Task 8: 修改train_multivariate - 处理pretrain加载

**Files:**
- Modify: `train.py:614-630` (模型创建和权重加载部分)

**Step 1: 在模型创建后添加pretrain加载逻辑**

在 `model, ts_model = create_model(...)` 之后，权重加载逻辑之前添加：

```python
        # ========== 处理pretrain加载（仅第一个维度）==========
        if pretrain_path is not None and dataset_idx == 0:
            load_pretrain_weights(pretrain_path, model, accelerator)
            print(f"[INFO] 预训练权重已加载，从头开始多变量训练")
```

**Step 2: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: add pretrain weight loading in train_multivariate"
```

---

## Task 9: 修改train_multivariate - 恢复训练状态并保存epoch checkpoint

**Files:**
- Modify: `train.py:697-835` (epoch训练循环部分)

**Step 1: 恢复epoch和早停状态**

在 `for epoch in range(epochs):` 之前添加：

```python
        # ========== 恢复训练状态（resume场景）==========
        start_epoch = 0
        if resume_state is not None and resume_state['dataset_idx'] == dataset_idx:
            start_epoch = resume_state['start_epoch']
            best_val_loss = resume_state['best_val_loss']
            patience_counter = resume_state['patience_counter']
            global_step = resume_state['global_step']
            print(f"[INFO] 从epoch {start_epoch} 继续训练，best_val_loss={best_val_loss:.4f}")

        for epoch in range(start_epoch, epochs):
```

**Step 2: 在每个epoch结束后保存完整checkpoint**

在验证完成后（保存best model之后）添加：

```python
            # ========== 保存完整checkpoint（每个epoch）==========
            epoch_checkpoint_path = os.path.join(
                checkpoint_dir,
                f'vetime_dim{current_dim}_epoch{epoch}_full.pth'
            )
            save_full_checkpoint(
                model, optimizer, epoch, global_step,
                dataset_idx, current_dim, prev_checkpoint_path,
                best_val_loss, patience_counter,
                epoch_checkpoint_path, accelerator
            )
```

**Step 3: 验证语法正确**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 4: 提交**

```bash
git add train.py
git commit -m "feat: add epoch checkpoint saving and state restoration"
```

---

## Task 10: 清理和测试

**Files:**
- Modify: `train.py`

**Step 1: 清理prev_checkpoint_path的resume处理**

确保在维度完成后正确更新 `prev_checkpoint_path`，在resume场景下也需要正确处理。

检查 `prev_checkpoint_path = checkpoint_path` 这一行（约第846行），确保逻辑正确。

**Step 2: 完整语法检查**

```bash
python -m py_compile train.py && echo "Syntax OK"
```

**Step 3: 测试help输出**

```bash
python train.py --help | head -50
```

**Step 4: 最终提交**

```bash
git add train.py
git commit -m "feat: complete resume training implementation"
```

---

## 完成验证清单

- [ ] `python train.py --help` 显示新参数
- [ ] `python -m py_compile train.py` 无语法错误
- [ ] 正常训练流程不受影响
- [ ] `--resume` 可从epoch checkpoint恢复
- [ ] `--pretrain_from` 可加载预训练权重

## 测试命令示例

```bash
# 测试正常训练（确保不影响现有功能）
python train.py --config config/multivariate_config.yaml --num_epochs 1

# 测试resume功能（先创建一个fake checkpoint）
python train.py --config config/multivariate_config.yaml \
    --resume ./output/multivariate_checkpoints/vetime_dim2_epoch0_full.pth

# 测试pretrain功能
python train.py --config config/multivariate_config.yaml \
    --pretrain_from ./checkpoints/weight_ts/pretrain_checkpoint_best_uni.pth
```
