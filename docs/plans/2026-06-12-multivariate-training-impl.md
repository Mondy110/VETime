# VETime 多变量顺序训练实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 VETime 项目添加多变量顺序训练功能，支持按维度顺序训练多变量时间序列数据集。

**Architecture:** 采用 TimeRCD 的"销毁重建 + strict=False"范式。在 train.py 中新增 train_multivariate() 函数，复用现有 Vision Encoder，通过配置文件控制训练模式。

**Tech Stack:** PyTorch, HuggingFace Accelerate, YAML 配置

---

## Task 1: 创建配置文件目录和示例配置

**Files:**
- Create: `config/multivariate_config.yaml`

**Step 1: 创建配置目录**

```bash
mkdir -p /mnt/sda/cjmProject/VETime/config
```

**Step 2: 创建多变量训练配置文件**

```yaml
# config/multivariate_config.yaml

# 训练模式: univariate(单变量) 或 multivariate(多变量顺序训练)
training_mode: multivariate

# 多变量顺序训练配置
multivariate:
  # 数据集列表，按训练顺序排列
  datasets:
    - path: ./dataset/dim_1.pkl
      dim: 1
    - path: ./dataset/dim_38.pkl
      dim: 38
    - path: ./dataset/dim_55.pkl
      dim: 55

  # 维度到批次大小的映射
  dim_batch_size_map:
    "1": 32
    "38": 8
    "55": 4

  # checkpoint 保存目录
  checkpoint_dir: ./output/multivariate_checkpoints

  # 是否严格按维度递增顺序检查
  enforce_dim_order: false

# 单变量训练配置（兼容原有方式）
univariate:
  dataset_path: ./dataset/univariate.pkl
  batch_size: 32
```

**Step 3: 验证配置文件创建成功**

```bash
cat /mnt/sda/cjmProject/VETime/config/multivariate_config.yaml
```

**Step 4: Commit**

```bash
git add config/multivariate_config.yaml
git commit -m "feat: add multivariate training config file"
```

---

## Task 2: 在 train.py 中添加 YAML 导入和配置加载函数

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 在文件顶部添加 yaml 导入**

在 `import` 区域添加：

```python
import yaml
from typing import Dict, Any
```

**Step 2: 添加配置加载函数**

在 `seed_worker` 函数后添加：

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
```

**Step 3: 验证语法正确**

```bash
python -c "import yaml; print('yaml imported successfully')"
```

**Step 4: Commit**

```bash
git add train.py
git commit -m "feat: add yaml import and load_config function"
```

---

## Task 3: 添加 --config 命令行参数

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 在 argparse 区域添加 config 参数**

在 `if __name__ == "__main__":` 块的参数定义区域添加：

```python
    parser.add_argument('--config', type=str, default=None,
                        help='训练配置文件路径（YAML格式），指定后将忽略部分命令行参数')
```

**Step 2: 验证参数添加成功**

```bash
python train.py --help | grep -A2 "config"
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add --config command line argument"
```

---

## Task 4: 创建 create_model 辅助函数

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 在 seed_worker 函数后添加 create_model 函数**

```python
def create_model(args, num_features: int, vision_model, config_v):
    """
    创建 VETime 模型（复用已存在的 vision_model）

    根据 num_features 自动构建对应维度的组件：
    - num_features=1: 不创建 BinaryAttentionBias
    - num_features>1: 创建 BinaryAttentionBias

    Args:
        args: 命令行参数
        num_features: 当前数据集的特征维度
        vision_model: 已实例化的视觉编码器（复用，不重建）
        config_v: 视觉模型配置

    Returns:
        model: VETIME 模型
        ts_model: 时间序列编码器
    """
    # 更新 TS Encoder 配置
    default_config_t.num_features = num_features

    # 仅实例化 TS Encoder（根据 num_features 自动构建 BinaryAttentionBias）
    ts_model = TS_Model(default_config_t)

    # 构建 VETime 完整模型
    model = VETIME(config_v, vision_model, default_config_t, ts_model, args.model_name)

    return model, ts_model
```

**Step 2: 验证语法正确**

```bash
python -c "from train import create_model; print('create_model function loaded')"
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add create_model helper function for multivariate training"
```

---

## Task 5: 创建 train_univariate 函数（封装原有逻辑）

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 将原有 main() 函数重命名为 train_univariate()**

找到 `def main(args):` 这一行，将其重命名为：

```python
def train_univariate(args):
    """
    单变量训练：完全保留原有训练逻辑

    此函数与原 main() 完全相同，确保单变量训练路径不受任何影响。
    """
    # ... 原有代码保持不变 ...
```

**Step 2: 在函数末尾添加 return 语句**

确保函数有明确的返回值（原有代码已有 `return output`）。

**Step 3: 验证语法正确**

```bash
python -c "from train import train_univariate; print('train_univariate loaded')"
```

**Step 4: Commit**

```bash
git add train.py
git commit -m "refactor: rename main to train_univariate for mode separation"
```

---

## Task 6: 创建 train_multivariate 函数骨架

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 在 train_univariate 函数后添加 train_multivariate 函数骨架**

```python
def train_multivariate(args, config: Dict[str, Any]):
    """
    多变量顺序训练（参考 TimeRCD 范式）

    核心策略：
    1. 销毁重建：每次切换维度时全新实例化模型
    2. 知识延续：strict=False 加载 checkpoint
    3. 参数纳管：重建 Optimizer 确保所有参数可训练
    4. 显存安全：Accelerator 循环外初始化，Vision Encoder 单次实例化

    Args:
        args: 命令行参数
        config: YAML 配置字典
    """
    mv_config = config['multivariate']
    datasets = mv_config['datasets']
    dim_batch_size_map = mv_config['dim_batch_size_map']
    checkpoint_dir = mv_config['checkpoint_dir']
    enforce_dim_order = mv_config.get('enforce_dim_order', False)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)
    print(f"[INFO] 随机种子已设置: {args.seed}")

    # ========== [关键] Accelerator 全局初始化（只执行一次）==========
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        project_dir="./output/logs"
    )
    device = accelerator.device

    logger.info(f"Using {accelerator.num_processes} {'GPUs' if accelerator.num_processes > 1 else 'CPU'}")

    # ========== [关键] Vision Encoder 只实例化一次 ==========
    print(f"[INFO] 正在加载 Vision Encoder (MAE) 权重: checkpoints/weight_v/{args.vision_name}")
    vision_model = V_model(
        args.vision_name,
        MAX_L=5000,
        unpatch=True,
        finetune_type='none'
    )
    # 移至正确的设备并开启 eval 模式
    vision_model = vision_model.to(device)
    vision_model.eval()

    # 【防御性代码】显式冻结所有参数，防止意外梯度计算
    for param in vision_model.parameters():
        param.requires_grad = False

    config_v = vision_model.config
    if 'mae' in args.vision_name:
        patch_size = config_v['patch_size']
    else:
        patch_size = config_v.patch_size
    args.patch_size = patch_size

    print(f"[INFO] Vision Encoder 权重加载完成！Patch Size: {patch_size}")
    print(f"[INFO] Vision Encoder 状态: 完全冻结 (作为特征对齐锚点)")

    # 检查维度顺序（可选）
    if enforce_dim_order:
        dims = [ds['dim'] for ds in datasets]
        if dims != sorted(dims):
            raise ValueError(f"数据集维度顺序非递增: {dims}，请调整 datasets 列表顺序或设置 enforce_dim_order: false")

    # 设置 LoRA 配置
    if args.ts_finetune_type == 'lora':
        default_config_t.use_lora = True
        print(f"[INFO] TS Encoder 微调类型: LoRA (r={default_config_t.lora_r}, α={default_config_t.lora_alpha})")
    else:
        default_config_t.use_lora = False
        print(f"[INFO] TS Encoder 微调类型: 完全冻结")

    # 全局 checkpoint 路径（用于维度间传递）
    prev_checkpoint_path = None

    # 数据设置
    data_setting = args.data_setting

    # ========== 外层维度循环 ==========
    for dataset_idx, dataset_info in enumerate(datasets):
        dataset_path = dataset_info['path']
        current_dim = dataset_info['dim']
        batch_size = dim_batch_size_map.get(str(current_dim), 4)

        print(f"\n{'='*60}")
        print(f"[多变量训练] 数据集 {dataset_idx+1}/{len(datasets)}: {dataset_path}")
        print(f"  维度: {current_dim}, Batch Size: {batch_size}")
        if prev_checkpoint_path is not None:
            print(f"  继承上一维度权重: {prev_checkpoint_path}")
        print(f"{'='*60}\n")

        # ========== 1. 刷新配置 ==========
        default_config_t.num_features = current_dim
        args.batch_size = batch_size

        # ========== 2. 全新实例化模型（复用 vision_model）==========
        model, ts_model = create_model(args, current_dim, vision_model, config_v)

        # ========== 3. 知识延续（strict=False 是灵魂）==========
        if prev_checkpoint_path is not None:
            print(f"[INFO] 加载上一维度 checkpoint: {prev_checkpoint_path}")
            state_dict = torch.load(prev_checkpoint_path, map_location='cpu')

            # 【防御性代码】剥离 DDP 的 'module.' 前缀，适配不同保存格式
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] 权重加载完成 (strict=False)")
            if missing:
                print(f"  缺失的参数（新维度组件，使用初始化值）: {len(missing)} 个")
            if unexpected:
                print(f"  未预期的参数: {len(unexpected)} 个")

        # 打印参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[INFO] 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # ========== 4. 重建 Optimizer ==========
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print(f"\n[INFO] 优化器配置: AdamW, lr={args.learning_rate}, weight_decay={args.weight_decay}")

        # ========== 5. 创建 DataLoader ==========
        train_dataset = AnomalyDataset(dataset_path, patch_size=patch_size, split="train")

        g = torch.Generator()
        g.manual_seed(args.seed)

        collatefn = partial(collate_fn, patch_size=patch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collatefn,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g
        )

        # ========== 6. prepare 动态组件 ==========
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        # ========== 7. 设置训练模式（注意：vision_model 需保持 eval）==========
        model.train()
        vision_model.eval()

        # ========== 8. 内层 Epoch 训练 ==========
        global_step = 0
        epochs = args.num_epochs
        img_size = data_setting['img_size']

        early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True,
                                        path=os.path.join(checkpoint_dir, f'temp_dim{current_dim}_best.pth'))

        for epoch in range(epochs):
            model.train()
            vision_model.eval()  # 确保每个 epoch 开始时 vision_model 保持 eval

            total_loss = 0
            all_probs, all_preds, all_labels = [], [], []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}[Train]",
                               disable=not accelerator.is_local_main_process)

            for batch in progress_bar:
                labels = batch["labels"]
                images = batch["image"]
                time_series, att_mask = batch['time_series'], batch['attention_mask']
                mask = batch['mask']
                period = batch['period']
                p_value = batch['padding_value']

                if labels.shape[1] > model.MAX_L:
                    data_splits = model.split_data(images, time_series, att_mask, labels)
                    loss1 = 0
                    loss2 = 0
                    logits_list = []

                    for data_part in data_splits:
                        img_part, ts_part, att_mask_part, label_part = data_part
                        images_folded, init_img_size = model.vit_encoder.fold_image(
                            img_part, period, p_value, **data_setting)

                        local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                            images_folded, ts_part, att_mask_part, init_img_size, label_part)

                        loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)
                        loss02, rec = model.weighted_reconstruction_loss(
                            local_embeddings2, ts_part, att_mask_part, label_part)

                        loss2 = loss2 + loss02 + 0.1 * loss_cl + 0.2 * load_balance_loss(m_w)
                        loss1 = loss1 + loss01
                        logits_list.append(logit)

                    logits = torch.cat(logits_list, dim=1)
                else:
                    images_folded, init_img_size = model.vit_encoder.fold_image(
                        images, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                        images_folded, time_series, att_mask, init_img_size, labels)

                    loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)
                    loss2, rec = model.weighted_reconstruction_loss(
                        local_embeddings2, time_series, att_mask, labels)
                    loss2 = loss2 + 0.2 * load_balance_loss(m_w) + 0.1 * loss_cl

                accelerator.backward(loss1 + loss2)

                global_step += 1
                if global_step % accelerator.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                batch_loss = loss1.item() + loss2.item()
                total_loss += batch_loss
                progress_bar.set_postfix({"loss": batch_loss})

                probs = torch.softmax(logits, dim=-1)[:, :, 1]
                preds = (probs > 0.5).float()
                probs, preds, labels = accelerator.gather_for_metrics((probs, preds, labels))

                if global_step % 10 == 0:
                    for i in range(probs.shape[0]):
                        all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                        all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                        all_labels.append(labels[i].detach().cpu().numpy().reshape(-1).astype(int))

                del images_folded, logits, loss1, probs, preds, labels, loss2
                del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, mask, period, p_value
                del images, time_series, att_mask, init_img_size
                torch.cuda.empty_cache()

            # 计算 epoch 指标
            if len(all_probs) > 0:
                all_probs_arr = np.concatenate(all_probs)
                all_preds_arr = np.concatenate(all_preds)
                all_labels_arr = np.concatenate(all_labels)
                train_metrics = fast_get_metrics(all_probs_arr, all_labels_arr)
                for k, v in train_metrics.items():
                    print(f"  Train {k}: {v:.4f}")
                del all_probs_arr, all_preds_arr, all_labels_arr
                gc.collect()
            else:
                train_metrics = {}

            avg_train_loss = total_loss / len(train_loader)
            accelerator.log({"epoch_train_loss": avg_train_loss}, step=epoch)
            print(f"\n[Epoch {epoch + 1}/{epochs}] Training Summary:")
            print(f"  Avg Loss: {avg_train_loss:.4f}")

            del all_probs, all_preds, all_labels
            gc.collect()

            # 验证和保存
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                model.eval()
                # 这里可以添加验证逻辑

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                timestamp = datetime.now().strftime("%m%d-%H")
                name_save = os.path.join(checkpoint_dir,
                    f'vetime_dim{current_dim}_epoch{epoch+1}_{timestamp}.pth')

                if accelerator.is_main_process:
                    torch.save(unwrapped_model.state_dict(), name_save)
                    logger.info(f"Model saved at epoch {epoch+1}")

                early_stopping(avg_train_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

                model.train()
                vision_model.eval()
                gc.collect()
                torch.cuda.empty_cache()

            del train_metrics
            gc.collect()

        # ========== 9. 保存维度最终 Checkpoint ==========
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = os.path.join(checkpoint_dir, f'vetime_dim{current_dim}_final.pth')

        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), checkpoint_path)
            print(f"[INFO] 维度 {current_dim} 最终 Checkpoint 已保存: {checkpoint_path}")

        prev_checkpoint_path = checkpoint_path

        # ========== 10. 彻底清理显存 ==========
        del model, optimizer, train_loader, ts_model, train_dataset
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print("[多变量训练] 所有数据集训练完成！")
    print(f"{'='*60}")

    accelerator.end_training()
    return {"status": "completed", "datasets": len(datasets)}
```

**Step 2: 验证语法正确**

```bash
python -c "from train import train_multivariate; print('train_multivariate loaded')"
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add train_multivariate function for dimension-wise sequential training"
```

---

## Task 7: 创建新的 main 入口函数

**Files:**
- Modify: `/mnt/sda/cjmProject/VETime/train.py`

**Step 1: 在文件末尾添加新的 main 函数**

```python
def main(args):
    """
    主入口：根据配置选择训练模式

    - univariate: 单变量训练（原有逻辑）
    - multivariate: 多变量顺序训练
    """
    # 加载配置文件
    if args.config is not None:
        config = load_config(args.config)
        training_mode = config.get('training_mode', 'univariate')
        print(f"[INFO] 加载配置文件: {args.config}")
        print(f"[INFO] 训练模式: {training_mode}")
    else:
        config = None
        training_mode = 'univariate'

    if training_mode == 'univariate':
        return train_univariate(args)
    elif training_mode == 'multivariate':
        if config is None:
            raise ValueError("多变量训练模式需要指定 --config 参数")
        return train_multivariate(args, config)
    else:
        raise ValueError(f"未知的训练模式: {training_mode}")
```

**Step 2: 修改 __main__ 入口**

找到 `if __name__ == "__main__":` 块，将最后的调用改为：

```python
    args = parser.parse_args()
    output_file_path = args.output_file_path.replace('result.json', f'{args.model_name.replace("/", "-")}_result.json')

    results = main(args)

    if results is not None:
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=4)
```

**Step 3: 验证语法正确**

```bash
python -c "from train import main; print('main function loaded')"
```

**Step 4: Commit**

```bash
git add train.py
git commit -m "feat: add main entry function with mode selection"
```

---

## Task 8: 验证单变量训练兼容性

**Files:**
- Test: 单变量训练流程

**Step 1: 验证不指定配置文件时的行为**

```bash
python train.py --help
```

**Step 2: 运行单变量训练（dry run，检查是否正确调用 train_univariate）**

```bash
python train.py --dataset_path ./dataset/test_univariate.pkl --num_epochs 1 --batch_size 2 2>&1 | head -50
```

**Step 3: 确认单变量路径无回归**

检查输出中是否包含 `[INFO] 训练模式: univariate` 或类似的提示。

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify univariate training compatibility"
```

---

## Task 9: 创建示例单变量配置文件

**Files:**
- Create: `config/univariate_config.yaml`

**Step 1: 创建单变量训练配置文件**

```yaml
# config/univariate_config.yaml

# 训练模式: univariate(单变量)
training_mode: univariate

# 单变量训练配置
univariate:
  dataset_path: ./dataset/univariate.pkl
  batch_size: 32
```

**Step 2: 验证配置文件**

```bash
cat config/univariate_config.yaml
```

**Step 3: Commit**

```bash
git add config/univariate_config.yaml
git commit -m "feat: add univariate training config example"
```

---

## Task 10: 最终验证和文档更新

**Files:**
- Modify: `README.md` (可选)

**Step 1: 验证多变量配置文件加载**

```bash
python -c "
from train import load_config
config = load_config('config/multivariate_config.yaml')
print(f\"Training mode: {config['training_mode']}\")
print(f\"Datasets: {len(config['multivariate']['datasets'])}\")
"
```

**Step 2: 运行完整语法检查**

```bash
python -m py_compile train.py && echo "Syntax check passed"
```

**Step 3: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete multivariate sequential training implementation"
```

---

## 执行选择

**Plan complete and saved to `docs/plans/2026-06-12-multivariate-training-impl.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
