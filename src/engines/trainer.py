"""单变量训练引擎。"""

import os
import gc
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from datetime import datetime

from accelerate import Accelerator

from src.utils.logger import get_logger
from src.engines.hooks import freeze_for_cls_warmup, restore_requires_grad
from src.losses.balance import load_balance_loss
from src.datasets.renderers import create_renderer  # 新增

logger = get_logger(__name__)


class Trainer:
    """
    单变量训练引擎，承载两阶段课程训练循环。

    Stage 1 (epoch < stage1_epochs): 纯重构预训练，分类损失归零
    Stage 2 (epoch >= stage1_epochs): 多任务联合训练，含分类预热
    """

    def __init__(self, cfg, model, train_loader, val_loader, accelerator,
                 data_setting, patch_size):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.data_setting = data_setting
        self.patch_size = patch_size

        self.global_step = 0
        self.accumulated_samples = 0
        self.start_epoch = 0
        self.device = accelerator.device

        # 初始化辅助视觉渲染器（默认 ViCO；配置可选择严格时频 STFT）
        renderer_name = self._get_renderer_name(cfg)
        self.vico_renderer = create_renderer(renderer_name)
        logger.info(f"辅助视觉渲染器: {self.vico_renderer}")

        # 从 cfg 中提取常用字段
        self.epochs = cfg.training.total_epochs
        self.stage1_epochs = cfg.training.stage1_epochs
        self.img_size = data_setting.get('img_size', 224)
        self.model_name = cfg.model.model_name
        self.dynamic_batch = getattr(cfg.data, 'dynamic_batch', False)
        self.effective_batch_size = cfg.data.effective_batch_size

        # Query Decoder 模式：单阶段训练，无需课程学习
        self.use_query_decoder = getattr(cfg.model, 'use_query_decoder', False)
        if self.use_query_decoder:
            self.stage1_epochs = 0  # 跳过 Stage 1，直接联合训练

        # 输出
        self.output = []
        self.name_save = f'./output/{self.model_name}__{self.img_size}_best.pth'
        self.output_path0 = f'./output/score/uni/{self.model_name}_train'
        self.checkpoint_dir = f'./output/checkpoints/{self.model_name}'
        os.makedirs(self.output_path0, exist_ok=True)

    def _get_renderer_name(self, cfg) -> str:
        """从配置中获取渲染器名称，默认 'vico'。"""
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'vision_branch'):
            return getattr(cfg.model.vision_branch, 'vico_renderer', 'vico')
        return 'vico'

    # =================================================================
    # Setup
    # =================================================================

    def setup(self):
        """构建 optimizer、scheduler、early_stopping，并用 accelerator.prepare 封装。"""
        cfg = self.cfg

        # ---- Optimizer ----
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params_list,
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
        logger.info(
            f"Optimizer: AdamW  lr={cfg.training.optimizer.lr}  "
            f"weight_decay={cfg.training.optimizer.weight_decay}"
        )

        # ---- Accelerator prepare ----
        if self.val_loader is not None:
            self.model, self.optimizer, self.train_loader, self.val_loader = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader
                )
            )
        else:
            self.model, self.optimizer, self.train_loader = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader
                )
            )

        # ---- Scheduler ----
        if self.dynamic_batch:
            # 近似值：训练集样本数 / effective_batch_size
            steps_per_epoch = max(1, len(self.train_loader.dataset) // self.effective_batch_size)
        else:
            grad_accum = self.accelerator.gradient_accumulation_steps
            steps_per_epoch = max(1, len(self.train_loader) // grad_accum)

        patience = cfg.training.early_stopping.patience
        warmup_steps = steps_per_epoch  # 1 epoch warmup
        total_steps = (patience + 2) * steps_per_epoch
        cosine_steps = total_steps - warmup_steps

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # 硬编码，与原始 train_univariate 一致
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cosine_steps, eta_min=cfg.training.scheduler.eta_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        logger.info(
            f"Scheduler: Warmup({warmup_steps} steps) + "
            f"Cosine(cosine_steps={cosine_steps}, eta_min={cfg.training.scheduler.eta_min}), "
            f"total={total_steps} steps"
        )

        # ---- TensorBoard ----
        run_name = f"vetime_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.accelerator.init_trackers(run_name)
        logger.info(f"TensorBoard run: {run_name}")

        # ---- EarlyStopping ----
        from Test_TSB import EarlyStopping
        self.early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=self.name_save
        )

        # ---- Resume ----
        self._resume_if_needed()

    def _resume_if_needed(self):
        """从 checkpoint 恢复训练状态（如有 cfg.paths.resume）。"""
        resume_path = getattr(self.cfg.paths, 'resume', None)
        if not resume_path:
            return
        if not os.path.exists(resume_path):
            logger.error(f"Resume checkpoint 不存在: {resume_path}")
            return

        logger.info(f"正在加载 resume checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            unwrapped = self.accelerator.unwrap_model(self.model)
            missing, unexpected = unwrapped.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            logger.info("模型权重已恢复")
            if missing:
                logger.info(f"  缺失的参数: {len(missing)} 个")
            if unexpected:
                logger.info(f"  未预期的参数: {len(unexpected)} 个")

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer 状态已恢复")

            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']

            best_val_loss_resume = checkpoint.get('best_val_loss')
            if best_val_loss_resume is not None:
                self.early_stopping.val_loss_min = best_val_loss_resume
                logger.info(f"早停最佳验证损失已恢复: {best_val_loss_resume:.4f}")

            # 恢复 scheduler
            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
                logger.info(
                    f"学习率调度器状态已恢复，当前 LR={self.scheduler.get_last_lr()[0]:.2e}"
                )
            else:
                logger.warning(
                    f"旧 checkpoint 无调度器状态，手动推进 {self.global_step} 步"
                )
                for _ in range(self.global_step):
                    self.scheduler.step()

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
            logger.info("随机状态已恢复")

            logger.info(
                f"从 epoch {self.start_epoch} 继续训练，global_step={self.global_step}"
            )
        else:
            # 旧格式：仅模型权重
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.load_state_dict(checkpoint, strict=False)
            logger.info("旧格式 checkpoint，仅恢复了模型权重，从 epoch 0 开始训练")

    # =================================================================
    # Training loop
    # =================================================================

    def train_epoch(self, epoch):
        """执行单个 epoch 的训练。"""
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        accelerator = self.accelerator
        data_setting = self.data_setting

        is_stage_1 = epoch < self.stage1_epochs
        if self.use_query_decoder:
            logger.info(
                f"[Query Decoder] Epoch {epoch+1}/{self.epochs}: "
                "单阶段联合训练 (重构 + 分类)"
            )
        elif is_stage_1:
            logger.info(
                f"[Phase 1] Epoch {epoch+1}/{self.epochs}: "
                "纯重构预训练 (异常分类损失已切断)"
            )
        else:
            logger.info(
                f"[Phase 2] Epoch {epoch+1}/{self.epochs}: "
                "多任务联合训练 (异常分类损失恢复)"
            )

        # ---- 分类预热 ----
        # Query Decoder 模式下不需要分类预热（单阶段训练）
        is_cls_warmup_epoch = (
            (not self.use_query_decoder)
            and (not is_stage_1)
            and (epoch == self.stage1_epochs)
            and (cfg.training.cls_warmup_ratio > 0)
        )
        cls_warmup_active = False
        saved_requires_grad = None
        cls_warmup_batches = 0
        if is_cls_warmup_epoch:
            cls_warmup_batches = max(1, int(len(self.train_loader) * cfg.training.cls_warmup_ratio))
            saved_requires_grad = freeze_for_cls_warmup(model, accelerator)
            cls_warmup_active = True
            logger.info(
                f"[Cls Warmup] Epoch {epoch+1}: 前 {cls_warmup_batches}/{len(self.train_loader)} "
                f"个 batch 仅训练分类网络 (ratio={cfg.training.cls_warmup_ratio})"
            )

        model.train()
        total_loss = 0
        total_loss_bce = 0
        total_loss_mse = 0
        total_loss_cl = 0
        total_loss_e = 0
        all_probs, all_preds, all_labels = [], [], []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}[Train]",
            disable=not accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # ---- 分类预热解冻检查 ----
            if cls_warmup_active and batch_idx == cls_warmup_batches:
                restore_requires_grad(model, accelerator, saved_requires_grad)
                optimizer.zero_grad()
                cls_warmup_active = False
                progress_bar.set_description(f"Epoch {epoch+1}[Train]")
                logger.info(
                    f"[Cls Warmup] 分类预热完成 (batch {batch_idx}/{len(self.train_loader)})，"
                    "已解冻所有参数，恢复多任务联合训练"
                )

            labels = batch["labels"]
            images = batch["image"]           # VETime 时域图像 (B, C, H, W)
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            time_series_raw = batch['time_series_raw']  # 未归一化原始时序，供 ViCO 渲染
            mask = batch['mask']
            period = batch['period']
            p_value = batch['padding_value']

            alpha_recon = cfg.loss.alpha_recon
            cl_weight = cfg.loss.cl_weight
            balance_weight = cfg.loss.balance_weight

            # ---- Forward + Loss ----
            # 使用 model.fold_images / model.split_sequence 代替内部调用，
            # 但损失计算保持与 train_univariate 完全一致的直接调用方式，
            # 以确保梯度流和数值行为完全相同。
            if labels.shape[1] > model.MAX_L:
                # 长序列分块
                data_splits = model.split_sequence(images, time_series, att_mask, labels, time_series_raw)
                loss1 = 0
                loss2 = 0
                batch_loss_bce = 0
                batch_loss_mse = 0
                batch_loss_cl = 0
                batch_loss_e = 0
                logits_list = []

                for data_part in data_splits:
                    img_part, ts_part, att_mask_part, label_part, ts_raw_part = data_part
                    images_folded, init_img_size = model.fold_images(
                        img_part, period, p_value, **data_setting
                    )

                    # 每个 chunk 从原始时序渲染辅助视觉图像。
                    images_vico_chunk = self.vico_renderer.render_batch(
                        ts_raw_part, att_mask=att_mask_part, img_size=self.img_size
                    )
                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                        hidden_states=images_folded,
                        hidden_states_vico=images_vico_chunk,
                        time_series=ts_part,
                        att_mask=att_mask_part,
                        init_img_size=init_img_size,
                        labels=label_part,
                    )

                    loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)
                    loss02, rec = model.weighted_reconstruction_loss(
                        local_embeddings2, ts_part, att_mask_part, label_part
                    )

                    # 两阶段训练范式：门控负载均衡损失（Query Decoder 模式下 m_w=None）
                    if m_w is None:
                        batch_loss_e_part = torch.tensor(0.0, device=self.device)
                    elif is_stage_1:
                        batch_loss_e_part = balance_weight * load_balance_loss(m_w[0])
                    else:
                        batch_loss_e_part = balance_weight * 0.5 * (
                            load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
                        )

                    # 两阶段训练范式：异常分类损失
                    if is_stage_1:
                        loss01 = torch.tensor(0.0, device=self.device)

                    # 记录未缩放的原始数值用于 log
                    batch_loss_bce += loss01.item()
                    batch_loss_mse += loss02.item()
                    batch_loss_cl += (cl_weight * loss_cl).item()
                    batch_loss_e += batch_loss_e_part.item()

                    # 只对 loss02 乘 alpha_recon，其他 Loss 保持原样
                    loss2 = loss2 + (alpha_recon * loss02) + cl_weight * loss_cl + batch_loss_e_part
                    loss1 = loss1 + loss01
                    logits_list.append(logit)

                logits = torch.cat(logits_list, dim=1)
                num_splits = len(data_splits)
                if num_splits > 0:
                    loss1 = loss1 / num_splits
                    loss2 = loss2 / num_splits
                    batch_loss_bce /= num_splits
                    batch_loss_mse /= num_splits
                    batch_loss_cl /= num_splits
                    batch_loss_e /= num_splits

            else:
                # 常规序列
                images_folded, init_img_size = model.fold_images(
                    images, period, p_value, **data_setting
                )

                # 渲染辅助视觉图像，att_mask 会过滤 padding。
                images_vico = self.vico_renderer.render_batch(
                    time_series_raw, att_mask=att_mask, img_size=self.img_size
                )
                local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                    hidden_states=images_folded,
                    hidden_states_vico=images_vico,
                    time_series=time_series,
                    att_mask=att_mask,
                    init_img_size=init_img_size,
                    labels=labels,
                )

                loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)
                loss_recon, rec = model.weighted_reconstruction_loss(
                    local_embeddings2, time_series, att_mask, labels
                )

                # 两阶段训练范式：门控负载均衡损失（Query Decoder 模式下 m_w=None）
                if m_w is None:
                    loss_e_tensor = torch.tensor(0.0, device=self.device)
                elif is_stage_1:
                    loss_e_tensor = balance_weight * load_balance_loss(m_w[0])
                else:
                    loss_e_tensor = balance_weight * 0.5 * (
                        load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
                    )

                # 两阶段训练范式：异常分类损失
                if is_stage_1:
                    loss1 = torch.tensor(0.0, device=self.device)

                loss2 = (alpha_recon * loss_recon) + loss_e_tensor + cl_weight * loss_cl

                batch_loss_bce = loss1.item()
                batch_loss_mse = loss_recon.item()
                batch_loss_cl = (cl_weight * loss_cl).item()
                batch_loss_e = loss_e_tensor.item()

            # ---- Backward ----
            accelerator.backward(loss1 + loss2)

            # ---- 梯度累积 ----
            current_bs = labels.shape[0]
            if self.dynamic_batch:
                self.accumulated_samples += current_bs
                if self.accumulated_samples >= self.effective_batch_size:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    self.accumulated_samples = 0
            else:
                self.global_step += 1
                if self.global_step % accelerator.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            batch_loss = loss1.item() + loss2.item()
            total_loss += batch_loss
            total_loss_bce += batch_loss_bce
            total_loss_mse += batch_loss_mse
            total_loss_cl += batch_loss_cl
            total_loss_e += batch_loss_e

            progress_bar.set_postfix({
                "Tot": f"{batch_loss:.3f}",
                "BCE": f"{batch_loss_bce:.3f}",
                "MSE": f"{batch_loss_mse:.3f}",
                "CL": f"{batch_loss_cl:.3f}",
                "Bal": f"{batch_loss_e:.4f}",
            })
            if cls_warmup_active and batch_idx < cls_warmup_batches:
                progress_bar.set_description(f"Epoch {epoch+1}[Train|CLS_Warmup]")

            # ---- TensorBoard ----
            if self.global_step > 0:
                unwrapped = accelerator.unwrap_model(model)
                accelerator.log({
                    "Loss/Total": batch_loss,
                    "Loss/BCE_Anomaly": batch_loss_bce,
                    "Loss/MSE_Recon": batch_loss_mse,
                    "Loss/CL_Contrastive": batch_loss_cl,
                    "Loss/Balance": batch_loss_e,
                    "Train/LR": optimizer.param_groups[0]['lr'],
                    "Gate/alpha": unwrapped.visual_cross_attn.alpha.item(),
                }, step=self.global_step)

            # ---- 采样预测指标 ----
            probs = torch.softmax(logits, dim=-1)[:, :, 1]
            preds = (probs > 0.5).float()
            probs, preds, labels = accelerator.gather_for_metrics((probs, preds, labels))
            if self.global_step % 10 == 0:
                for i in range(probs.shape[0]):
                    all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                    all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                    all_labels.append(labels[i].detach().cpu().numpy().reshape(-1).astype(int))

            # 清理
            del images_folded, logits, loss1, probs, preds, labels, loss2
            del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, mask, period, p_value
            del images, time_series, att_mask, init_img_size

        # ---- Epoch 结束：flush 剩余梯度 ----
        if self.dynamic_batch and self.accumulated_samples > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            self.global_step += 1
            self.accumulated_samples = 0

        # ---- 分类预热安全保障 ----
        if cls_warmup_active:
            restore_requires_grad(model, accelerator, saved_requires_grad)
            optimizer.zero_grad()
            cls_warmup_active = False
            logger.info(
                f"[Cls Warmup] Epoch {epoch+1} 结束，强制解冻所有参数（预热覆盖整个 epoch）"
            )

        # ---- Epoch 级别指标 ----
        avg_train_loss = total_loss / len(self.train_loader)
        avg_loss_bce = total_loss_bce / len(self.train_loader)
        avg_loss_mse = total_loss_mse / len(self.train_loader)
        avg_loss_cl = total_loss_cl / len(self.train_loader)
        avg_loss_e = total_loss_e / len(self.train_loader)

        unwrapped = accelerator.unwrap_model(model)
        accelerator.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_loss_bce": avg_loss_bce,
            "epoch_loss_mse": avg_loss_mse,
            "epoch_loss_cl": avg_loss_cl,
            "epoch_loss_e": avg_loss_e,
            "epoch_alpha_mean": unwrapped.visual_cross_attn.alpha.mean().item(),
            "epoch_alpha_max": unwrapped.visual_cross_attn.alpha.max().item(),
            "epoch_alpha_min": unwrapped.visual_cross_attn.alpha.min().item(),
        }, step=epoch)

        logger.info(
            f"\n[Epoch {epoch + 1}/{self.epochs}] Training Summary: "
            f"Avg Loss={avg_train_loss:.4f} "
            f"(BCE={avg_loss_bce:.4f}, MSE={avg_loss_mse:.4f}, "
            f"CL={avg_loss_cl:.4f}, Bal={avg_loss_e:.4f})"
        )

        # ---- 训练指标（采样） ----
        train_metrics = {}
        if len(all_probs) > 0:
            all_probs_arr = np.concatenate(all_probs)
            all_preds_arr = np.concatenate(all_preds)
            all_labels_arr = np.concatenate(all_labels)

            if np.any(np.isnan(all_probs_arr)):
                logger.warning("all_probs contains NaN values!")

            from evaluation.metrics import fast_get_metrics
            train_metrics = fast_get_metrics(all_probs_arr, all_labels_arr)
            for k, v in train_metrics.items():
                print(f"  Train {k}: {v:.4f}")

            del all_probs_arr, all_preds_arr, all_labels_arr
            gc.collect()

        del all_probs, all_preds, all_labels
        gc.collect()

        return avg_train_loss, train_metrics, avg_loss_bce, avg_loss_mse, avg_loss_cl, avg_loss_e

    # =================================================================
    # Validation
    # =================================================================

    def validate(self, epoch):
        """验证阶段：根据 val_mode 调用不同验证逻辑。"""
        cfg = self.cfg
        model = self.model
        accelerator = self.accelerator

        val_mode = getattr(cfg.data, 'val_mode', 'tsb')
        if val_mode == 'split' and self.val_loader is not None:
            avg_val_loss = self._evaluate_split()
            accelerator.log({"epoch_val_loss": avg_val_loss}, step=epoch)
            logger.info(f"  Avg Val Loss (split): {avg_val_loss:.4f}")
            return avg_val_loss
        else:
            # tsb 模式：由 run() 中的 TSB_test 处理
            return None

    def _evaluate_split(self):
        """基于训练集划分的验证集评估。"""
        model = self.model
        accelerator = self.accelerator
        data_setting = self.data_setting
        cl_weight = self.cfg.loss.cl_weight
        balance_weight = self.cfg.loss.balance_weight

        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                labels = batch["labels"]
                images = batch["image"]
                time_series, att_mask = batch['time_series'], batch['attention_mask']
                time_series_raw = batch['time_series_raw']
                period = batch['period']
                p_value = batch['padding_value']

                if labels.shape[1] > model.MAX_L:
                    data_splits = model.split_sequence(images, time_series, att_mask, labels, time_series_raw)
                    loss1_total = 0
                    loss2_total = 0

                    for data_part in data_splits:
                        img_part, ts_part, att_mask_part, label_part, ts_raw_part = data_part
                        images_folded, init_img_size = model.fold_images(
                            img_part, period, p_value, **data_setting
                        )

                        # 每个 chunk 从原始时序渲染辅助视觉图像。
                        images_vico_chunk = self.vico_renderer.render_batch(
                            ts_raw_part, att_mask=att_mask_part, img_size=self.img_size
                        )
                        local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                            hidden_states=images_folded,
                            hidden_states_vico=images_vico_chunk,
                            time_series=ts_part,
                            att_mask=att_mask_part,
                            init_img_size=init_img_size,
                            labels=label_part,
                        )

                        loss01, _ = model.anomaly_detection_loss(local_embeddings1, label_part)
                        loss02, _ = model.weighted_reconstruction_loss(
                            local_embeddings2, ts_part, att_mask_part, label_part
                        )

                        # Query Decoder 模式下 m_w=None，跳过 load_balance_loss
                        if m_w is None:
                            loss_balance_val = 0.0
                        else:
                            loss_balance_val = balance_weight * 0.5 * (
                                load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
                            )
                        loss2_total = loss2_total + loss02 + cl_weight * loss_cl + loss_balance_val
                        loss1_total = loss1_total + loss01

                    batch_loss = loss1_total.item() + loss2_total.item()
                else:
                    images_folded, init_img_size = model.fold_images(
                        images, period, p_value, **data_setting
                    )

                    images_vico = self.vico_renderer.render_batch(
                        time_series_raw, att_mask=att_mask, img_size=self.img_size
                    )
                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                        hidden_states=images_folded,
                        hidden_states_vico=images_vico,
                        time_series=time_series,
                        att_mask=att_mask,
                        init_img_size=init_img_size,
                        labels=labels,
                    )

                    loss1, _ = model.anomaly_detection_loss(local_embeddings1, labels)
                    loss2, _ = model.weighted_reconstruction_loss(
                        local_embeddings2, time_series, att_mask, labels
                    )
                    # Query Decoder 模式下 m_w=None，跳过 load_balance_loss
                    if m_w is None:
                        loss_balance_val = 0.0
                    else:
                        loss_balance_val = balance_weight * 0.5 * (
                            load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
                        )
                    loss2 = loss2 + loss_balance_val + cl_weight * loss_cl

                    batch_loss = loss1.item() + loss2.item()

                total_loss += batch_loss
                num_batches += 1

                del images_folded, local_embeddings1, local_embeddings2
                del images, time_series, att_mask, labels

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        model.train()
        return avg_loss

    # =================================================================
    # Main loop
    # =================================================================

    def run(self):
        """主训练循环：setup → epoch loop → early stopping。"""
        self.setup()
        model = self.model
        accelerator = self.accelerator
        cfg = self.cfg

        val_mode = getattr(cfg.data, 'val_mode', 'tsb')
        from Test_TSB import PASS_LIST, TSB_test
        from src.utils.checkpoint import save_full_checkpoint
        from src.utils.checkpoint_architecture import checkpoint_state_dict, make_model_checkpoint

        for epoch in range(self.start_epoch, self.epochs):
            # ---- 训练 ----
            (avg_train_loss, train_metrics,
             avg_loss_bce, avg_loss_mse, avg_loss_cl, avg_loss_e) = self.train_epoch(epoch)

            # ---- 验证 ----
            if val_mode == 'split' and self.val_loader is not None:
                avg_val_loss = self._evaluate_split()
                accelerator.log({"epoch_val_loss": avg_val_loss}, step=epoch)
                logger.info(f"  Avg Val Loss (split): {avg_val_loss:.4f}")

                # 早停
                self.early_stopping(avg_val_loss, model)
                if avg_val_loss <= self.early_stopping.val_loss_min:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_model_path = f'./output/{self.model_name}__{self.img_size}_best.pth'
                    if accelerator.is_main_process:
                        torch.save(
                            make_model_checkpoint(
                                unwrapped_model.state_dict(),
                                use_query_decoder=self.use_query_decoder,
                            ),
                            best_model_path,
                        )
                        logger.info(f"  Best model saved: {best_model_path} (val_loss={avg_val_loss:.4f})")

                if self.early_stopping.early_stop:
                    logger.info("Early stopping triggered (based on validation split).")
                    break

                # split 模式下定期 TSB_test
                if (epoch + 1) % 2 == 0 or epoch == self.epochs - 1:
                    model.eval()
                    # 构造 args-like 对象供 TSB_test 使用
                    args_for_tsb = _cfg_to_args(cfg)
                    avg_tsb_val_loss = TSB_test(
                        model, args_for_tsb, self.data_setting, self.device,
                        dataset_setting=PASS_LIST, verbose=False,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    timestamp = datetime.now().strftime("%m%d-%H")
                    name_save = (
                        f'./output/{self.model_name}__{self.img_size}'
                        f'_{avg_tsb_val_loss:.4f}_{timestamp}.pth'
                    )
                    torch.save(
                        make_model_checkpoint(
                            unwrapped_model.state_dict(),
                            use_query_decoder=self.use_query_decoder,
                        ),
                        name_save,
                    )
                    logger.info(
                        f"Model saved at epoch {epoch+1} with TSB_val_loss={avg_tsb_val_loss:.4f}"
                    )

                    epoch_log = {
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                        "val_loss_split": round(avg_val_loss, 6),
                        "val_loss_tsb": round(avg_tsb_val_loss, 6) if avg_tsb_val_loss is not None else None,
                    }
                    self.output.append(epoch_log)
                    model.train()
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    epoch_log = {
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                        "val_loss_split": round(avg_val_loss, 6),
                    }
                    self.output.append(epoch_log)

            else:
                # tsb 模式
                if (epoch + 1) % 2 == 0 or epoch == self.epochs - 1:
                    model.eval()
                    args_for_tsb = _cfg_to_args(cfg)
                    avg_val_loss = TSB_test(
                        model, args_for_tsb, self.data_setting, self.device,
                        dataset_setting=PASS_LIST, verbose=False,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    timestamp = datetime.now().strftime("%m%d-%H")
                    name_save = (
                        f'./output/{self.model_name}__{self.img_size}'
                        f'_{avg_val_loss:.4f}_{timestamp}.pth'
                    )
                    torch.save(
                        make_model_checkpoint(
                            unwrapped_model.state_dict(),
                            use_query_decoder=self.use_query_decoder,
                        ),
                        name_save,
                    )
                    logger.info(f"Model saved at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")

                    epoch_log = {
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                        "val_loss_tsb": round(avg_val_loss, 6) if avg_val_loss is not None else None,
                    }
                    self.output.append(epoch_log)

                    self.early_stopping(avg_val_loss, model)
                    if self.early_stopping.early_stop:
                        logger.info("Early stopping triggered (based on TSB validation).")
                        break

                    model.train()
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    epoch_log = {
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                    }
                    self.output.append(epoch_log)

            # ---- 保存完整 checkpoint ----
            accelerator.wait_for_everyone()
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            epoch_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'univariate_epoch{epoch}_full.pth'
            )
            save_full_checkpoint(
                model, self.optimizer, self.scheduler, epoch, self.global_step,
                0, 0, None,  # dataset_idx, current_dim, prev_checkpoint_path
                self.early_stopping.val_loss_min,
                0,  # patience_counter (由 EarlyStopping 管理)
                epoch_checkpoint_path, accelerator,
            )

            del train_metrics
            gc.collect()

        # ---- 训练结束 ----
        if val_mode == 'split' and self.val_loader is not None:
            best_model_path = f'./output/{self.model_name}__{self.img_size}_best.pth'
            if os.path.exists(best_model_path):
                logger.info(f"加载早停保存的最佳模型: {best_model_path}")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(
                    checkpoint_state_dict(torch.load(best_model_path, map_location='cpu'))
                )

        args_for_tsb = _cfg_to_args(cfg)
        loss_all = TSB_test(
            model, args_for_tsb, self.data_setting, self.device,
            dataset_setting=PASS_LIST, verbose=False,
        )
        logger.info(f"Final TSB validation loss: {loss_all}")
        accelerator.end_training()
        logger.info("Training completed!")

        return self.output


# =====================================================================
# 辅助函数
# =====================================================================

class _ArgsProxy:
    """将 OmegaConf DictConfig 包装为 TSB_test 兼容的 args 对象。"""

    def __init__(self, cfg):
        self.cfg = cfg
        # TSB_test 需要的属性
        self.model_name = cfg.model.model_name
        self.vision_name = cfg.model.vision_name
        self.ts_finetune_type = cfg.model.ts_finetune_type
        self.save_dir = getattr(cfg.paths, 'save_dir', './output')
        self.target_dir = os.path.join(self.save_dir, self.model_name)
        os.makedirs(self.target_dir, exist_ok=True)
        self.dataset_dir = getattr(cfg.paths, 'dataset_test_dir', '')
        self.ts_path = getattr(cfg.paths, 'ts_path', None)
        self.vetime_path = getattr(cfg.paths, 'vetime_path', None)
        self.data_setting = {
            'img_size': 224,
            'T_sqrt': False,
        }
        self.num_workers = getattr(cfg.data, 'num_workers', 5)
        self.use_vectorized_fold = getattr(cfg.model, 'use_vectorized_fold', True)

        # file_list
        if hasattr(cfg.paths, 'file_list') and cfg.paths.file_list:
            import pandas as pd
            if cfg.paths.file_list.endswith('.csv'):
                df = pd.read_csv(cfg.paths.file_list)
                self.file_list = df['filename'].tolist() if 'filename' in df.columns else df.iloc[:, 0].tolist()
            else:
                self.file_list = sorted(os.listdir(self.dataset_dir))
        else:
            self.file_list = sorted(os.listdir(self.dataset_dir)) if self.dataset_dir else []


def _cfg_to_args(cfg):
    """将 OmegaConf DictConfig 转换为 TSB_test 兼容的 args 对象。"""
    return _ArgsProxy(cfg)
