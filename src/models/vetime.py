import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Union, Optional
from dataclasses import dataclass
from src.losses.contrastive import win_Contrastive_Loss
from src.models.ts_encoder.ts_encoder import TimeSeriesEncoder
from src.models.ts_encoder.ts_model import TS_Model
from src.models.vts_module import V_Attention, VTS_Alignment, M_moe, GatedTimeFrequencyFusion


class VETIME(TS_Model):
    """Model for time series pretraining with masked reconstruction and anomaly detection."""

    def __init__(self, config_v, vision_model,config_t,ts_model, model_name=None, use_gradient_checkpointing=False, **kwargs):
        super().__init__(config_t, **kwargs)

        # vison setting
        self.vit_encoder = vision_model
        v_dim=vision_model.hidden_size
        t_dim=config_t.d_model
        self.name=model_name
        self.MAX_L=vision_model.MAX_L
        self.use_gradient_checkpointing = use_gradient_checkpointing

        t_dim2 = int(t_dim*2)
        self.mlp_i = nn.Sequential(
            nn.Linear(v_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, t_dim),
            nn.LayerNorm(t_dim),
        )
        self.pos_emb_v = nn.Parameter(torch.zeros(1, self.MAX_L, v_dim))
        nn.init.normal_(self.pos_emb_v, std=0.02)
        self.I_att = V_Attention(t_dim)

        # === 时域—时频视觉双分支 ===
        # 时频引导视觉适配器：用时域图特征查询多尺度 STFT 图特征
        # Channel-wise LayerScale (init_scale=1e-3) 实现近乎恒等映射，训练初期自动回退到纯 VETime
        self.visual_cross_attn = GatedTimeFrequencyFusion(t_dim, num_heads=8, dropout=0.1)

        # 辅助时频分支的 MLP（与时域图分支结构一致；保留名称以兼容 checkpoint）
        self.mlp_vico = nn.Sequential(
            nn.Linear(v_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, t_dim),
            nn.LayerNorm(t_dim),
        )

        # ts setting
        self.ts_encoder = ts_model
        self.patch_size =self.ts_encoder.patch_size
        self.projection_layer = self.ts_encoder.ts_encoder.projection_layer
        self.reconstruction_head = ts_model.reconstruction_head
        self.anomaly_head = ts_model.anomaly_head
        self.d_proj=ts_model.d_proj

        # fusion setting
        self.fusion = VTS_Alignment(v_dim,t_dim)
        self.mm_w = M_moe(t_dim)
        
        # 新增：Query-based 解码器（可选）
        self.query_decoder = None
        self.fusion_proj = None
        self.use_query_decoder = kwargs.get('use_query_decoder', False)

        if self.use_query_decoder:
            from src.models.vts_module import QueryDecoder
            self.query_decoder = QueryDecoder(
                d_model=t_dim,
                num_heads=4,
                dropout=0.1
            )
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(t_dim * 2),
                nn.Linear(t_dim * 2, t_dim)
            )
        
        # loss setting
        self.cl_loss=win_Contrastive_Loss(t_dim,temperature=0.1)

        # Enable gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def forward(self, hidden_states: torch.Tensor,
                time_series: torch.Tensor,
                att_mask: Optional[torch.Tensor] = None,
                init_img_size=None,
                hidden_states_vico: Optional[torch.Tensor] = None,
                init_img_size_vico=None,
                labels=None):

        # 使用 gradient checkpointing 包装整个 forward 的核心计算
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(
                hidden_states, time_series, att_mask, init_img_size,
                hidden_states_vico, init_img_size_vico, labels
            )
        else:
            return self._forward_impl(
                hidden_states, time_series, att_mask, init_img_size,
                hidden_states_vico, init_img_size_vico, labels
            )

    def _forward_impl(self, hidden_states: torch.Tensor, time_series: torch.Tensor,
                      att_mask: Optional[torch.Tensor] = None, init_img_size=None,
                      hidden_states_vico: Optional[torch.Tensor] = None,
                      init_img_size_vico=None, labels=None):
        """实际的 forward 实现"""
        TS_embeddings0,local_embeddings0,patch_mask=self.ts_encoder(time_series,att_mask)
        B, seq_len, num_features = time_series.size()

        patch_num = patch_mask.size(1) // num_features

        temporal_pos_emb = self.pos_emb_v[:, :patch_num, :]

        multivariate_pos_emb = temporal_pos_emb.repeat(1, num_features, 1)

        # === 分支 A: 趋势分解时域图 ===
        image_features_vetime, _ = self.vit_encoder(hidden_states)
        I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size)
        I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
        # I_embeddings0 = self.I_att(I_embeddings, patch_mask)
        Q_visual = self.I_att(I_embeddings_vetime, patch_mask)  # [B, N_TS, t_dim]

        # === 分支 B: 多尺度 STFT 时频图（参数名保留 vico 以兼容 checkpoint） ===
        if hidden_states_vico is not None:
            # 使用传入的多尺度 STFT 图像通过共享 MAE encoder 提取时频 tokens
            image_features_vico, _ = self.vit_encoder(hidden_states_vico)
            K_V_tokens = image_features_vico[:, 1:, :]  # [B, 196, v_dim] 原始 patch tokens
        else:
            # 兼容旧代码：无辅助图像输入时使用时域图的 patch tokens 作为 fallback
            K_V_tokens = image_features_vetime[:, 1:, :]  # [B, 196, v_dim]
        K_V_tokens_proj = self.mlp_vico(K_V_tokens)   # [B, 196, t_dim]

        # === 交叉注意力融合 ===
        # GatedTimeFrequencyFusion: 时域图 Query 查询时频图 Key/Value
        I_embeddings0 = self.visual_cross_attn(
            Q_VETime=Q_visual,
            K_ViCO=K_V_tokens_proj,
            V_ViCO=K_V_tokens_proj
        )  # [B, N_TS, t_dim]
        
        
        # 后续流程保持不变：fusion → MoE → 任务头
        I_embeddings, TS_embeddings = self.fusion(I_embeddings0, TS_embeddings0, patch_mask)
        loss_sc=self.compute_cl(I_embeddings,TS_embeddings,labels,num_features)
        
        # === 新增：Query-based 解码路径 ===
        if self.use_query_decoder and self.query_decoder is not None:
            # 生成三个专家特征
            F_TS = TS_embeddings0  # (B, N, D) 时间专家
            F_V = I_embeddings0      # (B, N, D) 视觉专家

            # 融合专家特征
            mix_out0_for_proj = torch.cat([TS_embeddings, I_embeddings], dim=-1)
            F_A = self.fusion_proj(mix_out0_for_proj)  # (B, N, D)

            # Query-based 解码
            F_rec, F_cls = self.query_decoder(F_TS, F_V, F_A, patch_mask)

            # 任务头投影 - 分类分支
            patch_proj = self.projection_layer(F_cls)
            local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
            local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
            local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

            # 任务头投影 - 重构分支
            patch_proj2 = self.projection_layer(F_rec)
            local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
            local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
            local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

            # 新路径无路由权重
            m_w = None

            return local_embeddings1, m_w, loss_sc, local_embeddings2
        
        mix_out0 = torch.cat([TS_embeddings,I_embeddings],dim=-1)
        # 两路任务干净并行：task 1 -> anomaly head(local_emb1), task 0 -> reconstruction head(local_emb2)
        # 任务映射保持与原实现一致（原 mask=None 分支 = task 1 = anomaly）。
        mix_out_a, m_w_a = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=1)
        mix_out_r, m_w_r = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=0)
        # 路由权重以 dict 单独保存（不再 m_w1+m_w2 破坏概率分布），供 load_balance_loss 各自平衡
        m_w = {0: m_w_r, 1: m_w_a}

        patch_proj = self.projection_layer(mix_out_a)
        local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, seq_len, num_features, d_proj)

        patch_proj2 = self.projection_layer(mix_out_r)
        local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B,

        return local_embeddings1,m_w,loss_sc,local_embeddings2

    def _forward_with_checkpointing(self, hidden_states: torch.Tensor, time_series: torch.Tensor,
                                    att_mask: Optional[torch.Tensor], init_img_size,
                                    hidden_states_vico: Optional[torch.Tensor], init_img_size_vico, labels):
        """使用 gradient checkpointing 的 forward"""
        from torch.utils.checkpoint import checkpoint

        # 将 forward 分成多个可检查点的部分
        B, seq_len, num_features = time_series.size()

        # Part 1: TS encoder (已内置 checkpointing)
        TS_embeddings0, local_embeddings0, patch_mask = self.ts_encoder(time_series, att_mask)

        # Part 2: Vision encoder + fusion (使用 checkpoint)
        def vision_fusion_forward(hidden_states, hidden_states_vico, patch_mask, init_img_size, TS_embeddings0, num_features):
            patch_num = patch_mask.size(1) // num_features
            temporal_pos_emb = self.pos_emb_v[:, :patch_num, :]
            multivariate_pos_emb = temporal_pos_emb.repeat(1, num_features, 1)

            # 分支 A: VETime 时域
            image_features_vetime, _ = self.vit_encoder(hidden_states)
            I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size)
            I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
            Q_visual = self.I_att(I_embeddings_vetime, patch_mask)

            # 分支 B: ViCO 频域
            if hidden_states_vico is not None:
                image_features_vico, _ = self.vit_encoder(hidden_states_vico)
                K_V_tokens = image_features_vico[:, 1:, :]
            else:
                K_V_tokens = image_features_vetime[:, 1:, :]
            K_V_tokens_proj = self.mlp_vico(K_V_tokens)

            # 交叉注意力融合
            I_embeddings0 = self.visual_cross_attn(
                Q_VETime=Q_visual,
                K_ViCO=K_V_tokens_proj,
                V_ViCO=K_V_tokens_proj
            )

            I_embeddings, TS_embeddings = self.fusion(I_embeddings0, TS_embeddings0, patch_mask)
            return I_embeddings, TS_embeddings, I_embeddings0

        I_embeddings, TS_embeddings, I_embeddings0 = checkpoint(
            vision_fusion_forward,
            hidden_states, hidden_states_vico, patch_mask, init_img_size, TS_embeddings0, num_features
        )

        loss_sc = self.compute_cl(I_embeddings, TS_embeddings, labels, num_features)

        # Part 3: 解码器路径（MoE 或 Query Decoder）
        if self.use_query_decoder and self.query_decoder is not None:
            # Query Decoder 路径
            def query_decoder_forward(F_TS, F_V, mix_out0, B, seq_len, num_features):
                # 融合专家特征
                F_A = self.fusion_proj(mix_out0)

                # Query-based 解码
                F_rec, F_cls = self.query_decoder(F_TS, F_V, F_A, patch_mask)

                # 任务头投影 - 分类分支
                patch_proj = self.projection_layer(F_cls)
                local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
                local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
                local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

                # 任务头投影 - 重构分支
                patch_proj2 = self.projection_layer(F_rec)
                local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
                local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
                local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

                return local_embeddings1, local_embeddings2

            F_TS = TS_embeddings0
            F_V = I_embeddings0
            mix_out0 = torch.cat([TS_embeddings, I_embeddings], dim=-1)

            local_embeddings1, local_embeddings2 = checkpoint(
                query_decoder_forward,
                F_TS, F_V, mix_out0, B, seq_len, num_features
            )
            m_w = None
        else:
            # MoE 路径
            def moe_projection_forward(mix_out0, TS_embeddings0, I_embeddings0, B, seq_len, num_features):
                # 两路任务干净并行：task 1 -> anomaly, task 0 -> reconstruction（与 _forward_impl 一致）
                mix_out_a, m_w_a = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=1)
                mix_out_r, m_w_r = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=0)

                patch_proj = self.projection_layer(mix_out_a)
                local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
                local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
                local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

                patch_proj2 = self.projection_layer(mix_out_r)
                local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
                local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
                local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

                # 注意：torch.utils.checkpoint 只支持 Tensor / Tensor tuple，不支持 dict。
                # 因此把 m_w 按 (task0, task1) 固定顺序平铺成 tuple 返回，外部重组为 dict。
                return local_embeddings1, local_embeddings2, m_w_r, m_w_a

            mix_out0 = torch.cat([TS_embeddings, I_embeddings], dim=-1)
            local_embeddings1, local_embeddings2, m_w_r, m_w_a = checkpoint(
                moe_projection_forward,
                mix_out0, TS_embeddings0, I_embeddings0, B, seq_len, num_features
            )
            # 重组为 dict，与 _forward_impl 的返回签名保持一致
            m_w = {0: m_w_r, 1: m_w_a}

        return local_embeddings1, m_w, loss_sc, local_embeddings2

    def compute_cl(self, I_emb, TS_emb, labels,num_features=1):
        if not self.training:
            return 0.0
        else:
            return self.cl_loss(I_emb, TS_emb, labels,num_features)

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-intensive modules."""
        # Vit encoder 已冻结，不需要 checkpointing
        # 只对 TS encoder 启用（self.ts_encoder 是 TS_Model, 内部 ts_encoder 是 TimeSeriesEncoder）
        if hasattr(self.ts_encoder, 'ts_encoder') and hasattr(self.ts_encoder.ts_encoder, 'gradient_checkpointing_enable'):
            self.ts_encoder.ts_encoder.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled for TS encoder")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.ts_encoder, 'ts_encoder') and hasattr(self.ts_encoder.ts_encoder, 'gradient_checkpointing_disable'):
            self.ts_encoder.ts_encoder.gradient_checkpointing_disable()
        print("[INFO] Gradient checkpointing disabled")

    def split_data(self,images, time_series, att_mask, labels, time_series_raw=None):
        """
        Split batched time-series data into chunks of length <= max_len along the time dimension (dim=1).

        Args:
            images:       [B, T, ...]
            time_series:  [B, T, F]
            att_mask:     [B, T]
            labels:       [B, T] or [B, T, 1]
            max_len:      int, maximum sequence length per chunk
            time_series_raw: Optional [B, T, F] tensor of un-normalized time series.
                             If provided, each chunk tuple includes a 5th element
                             ts_raw_chunk for ViCO rendering.

        Returns:
            List of tuples: [(img_chunk, ts_chunk, mask_chunk, label_chunk, ts_raw_chunk), ...]
                            When time_series_raw is None, ts_raw_chunk is omitted and
                            tuples have 4 elements (backward compatible).
        """
        B, T,_ = time_series.shape
        if T != labels.shape[1]:
            raise ValueError("Data and labels must have the same length in the first dimension.")

        if T % self.patch_size != 0:
            raise ValueError(f"Total length T={T} is not divisible by patch_size={self.patch_size}.")

        if self.MAX_L < self.patch_size:
            raise ValueError(f"MAX_length ({self.MAX_L}) must be >= patch_size ({self.patch_size}).")

        # Work in "patch units"
        num_patches = T // self.patch_size
        max_patches_per_chunk = self.MAX_L // self.patch_size  # floor division

        if max_patches_per_chunk == 0:
            raise ValueError("MAX_length is too small to fit even one patch.")

        # Minimum number of chunks needed
        min_splits = (num_patches + max_patches_per_chunk - 1) // max_patches_per_chunk

        # Now distribute num_patches into min_splits chunks as evenly as possible
        base_patches = num_patches // min_splits
        remainder = num_patches % min_splits

        chunks = []
        start_time = 0

        for i in range(min_splits):
            patches_in_this_chunk = base_patches + (1 if i < remainder else 0)
            chunk_length = patches_in_this_chunk * self.patch_size  # back to time steps
            end_time = start_time + chunk_length


            # Slice all tensors along time dimension (dim=1)
            img_chunk = images[:, :,:,start_time:end_time]          # [B, L, ...]
            ts_chunk = time_series[:, start_time:end_time,:]      # [B, L, F]
            mask_chunk = att_mask[:, start_time:end_time]       # [B, L]
            label_chunk = labels[:, start_time:end_time]        # [B, L] or [B, L, 1]

            if time_series_raw is not None:
                ts_raw_chunk = time_series_raw[:, start_time:end_time, :]
                chunks.append((img_chunk, ts_chunk, mask_chunk, label_chunk, ts_raw_chunk))
            else:
                chunks.append((img_chunk, ts_chunk, mask_chunk, label_chunk))
            start_time = end_time
        return chunks

    # =====================================================================
    # Unified Interface Methods
    # =====================================================================

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

    def split_sequence(self, images, time_series, att_mask, labels, time_series_raw=None):
        """
        封装 self.split_data 调用，用于长序列分块。

        Args:
            images: 折叠后的图像特征
            time_series: 时序数据 [B, L, C]
            att_mask: 注意力掩码 [B, L]
            labels: 标签 [B, L]
            time_series_raw: Optional [B, L, C] un-normalized time series for ViCO rendering.

        Returns:
            list of (sub_images, sub_ts, sub_att_mask, sub_labels[, sub_ts_raw]) chunks
        """
        return self.split_data(images, time_series, att_mask, labels, time_series_raw)

    def compute_loss(self, outputs, time_series, att_mask, labels, stage,
                     alpha_recon=0.05, cl_weight=0.1, balance_weight=0.01):
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

        # 专家平衡损失（Query Decoder 模式下 m_w=None，跳过）
        if m_w is None:
            loss_balance = torch.tensor(0.0, device=device)
        elif stage == 1:
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
            'loss_cl': loss_cl_val.item() if isinstance(loss_cl_val, torch.Tensor) else 0.0,
            'loss_balance': loss_balance.item() if isinstance(loss_balance, torch.Tensor) else 0.0,
            'logits': logits,
            'reconstruction': rec,
        }
