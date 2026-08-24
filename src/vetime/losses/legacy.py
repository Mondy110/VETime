import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class win_ContrastiveLoss_init(nn.Module):
    """
    N-level (token/time-step level) contrastive loss.
    Assumes feat_modality1 and feat_modality2 are aligned position-wise:
        feat_modality1[b, i] <-> feat_modality2[b, i]  (positive pair)
    All other pairs in the batch are treated as negatives.

    Inputs:
        feat_modality1: (B, N, D)
        feat_modality2: (B, N, D)
    Output:
        scalar loss (mean of symmetric InfoNCE)
    """
    def __init__(self, dim=512,temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.mlp1=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
    def _find_segments(self, labels: torch.Tensor):
        """
        labels: (B, N) or (N,) of 0/1, where 1 = foreground / event
        Returns: list of list of (start, end) tuples for each batch
        """
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        segments = []
        for lbl in labels:
            segs = []
            i = 0
            while i < len(lbl):
                if lbl[i] == 1:
                    start = i
                    while i < len(lbl) and lbl[i] == 1:
                        i += 1
                    segs.append((start, i))
                else:
                    i += 1
            segments.append(segs)
        return segments

    def intra_loss(self, z1, z2, label, valid, start, end):
        """
        z1, z2      : (N, D)   单样本(单变量)的两个模态表征（已 L2 归一化）
        label       : (N,)     前景(1)/背景(0) patch 标签
        valid       : (N,) bool 有效掩码，False 表示该 patch 为纯 padding
        start, end  : 前景段 [start, end)
        """
        N, D = z1.shape
        L = end - start

        # === Anchor ===
        anchor = z2[start:end].mean(dim=0)  # (D,)  前景段本身必为有效，无需 mask

        # === Positive (same positions, from z1) ===
        positive = z1[start:end].mean(dim=0)     # (D,)

        # === Negative candidates ===
        Neg_candidates = []

        # (1) Left adjacent single point: 必须背景 且 有效
        if start - 1 >= 0 and label[start - 1] == 0 and valid[start - 1]:
            Neg_candidates.append(z1[start - 1])

        # (2) Right adjacent single point: 必须背景 且 有效
        if end < N and label[end] == 0 and valid[end]:
            Neg_candidates.append(z1[end])

        # (3) Left window: [start - L, start)  仅对窗口内有效位置求均值
        left_win_start = max(0, start - L)
        left_win_end = start
        if left_win_end > left_win_start:
            vmask = valid[left_win_start:left_win_end]
            if vmask.any():
                Neg_candidates.append(z1[left_win_start:left_win_end][vmask].mean(dim=0))

        # (4) Right window: [end, end + L)  仅对窗口内有效位置求均值
        right_win_start = end
        right_win_end = min(N, end + L)
        if right_win_end > right_win_start:
            vmask = valid[right_win_start:right_win_end]
            if vmask.any():
                Neg_candidates.append(z1[right_win_start:right_win_end][vmask].mean(dim=0))

        # 如果没有找到有效负样本（整窗都是异常/无效），跳过这个 segment
        if len(Neg_candidates) == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True), left_win_start, right_win_end

        # Stack negatives: (K, D)
        negative = torch.stack(Neg_candidates)  # (K, D)
        K = negative.shape[0]

        # Compute similarities
        neg_sims = torch.matmul(negative, anchor.unsqueeze(-1)).squeeze(-1)  # (K,)
        pos_sim = torch.dot(anchor, positive).unsqueeze(0)                     # (1,)

        # InfoNCE：正样本置于 index 0。用 logsumexp 替代手动 exp/log，数值更稳。
        logits = torch.cat([pos_sim, neg_sims], dim=0) / self.temperature      # (K+1,)
        loss = -logits[0] + torch.logsumexp(logits, dim=0)

        return loss, left_win_start, right_win_end

    def inter_loss(self, z1, z2, label, valid, start, end, Neg_candidates_start):
        """
        start, end 这里是「扩展窗口」[win_start, win_end)（由 intra_loss 返回）。
        按设计保留扩展窗口跨度以维持上下文趋势感知；仅在窗口内对纯 padding
        位置做掩码均值（窗口范围不变，但 padding 点不参与聚合）。
        """
        N, D = z1.shape
        L = end - start

        # === Anchor / Positive: 窗口内有效位置的均值 ===
        vmask = valid[start:end]
        if vmask.any():
            anchor = z2[start:end][vmask].mean(dim=0)   # (D,)
            positive = z1[start:end][vmask].mean(dim=0) # (D,)
        else:
            anchor = z2[start:end].mean(dim=0)
            positive = z1[start:end].mean(dim=0)

        # === Negative candidates ===
        # 候选窗口已由 _sample_bg_windows_fast 校验过「全背景 且 全有效」，无需再 mask。
        Neg_candidates = [z1[i:i+L].mean(dim=0) for i in Neg_candidates_start]
        if len(Neg_candidates) == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # Stack negatives: (K, D)
        negative = torch.stack(Neg_candidates)  # (K, D)
        K = negative.shape[0]

        # Compute similarities
        neg_sims = torch.matmul(negative, anchor.unsqueeze(-1)).squeeze(-1)  # (K,)
        pos_sim = torch.dot(anchor, positive).unsqueeze(0)                     # (1,)

        # InfoNCE：正样本置于 index 0。用 logsumexp 替代手动 exp/log。
        logits = torch.cat([pos_sim, neg_sims], dim=0) / self.temperature      # (K+1,)
        loss = -logits[0] + torch.logsumexp(logits, dim=0)

        return loss


    def _sample_bg_windows_fast(self, label, valid, start, end, max_samples=10, max_attempts=50):
        N = label.shape[0]
        L = end - start

        Neg_candidates = []
        attempts = 0

        left_start, left_end = 0, max(0, start - L + 1)          # [0, start - L + 1)
        right_start, right_end = end, max(end, N - L + 1)       # [end, N - L + 1)

        total_left = left_end - left_start
        total_right = right_end - right_start

        if total_left + total_right == 0:
            return []

        while len(Neg_candidates) < max_samples and attempts < max_attempts:
            if total_left > 0 and (total_right == 0 or random.random() < total_left / (total_left + total_right)):
                i = random.randint(left_start, left_end - 1)
            else:
                i = random.randint(right_start, right_end - 1)
            # 负样本窗口必须「全为背景(0)」且「全部有效（无 padding）」
            if not ((label[i:i+L] == 0) & valid[i:i+L]).all():
                attempts += 1
                continue
            if any(abs(i - j) < L for j in Neg_candidates):
                attempts += 1
                continue
            Neg_candidates.append(i)
            attempts += 1

        return Neg_candidates

class win_Contrastive_Loss(win_ContrastiveLoss_init):
    """
    N-level (token/time-step level) contrastive loss.
    Assumes feat_modality1 and feat_modality2 are aligned position-wise:
        feat_modality1[b, i] <-> feat_modality2[b, i]  (positive pair)
    All other pairs in the batch are treated as negatives.

    Inputs:
        feat_modality1: (B, N, D)
        feat_modality2: (B, N, D)
    Output:
        InfoNCE loss
    """
    def __init__(self, dim=512,temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.mlp1=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
        )
    def forward(self, f1: torch.Tensor, f2: torch.Tensor,labels0,num_f=1) -> torch.Tensor:
        """
        feat_modality1: (B, N, D) 为ts
        feat_modality2: (B, N, D) 为vision
        """
        B, N, D = f1.shape
        assert f2.shape == (B, N, D), "视觉特征和时间序列特征维度必须完全一致！"

        N_seq = N // num_f

        # 前向传播时是 view(B, num_features * num_patches, D)
        # 所以我们还原回 (Batch, 变量数, 时间步数, 特征维度)
        f1_unflatten = f1.view(B, num_f, N_seq, D)
        f2_unflatten = f2.view(B, num_f, N_seq, D)

        # 将 Batch 和 变量(num_f) 维度合并！
        f1_aligned = f1_unflatten.view(B * num_f, N_seq, D)
        f2_aligned = f2_unflatten.view(B * num_f, N_seq, D)

        # ===== 三态掩码：前景(1) / 背景(0) / 无效(纯 padding) =====
        # 点级 padding 被标记为 -1。原代码用 sum>0 判前景会被 -1 拉偏（混合 patch
        # 里 padding 点数 >= 前景点数时前景被误判成背景），且纯 padding patch 会被
        # 当成合法背景混入负样本池。这里把「是否 padding」与「是否前景」解耦：
        mask        = labels0.view(B, N_seq, -1)                              # (B, N_seq, P)
        pad_point   = (mask == -1)                                           # 哪些点是 padding
        valid_patch = ~pad_point.all(dim=-1)                                 # patch 级有效：非纯 padding
        # 前景：存在「真实标注为 1」的点，且 patch 本身有效（忽略 -1 不参与判定）
        labels      = ((mask == 1) & ~pad_point).any(dim=-1) & valid_patch   # (B, N_seq) bool

        labels_aligned = labels.repeat_interleave(num_f, dim=0)              # (B*num_f, N_seq)
        valid_aligned  = valid_patch.repeat_interleave(num_f, dim=0)         # (B*num_f, N_seq)

        assert labels_aligned.shape == (B * num_f, N_seq), \
            f"标签形状错误: {labels_aligned.shape}"

        # Project features
        z10 = self.mlp1(f1_aligned)  # (B, N, D//4)
        z20 = self.mlp2(f2_aligned)  # (B, N, D//4)

        # L2 normalize
        z1s = F.normalize(z10, dim=-1)
        z2s = F.normalize(z20, dim=-1)

        total_loss = 0.0

        segments_per_batch = self._find_segments(labels_aligned)
        num_segments = 0
        for b in range(B * num_f):
            z1 = z1s[b]
            z2 = z2s[b]
            label = labels_aligned[b]
            valid = valid_aligned[b]
            for (start, end) in segments_per_batch[b]:
                L = end - start
                if L <= 0:
                    continue
                intra1, win_start, win_end = self.intra_loss(z1, z2, label, valid, start, end)
                intra2, _, _ = self.intra_loss(z2, z1, label, valid, start, end)

                inter1 = 0
                inter2 = 0

                if L > 1:
                    cand_strat = self._sample_bg_windows_fast(label, valid, win_start, win_end, 10)
                    inter1 = self.inter_loss(z1, z2, label, valid, win_start, win_end, cand_strat)
                    inter2 = self.inter_loss(z2, z1, label, valid, win_start, win_end, cand_strat)
                num_segments += 1
                # 刻意保留非对称相加：仅取 intra1 + inter2，建立单向对齐约束，
                # 避免反向梯度干扰主特征空间。intra2 / inter1 照常计算但有意不进入总损失。
                total_loss += intra1 + inter2

        if num_segments == 0:
            return torch.zeros((), device=f1.device)

        return total_loss / num_segments

def load_balance_loss(probs, top_k=3):
    """浓度损失（importance concentration loss），适配稠密软门控（topk=全部 expert）。

    在稠密软门控（router 对所有 expert 做 softmax、无逐 token 丢弃）下，传统
    Switch-Transformer 式的 importance·load 损失会退化为常数（每个 token 都选满
    全部 expert → load 恒等于 1/C → loss 恒等于 1，零梯度）。这里改用 importance
    浓度损失：

        loss = C * sum_c (importance_c)^2

    - importance_c = 所有 token 对 expert c 的平均路由概率
    - 当三模态均匀使用时 importance_c = 1/C → loss = C * C * (1/C)^2 = 1
    - 当某模态被独占时 importance = [1,0,0] → loss = C
    - 最小值在均匀分布处取到，梯度推动 router 均衡使用所有模态

    Args:
        probs: (B, C) 或 (B, T, C)，每个 token 上的概率分布（每行和为 1）
        top_k: 仅保留以兼容旧签名，稠密模式下不参与计算

    Returns:
        标量 loss（>= 1，均匀时 = 1）
    """
    if not isinstance(probs, torch.Tensor):
        return 0.0
    probs = probs.view(-1, probs.size(-1))   # (N, C)
    importance = probs.mean(dim=0)           # (C,)
    C = probs.size(-1)
    # 均匀时 = 1，集中时 > 1，鼓励三模态均衡使用
    return C * (importance ** 2).sum()
