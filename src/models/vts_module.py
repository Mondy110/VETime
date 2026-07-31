import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.ts_encoder.encoding_utils import RotaryEmbedding


class V_Attention(nn.Module):
    def __init__(self, dim_I, num_heads=8, dropout=0.1, ffn_ratio=4.0):
        super(V_Attention, self).__init__()
        self.dim_I = dim_I
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.cross_attn_a_to_b = nn.MultiheadAttention(
            embed_dim=dim_I,
            num_heads=num_heads,
            kdim=dim_I,
            vdim=dim_I,
            dropout=dropout,
            batch_first=True
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(dim_I, dim_I*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_I*4, dim_I),
            nn.LayerNorm(dim_I),
        )
        self.norm1_a = nn.LayerNorm(dim_I)

    def forward(self, feat_I, mask=None):
        B, N_a, _ = feat_I.shape
        out_a, _ = self.cross_attn_a_to_b(
            query=feat_I,
            key=feat_I,
            value=feat_I,
            key_padding_mask=~mask,
            need_weights=False  # 启用 Flash Attention，输出完全一致
        )
        out_a = feat_I + self.dropout(self.ffn_i(out_a))
        out_I = self.norm1_a(out_a)
        return out_I


class VTS_Alignment(nn.Module):
    def __init__(self, v_dim, TS_dim, embedding_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.TS_dim = TS_dim
        self.embedding_dim = embedding_dim

        t_dim2 = int(2 * self.TS_dim)
        self.mlp_i = nn.Sequential(
            nn.Linear(TS_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, TS_dim),
            nn.LayerNorm(TS_dim),
        )

        self.mlp_t = nn.Sequential(
            nn.Linear(TS_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, TS_dim),
            nn.LayerNorm(TS_dim),
        )

        self.cross_attn_a_to_b = nn.MultiheadAttention(
            embed_dim=TS_dim,
            num_heads=num_heads,
            kdim=TS_dim,
            vdim=TS_dim,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_b_to_a = nn.MultiheadAttention(
            embed_dim=TS_dim,
            num_heads=num_heads,
            kdim=TS_dim,
            vdim=TS_dim,
            dropout=dropout,
            batch_first=True
        )

        self.ffn_a = nn.Sequential(
            nn.Linear(TS_dim, t_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t_dim2, TS_dim)
        )

        self.ffn_b = nn.Sequential(
            nn.Linear(TS_dim, t_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t_dim2, TS_dim)
        )

        self.norm1_a = nn.LayerNorm(TS_dim)
        self.norm2_a = nn.LayerNorm(TS_dim)
        self.norm1_b = nn.LayerNorm(TS_dim)
        self.norm2_b = nn.LayerNorm(TS_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_I, feat_TS, mask=None):
        feat_I = self.mlp_i(feat_I)
        feat_TS = self.mlp_t(feat_TS)

        out_a, _ = self.cross_attn_a_to_b(
            query=feat_I,
            key=feat_TS,
            value=feat_TS,
            key_padding_mask=~mask,
            need_weights=False  # 启用 Flash Attention
        )
        out_a = feat_I + self.dropout(out_a)
        out_a = self.norm1_a(out_a)
        out_a = out_a + self.dropout(self.ffn_a(out_a))
        x_I = self.norm2_a(out_a)

        out_b, _ = self.cross_attn_b_to_a(
            query=feat_TS,
            key=feat_I,
            value=feat_I,
            key_padding_mask=~mask,
            need_weights=False  # 启用 Flash Attention
        )
        out_b = feat_TS + self.dropout(out_b)
        out_b = self.norm1_b(out_b)
        out_b = out_b + self.dropout(self.ffn_b(out_b))
        x_TS = self.norm2_b(out_b)

        return x_I, x_TS


class router(nn.Module):
    def __init__(self, dim, channel_num, num_tasks=2, topk=2, task_model='complex'):
        super().__init__()
        embed_dim = int(dim // 8)
        self.task_model = task_model
        self.l1 = nn.Linear(dim, embed_dim)
        self.l2 = nn.Linear(embed_dim, channel_num)
        self.topk = topk
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)

    def forward(self, x, task_id=None):
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        x = F.gelu(self.l1(x))

        if task_id is not None:
            task_id = int(task_id)
            task_id = torch.tensor(task_id, device=x.device, dtype=torch.long)
            task_bias = self.task_embedding(task_id)
            task_emb = task_bias.unsqueeze(0).expand(x.size(0), -1)
            x = x + task_emb

        logits = self.l2(x)
        topk_vals, topk_idx = torch.topk(logits, self.topk, dim=-1)
        topk_probs = torch.softmax(topk_vals, dim=-1).to(logits.dtype)
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, topk_idx, topk_probs)
        probs = probs.view(*original_shape[:-1], -1)
        return probs


class M_moe(nn.Module):
    """MMoE 软门控融合：每个任务拥有专属的 T/I/M 投影层，router 跨任务共享。

    任务映射（与原 checkpoint 语义保持一致）：
        task_id=0 -> reconstruction head (local_emb2)
        task_id=1 -> anomaly head        (local_emb1)
    """

    # 三种原材料的模态标识，与 task_proj 的子 ModuleDict key 对应
    _MODALITIES = ('T', 'I', 'M')

    def __init__(self, dst_feature_dims, num_tasks=2, topk=3):
        super(M_moe, self).__init__()
        self.dims = dst_feature_dims
        # 稠密软门控：topk=3 让 3 个模态全参与纯 softmax，无逐 token 丢弃
        self.Router = router(self.dims * 2, 3, topk=topk)
        # 跨任务共享的混合特征加工：把 cat([F_T, F_I]) 的 2*dim 映射回 dim
        self.mlp_m = nn.Sequential(
            nn.Linear(self.dims * 2, self.dims * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.dims * 2, self.dims),
        )

        # 任务专属投影层（按任务分组）：区别对待不同任务的特性
        self.task_proj = nn.ModuleDict()
        for t in range(num_tasks):
            task_dict = nn.ModuleDict()
            for m in self._MODALITIES:
                layers = []
                # 1. 任务层最上面的稳压层 (LayerNorm)
                layers.append(nn.LayerNorm(self.dims))

                # 2. 第一次线性变换
                layers.append(nn.Linear(self.dims, self.dims))

                # 3. 两个线性层之间的稳压层 (LayerNorm)
                layers.append(nn.LayerNorm(self.dims))

                # 4. 非线性激活
                layers.append(nn.GELU())

                # 5. 任务专用的正则化隔离
                # t=1 是分类任务(Anomaly)，需要 Dropout 防过拟合
                # t=0 是重构任务(Reconstruction)，严禁 Dropout 以保证输出数值极度平滑
                if t == 1:
                    layers.append(nn.Dropout(0.1))

                # 6. 第二次线性变换
                layers.append(nn.Linear(self.dims, self.dims))

                task_dict[m] = nn.Sequential(*layers)

            self.task_proj[str(t)] = task_dict

    def forward(self, F_M_raw, F_T, F_I, router_input, task_id):
        """对 F_T / F_I / F_M 做任务专属投影后，按 router 软门控加权求和。

        Args:
            F_M_raw: 混合原材料 (B, T, 2*dim)，先经共享 mlp_m 得到 F_M (B, T, dim)
            F_T:     时序特征 (B, T, dim)
            F_I:     图像特征 (B, T, dim)
            router_input: router 输入 (B, T, 2*dim)
            task_id: int (0=reconstruction, 1=anomaly)

        Returns:
            c_fusion: 任务专属的融合特征 (B, T, dim)
            m_w:      路由权重 (B, T, 3)，每行和为 1（供 load_balance_loss 使用）
        """
        # 1. 共享的混合特征（跨任务共用，避免重复计算）
        F_M = self.mlp_m(F_M_raw)

        # 2. 任务专属投影：解耦到各自任务的特征空间
        proj = self.task_proj[str(task_id)]
        F_T_p = proj['T'](F_T)
        F_I_p = proj['I'](F_I)
        F_M_p = proj['M'](F_M)

        # 3. 软门控加权（router 已对 3 路做 softmax，每行和为 1）
        m_w = self.Router(router_input, task_id)
        c_fusion = (
            F_T_p * m_w[..., 0:1] +
            F_I_p * m_w[..., 1:2] +
            F_M_p * m_w[..., 2:3]
        )
        return c_fusion, m_w


class QueryDecoder(nn.Module):
    """极致轻量化的共享任务解码器（修复优化版）。

    通过在长度轴级联任务 Queries，用单层共享的 Cross-Attention 与 FFN
    实现双任务在隐空间的自发分化与解耦，并原生支持块对齐 RoPE 与 Flash Attention 加速。
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # 1. 任务原型 Token（共享语义，形状 1x1xD）
        self.task_token_rec = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.task_token_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 2. 时间位置 Embedding（区分不同时间步，形状 1xmax_seq_lenxD）
        # 类似 ViT 的 position embedding，为每个时间位置提供独立语义
        self.temporal_query_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # RoPE 位置编码（复用现有）
        self.rope = RotaryEmbedding(d_model)

        # 3. 显式定义交叉注意力的四路投影层，以便在正确的位置插入 RoPE 算子
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Pre-LN 归一化器：用于在注意力机制计算前稳定 Query 主干流
        self.norm_cross = nn.LayerNorm(d_model)

        # 3. 共享前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        # Pre-LN 归一化器：用于在进入 FFN 映射前稳定特征
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """应用 RoPE 位置编码。与 VETime 原文投影后、分头前的旋转逻辑完全对齐。"""
        B, seq_len, embed_dim = x.shape
        assert embed_dim == self.d_model, "Embedding dimension mismatch"
        assert freqs.shape == (seq_len, embed_dim // 2), f"freqs shape mismatch: {freqs.shape}"

        # 拆分为复数对进行旋转
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)
        sin = freqs.sin().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)

        x_rot = torch.stack(
            [
                x_[..., 0] * cos - x_[..., 1] * sin,
                x_[..., 0] * sin + x_[..., 1] * cos,
            ],
            dim=-1
        )
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, F_TS: torch.Tensor, F_V: torch.Tensor, F_A: torch.Tensor, patch_mask: torch.Tensor = None) -> tuple:
        B, N, D = F_TS.shape

        # 1. 专家库级联（长 3N）
        K_expert = torch.cat([F_TS, F_V, F_A], dim=1)  # (B, 3N, D)
        V_expert = K_expert

        # 2. 动态构建 Query（任务原型 + 时间位置）
        assert N <= self.max_seq_len, f"序列长度 N={N} 超过 max_seq_len={self.max_seq_len}"

        # 时间位置 embedding 切片：(1, N, D)
        time_emb = self.temporal_query_embedding[:, :N, :]

        # 重建 Query = 任务原型 + 时间位置
        Q_rec_base = self.task_token_rec.expand(B, N, -1) + time_emb  # (B, N, D)

        # 分类 Query = 任务原型 + 时间位置（共享同一个 time_emb）
        Q_cls_base = self.task_token_cls.expand(B, N, -1) + time_emb  # (B, N, D)

        Q_combined = torch.cat([Q_rec_base, Q_cls_base], dim=1)  # (B, 2N, D)

        # ==========================================
        # 【Pre-LN 核心修改 1】：在执行 Q 投影前先进行层归一化
        # ==========================================
        Q_normed = self.norm_cross(Q_combined)  # (B, 2N, D)

        # 3. 执行任务专属线性投影（注意：q 的输入换成了归一化后的 Q_normed）
        q = self.q_proj(Q_normed)  # (B, 2N, D)
        k = self.k_proj(K_expert)  # (B, 3N, D)
        v = self.v_proj(V_expert)  # (B, 3N, D)

        # 4. 获取并组装"块对齐"的旋转频率 (Block-Aligned RoPE)
        freqs_N = self.rope(N)
        freqs_2N = torch.cat([freqs_N, freqs_N], dim=0)
        freqs_3N = torch.cat([freqs_N, freqs_N, freqs_N], dim=0)

        # 在投影之后、拆分多头之前，同步对 Q 和 K 灌入几何旋转特征
        q_rot = self.apply_rope(q, freqs_2N)
        k_rot = self.apply_rope(k, freqs_3N)

        # 5. 变形为标准多头注意力张量形状
        q_rot = q_rot.view(B, 2 * N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 2N, h_d)
        k_rot = k_rot.view(B, 3 * N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 3N, h_d)
        v = v.view(B, 3 * N, self.num_heads, self.head_dim).transpose(1, 2)         # (B, H, 3N, h_d)

        # 6. 构造变长填充掩码 (Padding Mask)
        attn_mask = None
        if patch_mask is not None:
            # 专家库包含 3 个专家，对应的 Key 掩码需要横向复制 3 遍：(B, 3N)
            kv_mask = patch_mask.repeat(1, 3)
            # 转换为 PyTorch 软注意力算子标准的布尔广播形状：(B, 1, 1, 3N)
            # 原文中 True 表示有效，这里通过 unsqueeze 适配标准的掩码消融
            attn_mask = kv_mask.unsqueeze(1).unsqueeze(2)

        # 7. 调用原生的统一缩放点积注意力，全额释放 Flash Attention 2 内核算力
        # 计算复杂度仅为稳定的线性增长 2N * 3N = 6N^2，在同一个物理熔炉中自发分化
        y = F.scaled_dot_product_attention(
            query=q_rot,
            key=k_rot,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # 8. 恢复序列形状并应用输出投影
        y = y.transpose(1, 2).contiguous().view(B, 2 * N, D)
        F_attn = self.out_proj(y)
        
        # ==========================================
        # 【Pre-LN 核心修改 2】：直接进行残差相加，外部不再包裹 LayerNorm
        # 此时残差路径上的 Q_combined 保持原汁原味，梯度畅通无阻
        # ==========================================
        F_combined = Q_combined + F_attn  # (B, 2N, D)

        # ==========================================
        # 【Pre-LN 核心修改 3】：共享 FFN 非线性映射
        # 同样采取“先归一化，再进子层，最后直接残差相加”的策略
        # ==========================================
        F_combined = F_combined + self.ffn(self.norm_ffn(F_combined))

        # 10. 输出端无损剥离（Token-level Decoupling）
        F_rec_out = F_combined[:, :N, :]  # 截取前 N 个 Token 作为重构专用表征
        F_cls_out = F_combined[:, N:, :]  # 截取后 N 个 Token 作为点级分类专用表征

        return F_rec_out, F_cls_out


class FrequencyGuidedVisualAdapter(nn.Module):
    '''
    频域引导视觉适配器 (Frequency-Guided Visual Adapter, FGVA)

    设计哲学:
    Frequency-Conditioned Residual Adaptation (频域条件化残差适配)
    - Q (时域特征) 作为 Identity Pathway 主干表示；
    - K/V (频域视觉特征) 作为 Conditional Context 提供修正信息；
    - FFN 在 (Q + Attn) 的隐式融合空间下生成 state-dependent 的残差 correction；
    - 外部通过矢量 LayerScale alpha 达成全局可控的知识注入。

    核心参数说明:
    - d_model: 特征维度
    - num_heads: 注意力头数
    - dropout: 适配器残差 Dropout 比例
    - ffn_ratio: 适配器 FFN 隐藏层膨胀倍数 (推荐 2.0，兼顾表达力与防过拟合)
    - init_scale: LayerScale 初始缩放系数 (推荐 1e-3，提供近乎恒等映射与温和梯度)
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
        init_scale: float = 1e-3
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 严格整除"

        # 1. 内部 Cross-Attention Projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 2. 归一化层 (Pre-LN 架构)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)  # K/V 共享 Memory 归一化
        self.norm_ffn = nn.LayerNorm(d_model)

        # 3. 内部条件精炼 FFN (单 Dropout 正则化)
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.ffn_dropout = nn.Dropout(dropout)

        # 4. Channel-wise LayerScale 向量门控 (1e-3 初始化)
        self.alpha = nn.Parameter(torch.ones(d_model) * init_scale)

    def forward(
        self,
        Q_VETime: torch.Tensor,
        K_ViCO: torch.Tensor,
        V_ViCO: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = Q_VETime.size(0)
        q_len = Q_VETime.size(1)
        k_len = K_ViCO.size(1)

        # ---------------------------------------------------------
        # Step A: 交叉注意力提取频域条件特征 (Pre-LN)
        # ---------------------------------------------------------
        normed_Q = self.norm_q(Q_VETime)
        normed_K = self.norm_kv(K_ViCO)
        normed_V = self.norm_kv(V_ViCO)

        q = self.q_proj(normed_Q).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_K).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_V).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~(key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)

        # 调用 Flash Attention 兼容的高性能注意力函数 (禁用内部 Dropout 保持条件确定性)
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        # ---------------------------------------------------------
        # Step B: 隐式融合空间 + 条件残差提炼
        # ---------------------------------------------------------
        # 隐式融合：将频域条件特征叠加至时域基线上
        implicit_fusion = Q_VETime + attn_out

        # FFN 结合时域基线与频域条件生成最终修正量
        normed_fusion = self.norm_ffn(implicit_fusion)
        delta = self.ffn(normed_fusion)
        delta = self.ffn_dropout(delta)

        # ---------------------------------------------------------
        # Step C: External LayerScale 显式广播门控注入
        # ---------------------------------------------------------
        # 将 alpha (d_model,) 显式扩展为 (1, 1, d_model) 以匹配 (B, L, D) 维度
        alpha_reshaped = self.alpha.view(1, 1, -1)
        F_out = Q_VETime + alpha_reshaped * delta

        return F_out

