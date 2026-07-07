import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class VisualCrossAttention(nn.Module):
    """
    纯交叉注意力融合模块：用 VETime 时域特征 (Q) 查询 ViCO 频域特征 (K, V)。

    废除刚性对角假设，让时间域 Query 自适应决定关注哪些频域 patch。
    Attention Matrix 形状 [B, N_TS, 196]，表示每个时间点与所有频域 patch 的软对齐。
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, ffn_ratio: float = 4.0):
        """
        Args:
            d_model: 特征维度
            num_heads: 多头注意力头数
            dropout: Dropout 概率
            ffn_ratio: FFN 扩展比例（默认 4 倍）
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 纯交叉注意力：无对角偏置，无位置编码
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=d_model,
            vdim=d_model,
            dropout=dropout,
            batch_first=True
        )

        # Post-norm + 残差结构
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN: Linear -> GELU -> Dropout -> Linear
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model)
        )

    def forward(self, Q_visual: torch.Tensor, K_V_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q_visual: 时域 Query 特征 [B, N_TS, d]
            K_V_tokens: 频域 Key/Value tokens [B, 196, d]

        Returns:
            fused_visual: 融合后的时域特征 [B, N_TS, d]
        """
        # 纯交叉注意力：query=Q_visual, key=K_V_tokens, value=K_V_tokens
        # need_weights=False 启用 Flash Attention
        attn_out, _ = self.cross_attn(
            query=Q_visual,
            key=K_V_tokens,
            value=K_V_tokens,
            need_weights=False
        )

        # 残差 + Post-norm
        x = Q_visual + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN + 残差 + Post-norm
        x = x + self.dropout(self.ffn(x))
        fused_visual = self.norm2(x)

        return fused_visual


class GatedTimeFrequencyFusion(nn.Module):
    """
    门控残差交叉注意力融合模块：用 VETime 时域特征查询 ViCO 频域特征。

    核心改进：
    - 可学习的门控参数 alpha（初始化为 0.0），实现安全热启动
    - 训练初期自动回退到纯 VETime，随训练逐渐激活频域信息
    - 废除刚性对角假设，让时间域 Query 自适应决定关注哪些频域 patch

    Attention Matrix 形状 [B, N_TS, 196]，表示每个时间点与所有频域 patch 的软对齐。
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, ffn_ratio: float = 4.0):
        """
        Args:
            d_model: 特征维度
            num_heads: 多头注意力头数
            dropout: Dropout 概率
            ffn_ratio: FFN 扩展比例（默认 4 倍）
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 纯交叉注意力：无对角偏置，无位置编码
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=d_model,
            vdim=d_model,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        # 可学习的门控参数，初始化为 0.0 实现稳定热启动
        # 训练初期网络自然回退到纯 VETime，随训练逐渐激活频域信息
        self.alpha = nn.Parameter(torch.tensor([0.0]))

        # FFN: Linear -> GELU -> Dropout -> Linear
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        Q_VETime: torch.Tensor,
        K_ViCO: torch.Tensor,
        V_ViCO: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q_VETime: 时域对齐的视觉特征 [B, N_TS, d]
                      (经过 PTA unfold + mlp_i + I_att)
            K_ViCO:   频域原始 2D patch tokens [B, 196, d]
                      (ViCO 分支 MAE 输出，未对齐)
            V_ViCO:   频域原始 2D patch tokens [B, 196, d]
                      (与 K_ViCO 相同)
            key_padding_mask: Optional mask for ViCO tokens [B, 196]

        Returns:
            F_out: 融合后的时域特征 [B, N_TS, d]
        """
        # 不对称交叉注意力: [B, N_TS, d] x [B, 196, d] -> [B, N_TS, d]
        F_cross, _ = self.cross_attn(
            query=Q_VETime,
            key=K_ViCO,
            value=V_ViCO,
            key_padding_mask=key_padding_mask,
            need_weights=False  # 启用 Flash Attention
        )

        # 门控残差连接到 VETime 时序主干
        # alpha=0 时完全回退到 Q_VETime，alpha=1 时等权重融合
        F_mid = Q_VETime + self.alpha * self.dropout(F_cross)
        F_mid = self.layer_norm1(F_mid)

        # 非线性 FFN语义精炼
        F_out = F_mid + self.ffn(F_mid)
        F_out = self.layer_norm2(F_out)

        return F_out
