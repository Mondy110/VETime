import torch


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
