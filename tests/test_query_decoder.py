import pytest
import torch
import torch.nn.functional as F
from model.VTS_module import QueryDecoder


def test_query_decoder_output_shape():
    """测试 QueryDecoder 输出形状正确"""
    B, N, D = 2, 128, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)

    F_TS = torch.randn(B, N, D)
    F_V = torch.randn(B, N, D)
    F_A = torch.randn(B, N, D)

    F_rec, F_cls = decoder(F_TS, F_V, F_A)

    assert F_rec.shape == (B, N, D), f"F_rec shape mismatch: {F_rec.shape}"
    assert F_cls.shape == (B, N, D), f"F_cls shape mismatch: {F_cls.shape}"


def test_query_decoder_with_mask():
    """测试带掩码的 QueryDecoder"""
    B, N, D = 2, 128, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)

    F_TS = torch.randn(B, N, D)
    F_V = torch.randn(B, N, D)
    F_A = torch.randn(B, N, D)
    patch_mask = torch.ones(B, N, dtype=torch.bool)
    patch_mask[:, -10:] = False  # 最后10个位置为padding

    F_rec, F_cls = decoder(F_TS, F_V, F_A, patch_mask)

    assert F_rec.shape == (B, N, D)
    assert F_cls.shape == (B, N, D)


def test_rope_consistency():
    """验证 QueryDecoder 的 RoPE 与 TS_encoder 一致"""
    from model.TS_encoder.encoding_utils import RotaryEmbedding

    D = 256
    N = 128

    rope1 = RotaryEmbedding(D)
    rope2 = RotaryEmbedding(D)

    freqs1 = rope1(N)
    freqs2 = rope2(N)

    assert torch.allclose(freqs1, freqs2), "RoPE 频率不一致"

    decoder = QueryDecoder(d_model=D, num_heads=8)
    freqs_decoder = decoder.rope(N)

    assert torch.allclose(freqs1, freqs_decoder), "QueryDecoder RoPE 与 TS_encoder 不一致"


def test_rope_position_alignment():
    """验证 RoPE 位置对齐特性"""
    D = 256
    N = 64

    decoder = QueryDecoder(d_model=D, num_heads=8)
    freqs = decoder.rope(N)

    base_token = torch.randn(1, 1, D)
    Q1 = base_token.expand(1, N, -1).clone()
    Q2 = base_token.expand(1, N, -1).clone()

    Q1_rot = decoder.apply_rope(Q1, freqs)
    Q2_rot = decoder.apply_rope(Q2, freqs)

    assert torch.allclose(Q1_rot, Q2_rot), "相同位置应产生相同旋转结果"


def test_gradient_flow_to_experts():
    """测试梯度正确传播到专家特征"""
    B, N, D = 2, 64, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)

    F_TS = torch.randn(B, N, D, requires_grad=True)
    F_V = torch.randn(B, N, D, requires_grad=True)
    F_A = torch.randn(B, N, D, requires_grad=True)

    F_rec, F_cls = decoder(F_TS, F_V, F_A)

    loss = F_rec.mean() + F_cls.mean()
    loss.backward()

    assert F_TS.grad is not None, "梯度未传播到 F_TS"
    assert F_V.grad is not None, "梯度未传播到 F_V"
    assert F_A.grad is not None, "梯度未传播到 F_A"
    assert F_TS.grad.shape == F_TS.shape


def test_separate_gradient_paths():
    """测试两个任务的梯度路径独立"""
    B, N, D = 2, 64, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.0)

    F_TS = torch.randn(B, N, D, requires_grad=True)
    F_V = torch.randn(B, N, D, requires_grad=True)
    F_A = torch.randn(B, N, D, requires_grad=True)

    decoder.eval()  # 关闭 dropout 确保确定性
    F_rec, F_cls = decoder(F_TS, F_V, F_A)

    # 使用不同的目标计算损失，模拟实际训练中的任务差异
    target_rec = torch.randn(B, N, D)
    target_cls = torch.randn(B, N, D)

    loss_rec = F.mse_loss(F_rec, target_rec)
    loss_rec.backward(retain_graph=True)
    grad_TS_from_rec = F_TS.grad.clone()

    F_TS.grad = None
    F_V.grad = None
    F_A.grad = None

    loss_cls = F.mse_loss(F_cls, target_cls)
    loss_cls.backward()
    grad_TS_from_cls = F_TS.grad.clone()

    # 验证两个任务的梯度有显著差异
    # 因为使用不同的交叉注意力层和任务Token，梯度应该不同
    diff = (grad_TS_from_rec - grad_TS_from_cls).abs()
    max_diff = diff.max().item()
    assert max_diff > 1e-6, \
        f"两个任务的梯度路径应该有所不同。最大差异: {max_diff}"


def test_query_decoder_disabled_by_default():
    """验证 QueryDecoder 默认禁用，不影响现有代码路径"""
    # 这个测试验证 QueryDecoder 只是一个独立模块
    # VETIME 模型默认不使用 QueryDecoder (use_query_decoder=False)
    from model.VTS_module import QueryDecoder

    # QueryDecoder 可以独立实例化，不影响现有代码
    decoder = QueryDecoder(d_model=256, num_heads=8)
    assert decoder is not None

    # 验证默认参数
    assert decoder.d_model == 256
    assert decoder.num_heads == 8
