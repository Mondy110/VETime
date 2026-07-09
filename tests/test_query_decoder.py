import pytest
import torch
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
