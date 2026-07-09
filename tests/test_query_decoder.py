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
