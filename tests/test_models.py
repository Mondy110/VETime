import pytest

def test_vetime_forward_output_shape():
    """验证 VETIME forward 输出形状正确。"""
    pytest.skip("需要预训练视觉编码器权重，标记为集成测试")

def test_vetime_compute_loss_returns_dict():
    """验证 compute_loss 返回正确格式的 dict。"""
    pytest.skip("需要完整模型实例，标记为集成测试")

def test_vetime_fold_images_delegates():
    """验证 fold_images 正确代理到 vit_encoder.fold_image。"""
    pytest.skip("需要视觉编码器，标记为集成测试")

def test_src_losses_import():
    """验证所有损失函数可从 src.losses 导入。"""
    from src.losses.contrastive import win_Contrastive_Loss
    from src.losses.balance import load_balance_loss
    from src.losses.anomaly import anomaly_detection_loss
    from src.losses.reconstruction import weighted_reconstruction_loss, masked_reconstruction_loss

def test_load_balance_loss():
    """验证 load_balance_loss 计算正确。"""
    import torch
    from src.losses.balance import load_balance_loss
    # Uniform distribution should give loss ≈ 1.0
    probs = torch.ones(8) / 8.0
    loss = load_balance_loss(probs)
    assert abs(loss.item() - 1.0) < 0.01, f"Expected ~1.0 for uniform, got {loss.item()}"

def test_anomaly_detection_loss_output():
    """验证 anomaly_detection_loss 输出格式。"""
    import torch
    from src.losses.anomaly import anomaly_detection_loss
    B, L, d = 2, 32, 256
    # Standalone function expects pre-computed logits [B, L, 2]
    logits = torch.randn(B, L, 2)
    labels = torch.zeros(B, L, dtype=torch.long)
    labels[0, 5:10] = 1  # 标记一些异常点
    loss, out_logits = anomaly_detection_loss(logits, labels)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert out_logits.shape == (B, L, 2), f"Expected logits (B,L,2), got {out_logits.shape}"

def test_weighted_reconstruction_loss():
    """验证 weighted_reconstruction_loss 输出格式。"""
    import torch
    from src.losses.reconstruction import weighted_reconstruction_loss
    B, L, C = 2, 16, 1
    # Without reconstruction_head: pass pre-reconstructed tensor
    reconstructed = torch.randn(B, L, C)
    original = torch.randn(B, L, C)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, :4] = False
    labels = torch.zeros(B, L, dtype=torch.long)
    loss, rec = weighted_reconstruction_loss(reconstructed, original, mask, labels)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert rec.shape == (B, L, C), f"Expected rec (B,L,C), got {rec.shape}"

def test_masked_reconstruction_loss():
    """验证 masked_reconstruction_loss 输出格式。"""
    import torch
    from src.losses.reconstruction import masked_reconstruction_loss
    B, L, C = 2, 16, 1
    reconstructed = torch.randn(B, L, C)
    original = torch.randn(B, L, C)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :8] = True
    loss, error = masked_reconstruction_loss(reconstructed, original, mask)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert error.shape == (B, L, C), f"Expected error (B,L,C), got {error.shape}"

def test_src_models_ts_encoder_import():
    """验证 ts_encoder 模块可从 src.models.ts_encoder 导入。"""
    from src.models.ts_encoder.config import TimeSeriesConfig, default_config_t
    from src.models.ts_encoder.encoding_utils import RotaryEmbedding, CustomTransformerEncoder

def test_src_models_vts_module_import():
    """验证 VTS 模块可从 src.models 导入。"""
    from src.models.vts_module import V_Attention, VTS_Alignment, M_moe, GatedTimeFrequencyFusion
