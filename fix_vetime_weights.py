#!/usr/bin/env python3
"""
修复 VETime.pth 中 ViT encoder 的 bias 参数问题。

问题：官方提供的 VETime.pth 中，ViT encoder 的所有 bias 参数被错误地保存为全 0。
解决：从原始 MAE 权重中恢复这些 bias 参数。

使用方法:
    python fix_vetime_weights.py

输出:
    checkpoints/VETime_fixed.pth - 修复后的权重文件
"""

import torch
import os

def main():
    print("=" * 60)
    print("VETime 权重修复脚本")
    print("=" * 60)

    # 路径配置
    mae_path = "checkpoints/weight_v/mae_visualize_base.pth"
    vetime_path = "checkpoints/VETime.pth"
    output_path = "checkpoints/VETime_fixed.pth"

    # 检查文件是否存在
    if not os.path.exists(mae_path):
        print(f"❌ 错误: 找不到 MAE 权重文件: {mae_path}")
        return
    if not os.path.exists(vetime_path):
        print(f"❌ 错误: 找不到 VETime 权重文件: {vetime_path}")
        return

    # 加载权重
    print("\n[1/4] 加载权重文件...")
    print(f"  - MAE 权重: {mae_path}")
    mae_ckpt = torch.load(mae_path, map_location="cpu", weights_only=False)
    mae_state = mae_ckpt["model"]

    print(f"  - VETime 权重: {vetime_path}")
    vetime_state = torch.load(vetime_path, map_location="cpu", weights_only=False)

    # 统计需要修复的参数
    print("\n[2/4] 检查需要修复的参数...")
    vit_prefix = "vit_encoder.encode_image.vision_model."
    vit_keys = [k for k in vetime_state.keys() if k.startswith(vit_prefix)]

    fixed_count = 0
    fixed_params = []

    for vetime_key in vit_keys:
        # 转换为 MAE 的 key 格式
        mae_key = vetime_key.replace(vit_prefix, "")

        if mae_key in mae_state:
            vetime_val = vetime_state[vetime_key]
            mae_val = mae_state[mae_key]

            # 检查是否是全 0 的参数（需要修复）
            if vetime_val.abs().sum().item() == 0 and mae_val.abs().sum().item() > 0:
                fixed_params.append(mae_key)
                fixed_count += 1

    print(f"  - ViT 总参数数: {len(vit_keys)}")
    print(f"  - 需要修复的参数: {fixed_count}")

    if fixed_count == 0:
        print("\n✅ 没有需要修复的参数，权重文件正常。")
        return

    print("\n  需要修复的参数列表:")
    for param in fixed_params[:10]:
        print(f"    - {param}")
    if len(fixed_params) > 10:
        print(f"    ... 还有 {len(fixed_params) - 10} 个参数")

    # 执行修复
    print("\n[3/4] 执行修复...")
    for vetime_key in vit_keys:
        mae_key = vetime_key.replace(vit_prefix, "")

        if mae_key in mae_state:
            vetime_val = vetime_state[vetime_key]
            mae_val = mae_state[mae_key]

            # 如果 VETime 参数全 0 但 MAE 参数不为 0，则修复
            if vetime_val.abs().sum().item() == 0 and mae_val.abs().sum().item() > 0:
                vetime_state[vetime_key] = mae_val.clone()
                print(f"  ✓ 修复: {mae_key}")

    # 验证修复结果
    print("\n[4/4] 验证修复结果...")
    verification_keys = [
        "blocks.0.attn.qkv.bias",
        "blocks.0.mlp.fc1.bias",
        "blocks.0.mlp.fc2.bias",
    ]

    all_verified = True
    for mae_key in verification_keys:
        vetime_key = vit_prefix + mae_key
        mae_val = mae_state[mae_key]
        vetime_val = vetime_state[vetime_key]

        is_same = torch.allclose(mae_val, vetime_val, atol=1e-6)
        status = "✅" if is_same else "❌"
        print(f"  {status} {mae_key}: MAE mean={mae_val.mean().item():.4f}, VETime mean={vetime_val.mean().item():.4f}")

        if not is_same:
            all_verified = False

    # 保存修复后的权重
    print(f"\n保存修复后的权重到: {output_path}")
    torch.save(vetime_state, output_path)

    if all_verified:
        print("\n" + "=" * 60)
        print("✅ 修复完成！")
        print("=" * 60)
        print(f"\n修复后的权重已保存到: {output_path}")
        print("\n使用方法:")
        print("  accelerate launch train.py \\")
        print("      --vetime_path checkpoints/VETime_fixed.pth \\")
        print("      ... 其他参数")
    else:
        print("\n⚠️ 部分参数验证失败，请检查修复结果。")


if __name__ == "__main__":
    main()
