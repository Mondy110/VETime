#!/usr/bin/env python
"""
按维度拆分多变量数据集

将 vetime_multi_train_all_10000.pkl 拆分为按维度分组的独立文件
"""
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

def split_dataset_by_dim(input_path, output_dir):
    """按维度拆分数据集"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载数据集: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"总样本数: {len(data)}")

    # 按维度分组
    dim_data = defaultdict(list)
    for sample in tqdm(data, desc="分组中"):
        if 'attribute' in sample and 'num_features' in sample['attribute']:
            dim = sample['attribute']['num_features']
        else:
            # 从 time_series shape 推断
            ts = sample.get('time_series')
            if ts is not None and hasattr(ts, 'shape'):
                dim = ts.shape[-1] if len(ts.shape) > 1 else 1
            else:
                continue
        dim_data[dim].append(sample)

    # 保存各维度文件
    saved_dims = []
    for dim, samples in sorted(dim_data.items()):
        output_path = os.path.join(output_dir, f'dim_{dim}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        saved_dims.append({
            'path': output_path,
            'dim': dim,
            'count': len(samples)
        })
        print(f"  维度 {dim}: {len(samples)} 样本 -> {output_path}")

    return saved_dims

if __name__ == '__main__':
    input_path = 'dataset/vetime_multi_train_all_10000.pkl'
    output_dir = 'dataset/vetime_multi_split'

    saved_dims = split_dataset_by_dim(input_path, output_dir)

    print(f"\n拆分完成！共 {len(saved_dims)} 个维度文件")

    # 生成配置片段
    print("\n配置文件片段（datasets 列表）:")
    print("  datasets:")
    for item in saved_dims:
        rel_path = item['path'].replace('dataset/', './dataset/')
        print(f"    - path: {rel_path}")
        print(f"      dim: {item['dim']}")
