#!/bin/bash
# 多变量顺序训练启动脚本 (4090 24GB 优化版)

export PYTHONHASHSEED=64
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

/home/cjm/.conda/envs/tslib2/bin/accelerate launch train.py \
    --config config/multivariate_config.yaml \
    --dataset_path ./dataset/vetime_multi_train_all_10000.pkl \
    --dataset_test_dir ./dataset/TSB-AD/Datasets/TSB-AD-M \
    --file_list ./dataset/TSB-AD/Datasets/File_List/TSB-AD-M.csv \
    --ts_path checkpoints/weight_ts/pretrain_checkpoint_best_multi.pth \
    --model_name VETime \
    --seed 64 \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 30 \
    --early_stop_patience 7 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --ts_finetune_type lora \
    --vision_name mae_visualize_base.pth \
    --output_file_path ./output/multivariate_result.json
