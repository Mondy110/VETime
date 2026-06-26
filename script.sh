# 单变量
/home/cjm/.conda/envs/tslib2/bin/accelerate launch train.py       --dataset_path ./dataset/vetime_train_all_150000.pkl       --ts_path checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth       --dataset_test_dir ./dataset/TSB-AD/Datasets/TSB-AD-U       --file_list ./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv       --model_name VETime-15w       --batch_size 32       --num_epochs 2       --learning_rate 5e-4       --weight_decay 1e-5       --num_workers 4 

export PYTHONHASHSEED=64
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

/home/cjm/.conda/envs/tslib2/bin/accelerate launch train.py \
    --dataset_path ./dataset/post_150000.pkl \
    --ts_path checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth \
    --dataset_test_dir ./dataset/TSB-AD/Datasets/TSB-AD-U \
    --file_list ./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv \
    --model_name VETime-post \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --seed 64

/home/cjm/.conda/envs/tslib2/bin/accelerate launch train.py \
    --dataset_path ./dataset/post_150000.pkl \
    --ts_path checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth \
    --dataset_test_dir ./dataset/TSB-AD/Datasets/TSB-AD-U \
    --file_list ./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv \
    --model_name VETime-post \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --use_vectorized_fold \
    --dynamic_batch \
    --val_ratio 0.05 \
    --val_mode split \
    --seed 64