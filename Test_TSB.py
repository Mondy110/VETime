# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import argparse
import logging
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    from TSB_AD.utils.slidingWindows import find_length_rank
except ModuleNotFoundError:  # optional benchmark dependency
    find_length_rank = None
from vetime.data.legacy_dataloader import create_random_mask
from vetime.data.pre_image import ts2image_Test
from vetime.metrics.tsb import get_metrics
from postprocess_runtime import (
    configure_postprocess_worker,
    postprocess_thread_limits,
    resolve_postprocess_workers,
)
from vetime.application.build_model import build_training_model
from vetime.infrastructure.logging import log_runtime_topology
from vetime.interfaces.cli import training_config_from_namespace

SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 用于 TSB-AD-U (单变量数据集): 排除非原生单变量数据集
PASS_LIST_UNI = [
    "Daphnet",
    "CATSv2",
    "SWaT",
    "LTDB",
    "TAO",
    "Exathlon",
    "MITDB",
    "MSL",
    "SMAP",
    "SMD",
    "SVDB",
    "OPP",
]


# 默认使用单变量过滤列表（保持向后兼容）
PASS_LIST = PASS_LIST_UNI


def _require_tsb_ad() -> None:
    if find_length_rank is None:
        raise RuntimeError(
            "TSB evaluation requires the optional 'tsb-ad' package. "
            "Install project requirements before running TSB_test."
        )

# 统计结果时使用的数据集分组
USE_LIST_UNI = [
    "IOPS",
    "MGAB",
    "NAB",
    "NEK",
    "Power",
    "SED",
    "Stock",
    "TODS",
    "WSD",
    "UCR",
    "YAHOO",
]


# 默认使用单变量统计列表（保持向后兼容）
USE_LIST = USE_LIST_UNI

DATA_INIT_SETTING = {
    "img_size": 224, 
    "T_sqrt":  False, 
    }

def dataloader_TSB(data, labels,data_setting,patch_size):
    
    time_series = np.array(data, dtype=float)
    lengths = time_series.shape[0]
    target_length = ((lengths + patch_size-1) // patch_size) * patch_size

    ts = (time_series - time_series.mean(axis=0, keepdims=True)) / (time_series.std(axis=0, keepdims=True) + 1e-2)
    
    img,period,pad_value = ts2image_Test(ts,patch_size,**data_setting) 

    image_inputs = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    period = torch.tensor(period, dtype=torch.float32).unsqueeze(0)
    pad_value = torch.tensor(pad_value, dtype=torch.float32).unsqueeze(0)    

    ts = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)
    labels=torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
    
    padded_ts = torch.nn.functional.pad(
            ts.transpose(1, 2),
            pad=(0, target_length-lengths),
            mode='constant',
            value=0.0
        ).transpose(1, 2)

    padded_labels = torch.nn.functional.pad(
            labels,
            pad=(0, target_length-lengths),
            mode='constant',
            value=-1
        )

    B, target_length, num_features = padded_ts.shape

    attention_mask= torch.ones(B, target_length, dtype=torch.bool)

    attention_mask[:, ts.shape[1]:] = False

    mask_time_series,mask  = create_random_mask(padded_ts, attention_mask,patch_size)

    return {
        'time_series': padded_ts,
        'mask_time_series':mask_time_series,
        'image': image_inputs,
        'mask': mask,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period':period,
        'p_value':pad_value,
    }


def TSB_test(
    model,
    args_test,
    data_setting=DATA_INIT_SETTING,
    device='cuda:0',
    dataset_setting=PASS_LIST,
    use_list=None,
    verbose=True,
    postprocess_workers=None,
    cpu_threads_per_worker=1,
):
    _require_tsb_ad()
    import os
    import time
    import pandas as pd
    import torch
    from tqdm import tqdm

    patch_size = model.patch_size
    target_dir = args_test.target_dir
    model_name = args_test.model_name
    file_list = args_test.file_list
    os.makedirs(target_dir, exist_ok=True)
    if verbose:
        print('Testing on TSB-AD datasets...')
    model.eval()
    model.to(device)
    runtime_log = []
    progress_bar = tqdm(file_list, desc=f"[Stage 1] Saving results for {model_name}")

    for filename in progress_bar:
        if any(filter_item in filename for filter_item in dataset_setting):
            continue

        output_path = os.path.join(target_dir, f'{filename.split(".")[0]}_results.pkl')

        file_path = os.path.join(args_test.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        datas = df.iloc[:, :-1].values.astype(float)
        labels_full = df['Label'].astype(int).to_numpy()

        train_index = int(filename.split('.')[0].split('_')[-3])
        data = datas[train_index:, :]
        labels = labels_full[train_index:]
        start_time = time.time()
        batch = {k: v.to(device) for k, v in dataloader_TSB(data, labels, data_setting, patch_size).items()}
        labels_tensor = batch["labels"]
        images = batch["image"]
        time_series = batch["time_series"]
        att_mask = batch["attention_mask"]
        with torch.no_grad():
            if len(labels) > model.MAX_L:
                data_splits = model.split_data(images, time_series, att_mask, labels_tensor)
                logits_list = []
                for data_part in data_splits:
                    img_part, ts_part, att_mask_p, label_part = data_part
                    images_folded, init_img_size = model.vit_encoder.fold_image(
                        img_part, batch['period'].cpu().numpy(), batch['p_value'], **data_setting
                    )
                    local_embeddings, _, _, _ = model(images_folded, ts_part, att_mask_p, init_img_size)
                    _, logits_part = model.anomaly_detection_loss(local_embeddings, label_part)
                    logits_list.append(logits_part)
                logits = torch.cat(logits_list, dim=1)
            else:
                images_folded,init_img_size = model.vit_encoder.fold_image(
                    images, batch['period'].cpu().numpy(), batch['p_value'], **data_setting
                )
                local_embeddings, _, _, _ = model(images_folded, time_series, att_mask, init_img_size)
                _, logits = model.anomaly_detection_loss(local_embeddings, labels_tensor)

        probs = torch.softmax(logits, dim=-1)[:, :, 1].detach().squeeze().cpu().numpy()
        labels_np = labels_tensor.squeeze().cpu().numpy()
        values = time_series.detach().squeeze().cpu().numpy()
        pd.DataFrame({
            'value': values.tolist(),
            'label': labels_np.tolist(),
            'anomaly_score': probs.tolist(),
        }).to_pickle(output_path)

        run_time = time.time() - start_time
        if verbose:
            print(f"Saved {output_path} (time: {run_time:.4f}s)")
        runtime_log.append({
            'filename': filename,
            'run_time_seconds': run_time
        })

        # 彻底清理所有中间变量
        del df, datas, labels_full, data, labels, batch, images, time_series, att_mask, labels_tensor
        del images_folded, local_embeddings, logits, probs, labels_np, values
        torch.cuda.empty_cache()
        import gc; gc.collect()

    log_df = pd.DataFrame(runtime_log)
    csv_save_path = os.path.join(os.getcwd(), f'runtime_log_{model_name}.csv')
    log_df.to_csv(csv_save_path, index=False)
    avg_f1 = TSB_test_parallel_postprocess(
        args_test, data_setting, dataset_setting, use_list, verbose=verbose,
        num_workers=postprocess_workers,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    return 1.0 - avg_f1  # 返回验证损失（1-F1，越小越好）


def _process_single_result_file(args):
    import gc
    result_path, filename, sliding_window = args
    try:
        df = pd.read_pickle(result_path)
        probs = np.array(df['anomaly_score'].tolist())
        labels = np.array(df['label'].tolist())
        labels_len = len(labels)
        del df

        pred_threshold = np.mean(probs) + 3 * np.std(probs)
        evaluation_result = get_metrics(probs, labels, slidingWindow=sliding_window, pred=probs > pred_threshold)

        del probs, labels, pred_threshold
        gc.collect()

        result = {
            'filename': filename,
            'length': labels_len,
            'metrics': evaluation_result,
        }
        del evaluation_result, labels_len
        return result
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None


def TSB_test_parallel_postprocess(
    args_test,
    data_setting=DATA_INIT_SETTING,
    dataset_setting=PASS_LIST,
    use_list=None,
    num_workers=None,
    cpu_threads_per_worker=1,
    verbose=True
):
    _require_tsb_ad()
    # 如果没有提供 use_list，使用全局的 USE_LIST
    if use_list is None:
        use_list = USE_LIST

    target_dir = args_test.target_dir
    file_list = args_test.file_list

    import gc
    tasks = []
    for filename in file_list:
        if any(filter_item in filename for filter_item in dataset_setting):
            continue
        result_path = os.path.join(target_dir, f'{filename.split(".")[0]}_results.pkl')
        if not os.path.exists(result_path):
            continue
        file_path = os.path.join(args_test.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        datas = df.iloc[:, 0:-1].values.astype(float)
        slidingWindow = find_length_rank(datas[:,0].reshape(-1, 1), rank=1)

        tasks.append((result_path, filename, slidingWindow))
        del df, datas, slidingWindow
    gc.collect()

    results = []
    ctx = mp.get_context('spawn')
    # Keep file-level parallelism independent from training DataLoader workers.
    cpu_cnt = mp.cpu_count()
    safe_workers = resolve_postprocess_workers(num_workers, cpu_count=cpu_cnt)
    if verbose:
        print(
            f"[INFO] CPU cores: {cpu_cnt}, using {safe_workers} post-process workers "
            f"with {cpu_threads_per_worker} PyTorch CPU thread(s) each"
        )
    # Spawned workers inherit these BLAS/OpenMP limits at interpreter startup.
    with postprocess_thread_limits(cpu_threads_per_worker):
        with ProcessPoolExecutor(
            max_workers=safe_workers,
            mp_context=ctx,
            initializer=configure_postprocess_worker,
            initargs=(cpu_threads_per_worker,),
        ) as executor:
            futures = [executor.submit(_process_single_result_file, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="[Stage 2] Post-processing"):
                res = future.result()
                if res:
                    results.append(res)
                del res, future
                gc.collect()

    write_csv = []
    col_w = None
    for res in results:
        row = [res['filename'], res['length'], 0] + list(res['metrics'].values())  # Time 设为 0
        write_csv.append(row)
        if col_w is None:
            col_w = ['file', 'Length', 'Time'] + list(res['metrics'].keys())

    w_csv = pd.DataFrame(write_csv, columns=col_w)

    use_data = use_list

    summary_rows = []
    for dataset_name in use_data:
        mask = w_csv['file'].str.contains(dataset_name, case=True, na=False)
        subset = w_csv[mask]
        if len(subset) == 0:
            continue
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        mean_values = subset[numeric_cols].mean(axis=0).round(4)
        summary_row = {'file': f"{dataset_name}_MEAN", 'Time': mean_values.get('Time', 0)}
        for col in col_w[2:]:
            summary_row[col] = mean_values.get(col, np.nan)
        summary_rows.append(summary_row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows, columns=col_w)
        final_csv = pd.concat([w_csv, summary_df], ignore_index=True)
    else:
        final_csv = w_csv

    timestamp = datetime.now().strftime("%m%d-%H")
    suffix = "_sq.csv" if data_setting["T_sqrt"] else "_P.csv"
    output_csv = f'{args_test.save_dir}/{args_test.model_name}_{data_setting["img_size"]}_{timestamp}{suffix}'
    os.makedirs(args_test.save_dir, exist_ok=True)
    final_csv.to_csv(output_csv, index=False)
    print(f"📊 Final results saved to: {output_csv}")

    # 计算所有数据集的平均F1作为验证指标
    avg_pr = 0.0
    if summary_rows and 'VUS-PR' in final_csv.columns:
        # 获取所有_MEAN行的平均F1
        mean_rows = final_csv[final_csv['file'].str.contains('_MEAN', na=False)]
        if not mean_rows.empty:
            avg_pr = mean_rows['VUS-PR'].mean()
            print(f"📈 Average VUS-PR across datasets: {avg_pr:.4f}")

    return avg_pr    

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience  
        self.verbose = verbose
        self.delta = delta      
        self.path = path         
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing on TSB-AD')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/TSB-AD/Datasets/TSB-AD-U')
    parser.add_argument('--model_name', default= 'VETime', type=str, help='Name of the model')
    parser.add_argument('--save_dir', type=str, default='./output/metrics/uni/')

    parser.add_argument('--output_file_path', default='./output/result.json',type=str, help='Path to the output file')
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--data_setting', type=str, default=DATA_INIT_SETTING, help='Device to use for evaluation')
    parser.add_argument('--ts_path', type=str, default='./checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth'
                        , help='TS_weight')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Strict v3 VETime model checkpoint')
    parser.add_argument('--vision_path', type=str, default='./checkpoints/weight_v',
                        help='MAE checkpoint directory')
    parser.add_argument('--vision_name', type=str, default='mae_visualize_base.pth'
                        , help='vision_weight')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='TSB metric post-process worker count (default: 4, capped by cpu_count-2)')
    parser.add_argument('--cpu_threads_per_worker', type=int, default=1,
                        help='PyTorch CPU threads allowed inside each TSB metric worker')

    args_test = parser.parse_args()
    # A full v3 model already includes the temporal encoder; its checkpoint is
    # therefore the sole initialization source for evaluation.
    if args_test.model_checkpoint:
        args_test.ts_path = None

    dataset_pass_list = PASS_LIST_UNI
    dataset_use_list = USE_LIST_UNI
    print(f"[INFO] 使用单变量 TSB-AD 过滤列表: {USE_LIST_UNI}")

    args_test.target_dir = os.path.join(args_test.save_dir, args_test.model_name)
    os.makedirs(args_test.target_dir, exist_ok = True)
    args_test.file_list = sorted(os.listdir(args_test.dataset_dir))

    args_test.output_file_path = args_test.output_file_path.replace('result.json', f'{args_test.model_name.replace("/", "-")}_result.json')

    device = args_test.device

    config = training_config_from_namespace(args_test)
    model = build_training_model(config)
    model.eval().to(device)
    log_runtime_topology(
        model,
        torch.device(device),
        "temporal_pretrain" if args_test.ts_path else "model_checkpoint" if args_test.model_checkpoint else "random_initialization",
    )

    TSB_test(
        model, args_test, args_test.data_setting, device,
        dataset_setting=dataset_pass_list, use_list=dataset_use_list,
        postprocess_workers=args_test.num_workers,
        cpu_threads_per_worker=args_test.cpu_threads_per_worker,
    )
