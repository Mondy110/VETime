"""TSB-AD 基准测试引擎。

从 Test_TSB.py 抽离的评估逻辑，封装为 Evaluator 类。
保留与原始 TSB_test 完全一致的推理行为，包括：
- dataloader_TSB 的标准化方式（std + 1e-2）
- 长序列分块推理
- 并行后处理与指标汇总
"""

import gc
import os
import time
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.logger import get_logger
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_Test

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量：与 Test_TSB.py 保持一致
# ---------------------------------------------------------------------------

DATA_INIT_SETTING = {"img_size": 224, "T_sqrt": False}

# 单变量数据集过滤
PASS_LIST_UNI = [
    "Daphnet", "CATSv2", "SWaT", "LTDB", "TAO",
    "Exathlon", "MITDB", "MSL", "SMAP", "SMD", "SVDB", "OPP",
]

# 多变量数据集过滤
PASS_LIST_MULTI = [
    "CATSv2", "CreditCard", "Daphnet", "Exathlon", "GECCO",
    "GHL", "Genesis", "LTDB", "MITDB", "OPPORTUNITY", "SVDB", "TAO",
]

USE_LIST_UNI = [
    "IOPS", "MGAB", "NAB", "NEK", "Power",
    "SED", "Stock", "TODS", "WSD", "UCR", "YAHOO",
]

USE_LIST_MULTI = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]

# 默认使用单变量列表
PASS_LIST = PASS_LIST_UNI
USE_LIST = USE_LIST_UNI


# ---------------------------------------------------------------------------
# dataloader_TSB：与 Test_TSB.py 完全一致的推理数据准备
# ---------------------------------------------------------------------------

def dataloader_TSB(
    data: np.ndarray,
    labels: np.ndarray,
    data_setting: dict,
    patch_size: int,
) -> Dict[str, torch.Tensor]:
    """推理时数据准备，与 Test_TSB.py 中同名函数行为完全一致。

    注意：标准化使用 std + 1e-2（与训练的 1e-4 不同），这是为了向后兼容，
    切勿修改。
    """
    time_series = np.array(data, dtype=float)
    lengths = time_series.shape[0]
    target_length = ((lengths + patch_size - 1) // patch_size) * patch_size

    # 关键：1e-2 与训练时 1e-4 不同，必须保留
    ts = (time_series - time_series.mean(axis=0, keepdims=True)) / (
        time_series.std(axis=0, keepdims=True) + 1e-2
    )

    img, period, pad_value = ts2image_Test(ts, patch_size, **data_setting)

    image_inputs = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    period = torch.tensor(period, dtype=torch.float32).unsqueeze(0)
    pad_value = torch.tensor(pad_value, dtype=torch.float32).unsqueeze(0)

    ts = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)

    padded_ts = F.pad(
        ts.transpose(1, 2),
        pad=(0, target_length - lengths),
        mode="constant",
        value=0.0,
    ).transpose(1, 2)

    padded_labels = F.pad(
        labels,
        pad=(0, target_length - lengths),
        mode="constant",
        value=-1,
    )

    B, target_length, num_features = padded_ts.shape

    attention_mask = torch.ones(B, target_length, dtype=torch.bool)
    attention_mask[:, ts.shape[1]:] = False

    mask_time_series, mask = create_random_mask(padded_ts, attention_mask, patch_size)

    return {
        "time_series": padded_ts,
        "mask_time_series": mask_time_series,
        "image": image_inputs,
        "mask": mask,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "period": period,
        "p_value": pad_value,
    }


# ---------------------------------------------------------------------------
# 并行后处理 worker（与 Test_TSB._process_single_result_file 一致）
# ---------------------------------------------------------------------------

def _process_single_result_file(args):
    """单个结果文件的指标计算 worker。"""
    result_path, filename, sliding_window = args
    try:
        df = pd.read_pickle(result_path)
        probs = np.array(df["anomaly_score"].tolist())
        labels = np.array(df["label"].tolist())
        labels_len = len(labels)
        del df

        pred_threshold = np.mean(probs) + 3 * np.std(probs)
        from evaluation.metrics import get_metrics
        evaluation_result = get_metrics(
            probs, labels, slidingWindow=sliding_window, pred=probs > pred_threshold
        )

        del probs, labels, pred_threshold
        gc.collect()

        result = {
            "filename": filename,
            "length": labels_len,
            "metrics": evaluation_result,
        }
        del evaluation_result, labels_len
        return result
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """TSB-AD 基准测试引擎。

    用法::

        evaluator = Evaluator(cfg, model)
        val_loss = evaluator.evaluate_benchmark(
            dataset_dir=...,
            file_list=...,
            target_dir=...,
            save_dir=...,
        )
    """

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        accelerator=None,
        data_setting: Optional[dict] = None,
    ):
        """
        Args:
            cfg: OmegaConf DictConfig 配置对象。
            model: 已加载权重的 VETIME 模型。
            accelerator: 可选的 HuggingFace Accelerator 实例。
            data_setting: 数据初始化参数，默认 DATA_INIT_SETTING。
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator
        self.device = next(model.parameters()).device
        self.data_setting = data_setting or dict(DATA_INIT_SETTING)

        # 从模型获取 patch_size
        self.patch_size = model.patch_size

        # 从 cfg 提取常用字段
        model_name = getattr(cfg, "model", None)
        if model_name is not None:
            self.model_name = getattr(model_name, "model_name", "VETime")
        else:
            self.model_name = "VETime"

    # ------------------------------------------------------------------
    # 单数据集推理
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """对单个数据集进行推理，返回异常分数。

        Args:
            data: 时序数据 (L, C)。
            labels: 标签 (L,)。
            save_path: 若提供，将结果保存为 pkl。
            verbose: 是否打印耗时。

        Returns:
            dict: 包含 probs, labels_np, values, run_time 等字段。
        """
        self.model.eval()
        patch_size = self.patch_size
        data_setting = self.data_setting

        start_time = time.time()

        batch = {k: v.to(self.device) for k, v in dataloader_TSB(data, labels, data_setting, patch_size).items()}
        labels_tensor = batch["labels"]
        images = batch["image"]
        time_series = batch["time_series"]
        att_mask = batch["attention_mask"]

        with torch.no_grad():
            if len(labels) > self.model.MAX_L:
                data_splits = self.model.split_sequence(images, time_series, att_mask, labels_tensor)
                logits_list = []
                for data_part in data_splits:
                    img_part, ts_part, att_mask_p, label_part = data_part
                    images_folded, init_img_size = self.model.fold_images(
                        img_part,
                        batch["period"].cpu().numpy(),
                        batch["p_value"],
                        **data_setting,
                    )
                    local_embeddings, _, _, _ = self.model(images_folded, ts_part, att_mask_p, init_img_size)
                    _, logits_part = self.model.anomaly_detection_loss(local_embeddings, label_part)
                    logits_list.append(logits_part)
                logits = torch.cat(logits_list, dim=1)
            else:
                images_folded, init_img_size = self.model.fold_images(
                    images,
                    batch["period"].cpu().numpy(),
                    batch["p_value"],
                    **data_setting,
                )
                local_embeddings, _, _, _ = self.model(images_folded, time_series, att_mask, init_img_size)
                _, logits = self.model.anomaly_detection_loss(local_embeddings, labels_tensor)

        probs = torch.softmax(logits, dim=-1)[:, :, 1].detach().squeeze().cpu().numpy()
        labels_np = labels_tensor.squeeze().cpu().numpy()
        values = time_series.detach().squeeze().cpu().numpy()
        run_time = time.time() - start_time

        result = {
            "probs": probs,
            "labels_np": labels_np,
            "values": values,
            "run_time": run_time,
        }

        if save_path is not None:
            pd.DataFrame({
                "value": values.tolist(),
                "label": labels_np.tolist(),
                "anomaly_score": probs.tolist(),
            }).to_pickle(save_path)
            if verbose:
                logger.info(f"Saved {save_path} (time: {run_time:.4f}s)")

        # 清理
        del batch, images, time_series, att_mask, labels_tensor
        del images_folded, local_embeddings, logits, probs, labels_np, values
        torch.cuda.empty_cache()
        gc.collect()

        return result

    # ------------------------------------------------------------------
    # TSB-AD 基准遍历
    # ------------------------------------------------------------------

    def evaluate_benchmark(
        self,
        dataset_dir: str,
        file_list: Optional[List[str]] = None,
        target_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        pass_list: Optional[List[str]] = None,
        use_list: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> float:
        """遍历 TSB-AD 所有数据集，执行推理 + 后处理，返回 1 - avg_vus_pr。

        Args:
            dataset_dir: 数据集根目录路径。
            file_list: 要测试的文件列表，默认从 dataset_dir 读取。
            target_dir: 推理结果 pkl 保存目录。
            save_dir: 最终指标 CSV 保存目录。
            pass_list: 需要跳过的数据集关键字列表。
            use_list: 用于汇总统计的数据集名称列表。
            num_workers: 并行后处理 worker 数。
            verbose: 是否打印进度。

        Returns:
            float: 1 - avg_vus_pr（越小越好）。
        """
        if file_list is None:
            file_list = sorted(os.listdir(dataset_dir))
        if target_dir is None:
            target_dir = os.path.join(save_dir or "./output", self.model_name)
        if save_dir is None:
            save_dir = "./output/metrics/uni/"
        if pass_list is None:
            pass_list = PASS_LIST_UNI
        if use_list is None:
            use_list = USE_LIST_UNI

        os.makedirs(target_dir, exist_ok=True)
        self.model.eval()
        self.model.to(self.device)

        runtime_log = []
        progress_bar = tqdm(file_list, desc=f"[Stage 1] Saving results for {self.model_name}")

        for filename in progress_bar:
            if any(filter_item in filename for filter_item in pass_list):
                continue

            output_path = os.path.join(target_dir, f'{filename.split(".")[0]}_results.pkl')
            file_path = os.path.join(dataset_dir, filename)
            df = pd.read_csv(file_path).dropna()
            datas = df.iloc[:, :-1].values.astype(float)
            labels_full = df["Label"].astype(int).to_numpy()

            train_index = int(filename.split(".")[0].split("_")[-3])
            data = datas[train_index:, :]
            labels = labels_full[train_index:]

            result = self.evaluate_dataset(data, labels, save_path=output_path, verbose=verbose)
            runtime_log.append({
                "filename": filename,
                "run_time_seconds": result["run_time"],
            })

            del df, datas, labels_full, data, labels
            torch.cuda.empty_cache()
            gc.collect()

        # 保存运行时间日志
        log_df = pd.DataFrame(runtime_log)
        csv_save_path = os.path.join(os.getcwd(), f"runtime_log_{self.model_name}.csv")
        log_df.to_csv(csv_save_path, index=False)

        # 并行后处理
        avg_vus_pr = self._parallel_postprocess(
            dataset_dir=dataset_dir,
            file_list=file_list,
            target_dir=target_dir,
            save_dir=save_dir,
            pass_list=pass_list,
            use_list=use_list,
            num_workers=num_workers,
            verbose=verbose,
        )

        return 1.0 - avg_vus_pr

    # ------------------------------------------------------------------
    # 并行后处理（对应 Test_TSB.py 中 TSB_test_parallel_postprocess）
    # ------------------------------------------------------------------

    def _parallel_postprocess(
        self,
        dataset_dir: str,
        file_list: List[str],
        target_dir: str,
        save_dir: str,
        pass_list: Optional[List[str]] = None,
        use_list: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> float:
        """并行后处理：读取推理结果，计算各文件指标，汇总平均 VUS-PR。

        Returns:
            float: avg_vus_pr
        """
        from TSB_AD.utils.slidingWindows import find_length_rank

        if pass_list is None:
            pass_list = PASS_LIST_UNI
        if use_list is None:
            use_list = USE_LIST_UNI

        tasks = []
        for filename in file_list:
            if any(filter_item in filename for filter_item in pass_list):
                continue
            result_path = os.path.join(target_dir, f'{filename.split(".")[0]}_results.pkl')
            if not os.path.exists(result_path):
                continue
            file_path = os.path.join(dataset_dir, filename)
            df = pd.read_csv(file_path).dropna()
            datas = df.iloc[:, 0:-1].values.astype(float)
            slidingWindow = find_length_rank(datas[:, 0].reshape(-1, 1), rank=1)
            tasks.append((result_path, filename, slidingWindow))
            del df, datas, slidingWindow
        gc.collect()

        results = []
        ctx = mp.get_context("spawn")
        cpu_cnt = mp.cpu_count()
        if num_workers is None:
            safe_workers = max(1, cpu_cnt - 2)
        else:
            safe_workers = min(num_workers, max(1, cpu_cnt - 2))
        if verbose:
            logger.info(f"CPU cores: {cpu_cnt}, using {safe_workers} workers for post-processing")

        with ProcessPoolExecutor(max_workers=safe_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_process_single_result_file, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="[Stage 2] Post-processing"):
                res = future.result()
                if res:
                    results.append(res)
                del res, future
                gc.collect()

        # 汇总为 CSV
        write_csv = []
        col_w = None
        for res in results:
            row = [res["filename"], res["length"], 0] + list(res["metrics"].values())
            write_csv.append(row)
            if col_w is None:
                col_w = ["file", "Length", "Time"] + list(res["metrics"].keys())

        w_csv = pd.DataFrame(write_csv, columns=col_w)

        summary_rows = []
        for dataset_name in use_list:
            mask = w_csv["file"].str.contains(dataset_name, case=True, na=False)
            subset = w_csv[mask]
            if len(subset) == 0:
                continue
            numeric_cols = subset.select_dtypes(include=[np.number]).columns
            mean_values = subset[numeric_cols].mean(axis=0).round(4)
            summary_row = {"file": f"{dataset_name}_MEAN", "Time": mean_values.get("Time", 0)}
            for col in col_w[2:]:
                summary_row[col] = mean_values.get(col, np.nan)
            summary_rows.append(summary_row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows, columns=col_w)
            final_csv = pd.concat([w_csv, summary_df], ignore_index=True)
        else:
            final_csv = w_csv

        timestamp = datetime.now().strftime("%m%d-%H")
        suffix = "_sq.csv" if self.data_setting.get("T_sqrt", False) else "_P.csv"
        img_size = self.data_setting.get("img_size", 224)
        output_csv = f"{save_dir}/{self.model_name}_{img_size}_{timestamp}{suffix}"
        os.makedirs(save_dir, exist_ok=True)
        final_csv.to_csv(output_csv, index=False)
        if verbose:
            logger.info(f"Final results saved to: {output_csv}")

        # 计算所有数据集的平均 VUS-PR
        avg_pr = 0.0
        if summary_rows and "VUS-PR" in final_csv.columns:
            mean_rows = final_csv[final_csv["file"].str.contains("_MEAN", na=False)]
            if not mean_rows.empty:
                avg_pr = mean_rows["VUS-PR"].mean()
                if verbose:
                    logger.info(f"Average VUS-PR across datasets: {avg_pr:.4f}")

        return avg_pr

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(score, labels, sliding_window=100):
        """调用 evaluation.metrics 计算指标。"""
        from evaluation.metrics import get_metrics
        return get_metrics(score, labels, slidingWindow=sliding_window)
