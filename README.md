# VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Arxiv](https://img.shields.io/badge/arXiv-2602.16681-b31b1b.svg)](https://arxiv.org/abs/2602.16681)

Official code repository for VETime (https://arxiv.org/abs/2602.16681), implemented in PyTorch. VETime proposes a novel time-series anomaly detection framework that unifies temporal and visual modalities through fine-grained alignment and dynamic fusion, achieving state-of-the-art zero-shot localization performance with lower computational overhead than existing vision-based approaches.

---

## 📊 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Evaluation Metrics](#-evaluation-metrics)
- [License](#-license)
- [Citation](#-citation)
- [References](#-references)

---

## 📄 Overview

Time-series anomaly detection (TSAD) requires identifying both immediate Point Anomalies and long-range Context Anomalies. However, existing foundation models face a fundamental trade-off: 1D temporal models provide fine-grained pointwise localization but lack a global contextual perspective, while 2D vision-based models capture global patterns but suffer from information bottlenecks due to a lack of temporal alignment and coarse-grained pointwise detection. To resolve this dilemma, we propose **VETime** , the first TSAD framework that unifies temporal and visual modalities through fine-grained visual-temporal alignment and dynamic fusion. 

VETime introduces a Reversible Image Conversion and a Patch-Level Temporal Alignment module to establish a shared visual-temporal timeline, preserving discriminative details while maintaining temporal sensitivity. Furthermore, we design an Anomaly Window Contrastive Learning mechanism and a Task-Adaptive Multi-Modal Fusion to adaptively integrate the complementary perceptual strengths of both modalities. Extensive experiments demonstrate that VETime significantly outperforms state-of-the-art models in zero-shot scenarios, achieving superior localization precision with lower computational overhead than current vision-based approaches.

---

## 📁 Project Structure

```
VETime/
├── src/vetime/               # Application package
│   ├── models/               # temporal, vision, multimodal composition
│   ├── data/                 # datasets, collation, image conversion
│   ├── losses/               # training losses
│   ├── metrics/              # TSB metrics and supporting implementations
│   └── infrastructure/       # checkpointing and runtime logging
├── train_hydra.py            # Primary Hydra training entry point
├── train.py                  # CLI training entry point
├── Test_TSB.py               # TSB-AD evaluation entry point
├── dataset/
│   └── TSB-AD/               # TSB-AD benchmark assets (data only)
├── checkpoints/
│   ├── weight_ts/            # legacy-compatible temporal pretraining weights
│   └── weight_v/             # MAE weights
└── requirements.txt          # Dependencies list
```

---

## 📦 Installation

### Requirements

- Python 3.8+ (Tested on 3.11)
- PyTorch 2.3.0+ with CUDA
- CUDA 12.1 (Recommended)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yyyangcoder/VETime.git
cd VETime

# 2. Create conda environment
conda create -n VETime python=3.11
conda activate VETime

# 3. Install PyTorch (adjust according to your CUDA version)
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install TSB-AD package
cd dataset/TSB-AD && pip install -e .
```

---

## 🚀 Quick Start

### Download Pre-trained Model Checkpoints
Download the pre-trained model checkpoints from Hugging Face:

```bash
huggingface-cli download yyyang0/VETime-checkpoints --local-dir ./checkpoints
```

### Download TSB-AD Datasets

This project uses the **TSB-AD** (Time Series Benchmark for Anomaly Detection) datasets:

**Download Links**:
- [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip)
- [TSB-AD-M](https://www.thedatum.org/datasets/TSB-AD-M.zip)

### Dataset Structure

After downloading and extracting, place the datasets in the following directory structure:

```
./dataset/TSB-AD/Datasets/
├── TSB-AD-U/          # Univariate datasets
├── TSB-AD-M/          # Multivariate datasets
└── File_List/         # Evaluation split files
```

### Training and checkpoint protocol

`train_hydra.py` is the primary entry point.  It prints the device, clean
composition class, CMRG/QueryDecoder state, trainable parameter groups, and
the initialization source before training begins.

First verify the two supplied pretraining checkpoints without starting a run:

```bash
export PYTHONPATH=src:.
python scripts/validate_checkpoints.py --full-model
```

For a new run initialized from the supplied temporal pretraining and MAE
weights:

```bash
python train_hydra.py
```

The initialization sources are mutually exclusive:

- `paths.ts_path`: the original temporal pretraining checkpoint; MAE is always read from `paths.vision_path` and is frozen.
- `paths.model_checkpoint`: a strict v3 VETime model checkpoint.
- `paths.resume`: a strict v3 training-resume checkpoint (model, optimizer, scheduler, RNG and progress).

Legacy complete VETime checkpoints (`--vetime_path`) are deliberately not supported. To use a v3 model checkpoint with Hydra, disable the temporal source in the same command:

```bash
python train_hydra.py paths.ts_path=null paths.model_checkpoint=./output/VETime-query-08-24__224_best.pth
```

To resume an interrupted run:

```bash
python train_hydra.py paths.ts_path=null paths.resume=./output/checkpoints/VETime-query-08-24/univariate_epoch0_full.pth
```

Best model files use the v3 `vetime_model` format. Periodic files use the v3
`training_resume` format. Both formats perform strict loading and reject raw
state dictionaries or obsolete full-model checkpoints. When evaluating a v3
model, provide the same architecture switches used for training (for example
`--cmrg_enabled` when CMRG was enabled); strict loading will otherwise report
the mismatch instead of silently dropping weights.

---

## 📈 Evaluate Model

### Evaluation Metrics

VETime employs the following comprehensive metrics for evaluation:

| Metric | Description |
|--------|-------------|
| **VUS-PR** | Volume Under Surface (Precision-Recall) |
| **Affiliation Metrics** | Event-based evaluation metrics |
| **F1-T** | A range-based metric that evaluates anomaly detection performance by considering the temporal context of anomalies |
| **Standard-F1** | Standard F1 score |
---


### Evaluate on TSB-AD benchmark

```bash
python Test_TSB.py \
    --model_name VETime \
    --dataset_dir ./dataset/TSB-AD/Datasets/TSB-AD-U \
    --model_checkpoint ./output/VETime-query-08-24__224_best.pth \
    --cmrg_enabled
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you find VETime useful in your research, please consider citing our paper:

```bibtex
@article{yang2026vetime,
  title={VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection},
  author={Yingyuan Yang and Tian Lan and Yifei Gao and Yimeng Lu and Wenjun He and Meng Wang and Chenghao Liu and Chen Zhang},
  journal={arXiv preprint arXiv:2602.16681},
  year={2026}
}
```

## 🔗 References

### Related Papers

- **TSB-AD**: "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" (NeurIPS 2024)
- **Time-RCD**: "Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy"

### Related Repositories

- [MAE](https://github.com/facebookresearch/mae)
- [TSB-AD](https://github.com/TheDatumOrg/TSB-AD)
- [Time-RCD](https://github.com/thu-sail-lab/Time-RCD)
