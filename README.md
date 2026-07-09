# UNETR-pp-PEFT: Adapting a Pretrained Segmentation Transformer to a Data-Scarce Medical Imaging Task

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-3D%20Medical%20Imaging-orange)
<!-- ![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen) -->
![Best Dice](https://img.shields.io/badge/Best%20Dice-0.9515-success)

> Investigating whether a segmentation transformer pretrained on 13 abdominal organs can be efficiently adapted to a new, data-scarce single-organ task — without training from scratch.

## Overview

State-of-the-art 3D Vision Transformers (ViTs) like **UNETR++** utilize advanced spatial attention mechanisms to achieve highly accurate volumetric segmentation. However, these heavy transformer architectures are computationally expensive and require massive labeled datasets to train from scratch. In applied computer vision, engineers are far more likely to adapt an existing pretrained ViT to a data-scarce task than to train from zero.*(For full architectural details on the Efficient Paired Attention mechanism, refer to the original paper: [UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation (Shaker et al., IEEE TMI 2024)](https://arxiv.org/abs/2212.04497)).*

This repository explores the limits of ViT domain adaptation. It asks a concrete question: **can a 3D segmentation transformer pretrained on a complex 13-organ dataset (Synapse/BTCV) be adapted to a new, single-organ target using a severely limited data budget?**

This repository establishes **Phase 1: the Full Fine-Tuning Baseline**. By achieving >0.95 Dice on just 33 training patients, it proves that the pretrained transformer weights transfer effectively to a novel task. This establishes a strong benchmark for upcoming parameter-efficient fine-tuning (PEFT) experiments.

## Phase 1 Key Results (Full Fine-Tuning)

| | |
|---|---|
| **Backbone** | UNETR++, pretrained on Synapse (13-organ abdominal CT) |
| **Target task** | MSD Spleen (Task09_Spleen) — 41 CT volumes, single organ, binary segmentation |
| **Adaptation strategy** | Full fine-tuning (all backbone weights + freshly initialized 2-class output heads) |
| **Data split** | 33 train / 8 validation |
| **Best Mean Dice** | **0.9515**    |

For reference, this matches the performance of a fully-converged CNN U-Net trained from scratch on this dataset, but achieves it with a fraction of the domain-specific data — a meaningful result given the backbone was never trained on spleen data directly.

## Fine-Tuning Pipeline

Before trusting any result from a pretrained checkpoint, this pipeline verifies the transfer explicitly:

- **Exact key accounting** — the model contains 636 total parameter/buffer tensors. Exactly 6 belong to the three multi-resolution output heads, which are freshly initialized for 2 classes instead of Synapse's original 13. The remaining 630 backbone tensors matched the checkpoint by name and shape with zero exceptions, confirmed by an explicit programmatic audit rather than assumed from the result alone.
- **MONAI-based 3D data pipeline** — on-the-fly intensity windowing (Hounsfield units), foreground-aware cropping, anisotropic spacing normalization, and positive/negative-balanced patch sampling (`RandCropByPosNegLabeld`) to combat extreme class imbalance.
- **Deep-supervision shape alignment** — UNETR++ outputs 3 auxiliary masks at different decoder resolutions. The pipeline uses `monai.losses.DeepSupervisionLoss` to correctly downsample the ground truth and weight each scale during backpropagation, rather than naively comparing mismatched resolutions.
- **Automated run tracking** — every run generates a timestamped `runs/` directory containing an isolated `config.yaml`, TensorBoard metrics, training logs, and a fully resumable checkpoint.

## Repository Structure

```text
.
├── configs/
│   └── config.yaml             
├── data/
│   └── dataset.py           
├── unetr_pp/ # UNETR++ model (adapted from official repo)          
├── utils/
│   ├── config.py               
│   └── model_utils.py           
├── train.py               
└── runs/                        
```

## Setup

1. Clone the repository and install the dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```

2. Download the **MSD Spleen** dataset (Task09_Spleen) from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
3. Download the official **Synapse-pretrained UNETR++ checkpoint** from the [official repository](https://github.com/Amshaker/unetr_plus_plus).
4. Update `data_dir` and `weights_path` in `configs/config.yaml`.

## Usage

```bash
python train.py
```

Training logs, the best checkpoint (by validation Dice), and a fully resumable checkpoint are automatically written to `runs/<timestamp>_spleen_unetr_pp/`. Resuming an interrupted run is as simple as passing the run's directory name into `resume_dir` inside the config file.

<!-- ## Current Status & Next Steps

**Phase 1 (the fine-tuning baseline) is complete.** The next phases of this project will explore:

- **Parameter-efficient adapters** — inserting lightweight adapter modules into the Efficient Paired Attention (EPA) blocks while freezing the backbone, to drastically reduce trainable parameters.
- **Linear probing** — freezing the entire encoder/decoder and training only the output heads, to test the raw feature-extraction power of the Synapse weights on their own.
- **nnU-Net baseline** — a CNN trained from scratch on the same split, as the control condition for the central question above. -->

## Acknowledgments

This project builds on the official [UNETR++ implementation](https://github.com/Amshaker/unetr_plus_plus) (Apache License 2.0) by Shaker et al., and uses the [MONAI](https://project-monai.github.io/) framework for data loading, losses, and inference. Dataset: [Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task09_Spleen.