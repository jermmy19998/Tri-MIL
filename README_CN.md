# Tri-MIL

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

<p align="center">
  <img src="_readme/tri_mil_logo.png" width="260" alt="Tri-MIL logo">
</p>

<p align="center">
  一个面向 WSI 预处理、特征提取、MIL 训练、评估与可视化的统一工作流。
</p>

Tri-MIL 是一个面向计算病理的项目，由两个互补方向融合扩展而来：

- [Trident](https://github.com/mahmoodlab/trident)：提供现代化的 WSI 预处理与病理基础模型特征提取能力
- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE)：提供统一的多实例学习（MIL）训练与评估框架

Tri-MIL 不是把两个项目简单拼接在一起，而是希望把它们整合为一条完整、实用的弱监督全切片学习流程：从原始切片，到特征，再到 MIL 实验与结果可视化。

## Tri-MIL 一览

| 模块 | 作用 | 主要路径 |
|---|---|---|
| 预处理 | 读取 WSI、组织分割、patch 坐标生成、patch/slide 特征提取 | `trident/`, `run_batch_of_slides.py`, `run_single_slide.py` |
| 训练 | 基于统一配置接口训练 MIL 模型 | `configs/`, `modules/`, `process/`, `train_mil.py` |
| 评估 | 测试模型并导出指标与推理结果 | `test_mil.py`, `infer_mil.py` |
| 工具 | 数据集划分、结果可视化与热图生成 | `split_scripts/`, `vis_scripts/`, `draw_heatmap/` |

## 为什么做 Tri-MIL

很多病理项目都会把这些步骤拆散在多个仓库里：

- WSI 预处理
- 特征提取
- 数据集划分
- MIL 训练
- 测试与可视化

Tri-MIL 的目标是把这些环节统一在一个仓库里，让实验更容易复现、扩展和维护。

## 工作流

| 阶段 | 内容 | 输出 |
|---|---|---|
| 1. WSI 预处理 | 组织分割与 patch 坐标生成 | contours、thumbnails、coordinates |
| 2. 特征提取 | 生成 patch 特征或 slide 特征 | feature files |
| 3. 数据组织 | 构建数据集 CSV 与 train/val/test 划分 | 标准化 csv 文件 |
| 4. MIL 训练 | 从 YAML 配置训练指定 MIL 方法 | checkpoints、logs、metrics |
| 5. 评估与可视化 | 测试模型并解释输出行为 | metrics、ROC、heatmaps、attention maps |

## 核心能力

| 方向 | 说明 |
|---|---|
| WSI 读取 | 支持 OpenSlide、CuCIM、SDPC、普通图像、CZI、OME-Zarr |
| 组织分割 | 支持 HEST、GrandQC、Otsu |
| 特征提取 | 支持统一的 patch encoder 和 slide encoder 调用 |
| MIL 实验 | 多种 MIL 方法共享统一配置接口 |
| 数据划分 | 支持用户自定义和多种 k-fold 划分 |
| 可视化 | 支持 feature map、attention map、heatmap 与推理结果分析 |

## 支持的 Encoder

### Patch Encoder

Tri-MIL 当前通过集成预处理栈支持以下 patch encoder：

| Patch Encoder | Embedding Dim | Args | Link |
|---|---:|---|---|
| ViT-S/16 | 384 | `--patch_encoder vit --patch_size 256 --mag 20` | [timm/vit_small_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k) |
| UNI | 1024 | `--patch_encoder uni_v1 --patch_size 256 --mag 20` | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| UNI2-h | 1536 | `--patch_encoder uni_v2 --patch_size 256 --mag 20` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| CONCH | 512 | `--patch_encoder conch_v1 --patch_size 512 --mag 20` | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| CONCHv1.5 | 768 | `--patch_encoder conch_v15 --patch_size 512 --mag 20` | [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5) |
| Virchow | 2560 | `--patch_encoder virchow --patch_size 224 --mag 20` | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| Virchow2 | 2560 | `--patch_encoder virchow2 --patch_size 224 --mag 20` | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| Phikon | 768 | `--patch_encoder phikon --patch_size 224 --mag 20` | [owkin/phikon](https://huggingface.co/owkin/phikon) |
| Phikon-v2 | 1024 | `--patch_encoder phikon_v2 --patch_size 224 --mag 20` | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2/) |
| KEEP | 768 | `--patch_encoder keep --patch_size 256 --mag 20` | [Astaxanthin/KEEP](https://huggingface.co/Astaxanthin/KEEP) |
| Prov-Gigapath | 1536 | `--patch_encoder gigapath --patch_size 256 --mag 20` | [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| H-Optimus-0 | 1536 | `--patch_encoder hoptimus0 --patch_size 224 --mag 20` | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| H-Optimus-1 | 1536 | `--patch_encoder hoptimus1 --patch_size 224 --mag 20` | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) |
| H0-mini | 768/1536 | `--patch_encoder h0-mini --patch_size 224 --mag 20` | [bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini) |
| MUSK | 1024 | `--patch_encoder musk --patch_size 384 --mag 20` | [xiangjx/musk](https://huggingface.co/xiangjx/musk) |
| Midnight-12k | 3072 | `--patch_encoder midnight12k --patch_size 224 --mag 20` | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) |
| OpenMidnight | 1536 | `--patch_encoder openmidnight --patch_size 224 --mag 20` | [SophontAI/OpenMidnight](https://huggingface.co/SophontAI/OpenMidnight) |
| GPFM | 1024 | `--patch_encoder gpfm --patch_size 224 --mag 20` | [majiabo/GPFM](https://huggingface.co/majiabo/GPFM) |
| GenBio-PathFM | 4608 | `--patch_encoder genbio-pathfm --patch_size 224 --mag 20` | [genbio-ai/genbio-pathfm](https://huggingface.co/genbio-ai/genbio-pathfm) |
| Gemma 4 | 768/1152 | `--patch_encoder {gemma4-e4b, gemma4-26b} --patch_size 224 --mag 20` | [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) / [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) |
| Kaiko | 384/768/1024 | `--patch_encoder {kaiko-vits8, kaiko-vits16, kaiko-vitb8, kaiko-vitb16, kaiko-vitl14} --patch_size 256 --mag 20` | [1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795](https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795) |
| Lunit | 384 | `--patch_encoder lunit-vits8 --patch_size 224 --mag 20` | [1aurent/vit_small_patch8_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) |
| Hibou | 1024 | `--patch_encoder hibou_l --patch_size 224 --mag 20` | [histai/hibou-L](https://huggingface.co/histai/hibou-L) |
| CTransPath-CHIEF | 768 | `--patch_encoder ctranspath --patch_size 256 --mag 10` | - |
| ResNet50 | 1024 | `--patch_encoder resnet50 --patch_size 256 --mag 20` | - |

### Slide Encoder

当前配置的 slide encoder 包括：

| Slide Encoder | 默认 patch encoder / 依赖 | 说明 |
|---|---|---|
| `chief` | `ctranspath` | 需要本地配置 CHIEF 路径 |
| `madeleine` | `conch_v1` | 需要 MADELEINE 依赖 |
| `gigapath` | `gigapath` | slide-level embedding workflow |

## 快速开始

### 1. 安装

```bash
conda create -n tri-mil python=3.10
conda activate tri-mil
pip install -e .
```

### 2. 预处理切片

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder uni_v1 --mag 20 --patch_size 256
```

### 3. 训练 MIL 模型

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml
```

### 4. 测试训练好的模型

```bash
python test_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv /path/to/test.csv --model_weight_path /path/to/model.pth --test_log_dir /path/to/test_logs
```

## 仓库结构

| 路径 | 作用 |
|---|---|
| `trident/` | 内嵌的预处理与特征提取后端 |
| `configs/` | MIL 实验配置 |
| `modules/` | MIL 模型实现 |
| `process/` | 训练与测试流程 |
| `datasets/` | 示例数据集 CSV |
| `split_scripts/` | 数据集划分脚本 |
| `vis_scripts/` | 可视化工具 |
| `draw_heatmap/` | 热图生成 |
| `feature_extractor/` | 兼容旧流程的特征提取工具 |

## 支持的 MIL 模型

Tri-MIL 当前包含以下 MIL 模型配置与实现：

| Model | Paper / Method | Venue / Year |
|---|---|---|
| `MEAN_MIL` | Mean pooling MIL baseline | baseline |
| `MAX_MIL` | Max pooling MIL baseline | baseline |
| `AB_MIL` | [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) | ICML 2018 |
| `MIXUP_MIL` | [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | ICLR 2018 |
| `DT_MIL` | [Deformable Transformer for Multi-instance Learning on Histopathological Image](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_20) | MICCAI 2021 |
| `TRANS_MIL` | [Transformer based Correlated Multiple Instance Learning for WSI Classification](https://arxiv.org/abs/2106.00908) | NeurIPS 2021 |
| `DS_MIL` | [Dual-stream MIL Network for WSI Classification with SSL Contrastive Learning](https://arxiv.org/abs/2011.08939) | CVPR 2021 |
| `CLAM_SB_MIL`, `CLAM_MB_MIL` | [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) | Nat Biomed Eng 2021 |
| `PGCN_MIL` | [Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks](https://github.com/mahmoodlab/Patch-GCN) | MICCAI 2021 |
| `REMIX_MIL` | [A General and Efficient Framework for MIL based WSI Classification](https://arxiv.org/abs/2110.09632) | MICCAI 2022 |
| `S4_MIL` | [Efficiently Modeling Long Sequences with Structured State Spaces](https://github.com/isyangshu/MambaMIL) | ICLR 2022 |
| `DG_MIL` | [Distribution Guided Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2206.08861) | MICCAI 2022 |
| `DTFD_MIL` | [Double-Tier Feature Distillation MIL for Histopathology WSI Classification](https://arxiv.org/abs/2203.12081) | CVPR 2022 |
| `ADD_MIL` | [Additive MIL: Intrinsically Interpretable MIL for Pathology](https://arxiv.org/pdf/2206.01794) | NeurIPS 2022 |
| `ILRA_MIL` | [Exploring Low-rank Property in MIL for Whole Slide Image classification](https://openreview.net/pdf?id=01KmhBsEPFO) | ICLR 2023 |
| `IIB_MIL` | [Integrated instance-level and bag-level MIL with label disambiguation](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54) | MICCAI 2023 |
| `IB_MIL` | [Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images](https://github.com/HHHedo/IBMIL) | CVPR 2023 |
| `RANKMIX_MIL` | [Data Augmentation for Classifying WSIs with Diverse Sizes](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_RankMix_Data_Augmentation_for_Weakly_Supervised_Learning_of_Classifying_Whole_CVPR_2023_paper.html) | CVPR 2023 |
| `MHIM_MIL` | [MIL Framework with Masked Hard Instance Mining for WSI Classification](https://arxiv.org/abs/2307.15254) | ICCV 2023 |
| `WIKG_MIL` | [Dynamic Graph Representation with Knowledge-aware Attention for WSI Analysis](https://arxiv.org/abs/2403.07719) | CVPR 2024 |
| `AMD_MIL` | [Agent Aggregator with Mask Denoise Mechanism for Histopathology WSI Analysis](https://dl.acm.org/doi/10.1145/3664647.3681425) | MM 2024 |
| `FR_MIL` | [Distribution Re-calibration based MIL with Transformer for WSI Classification](https://ieeexplore.ieee.org/abstract/document/10640165) | TMI 2024 |
| `PSEBMIX_MIL` | [Pseudo-Bag Mixup Augmentation for MIL Based Whole Slide Image Classification](https://ieeexplore.ieee.org/abstract/document/10385148) | TMI 2024 |
| `LONG_MIL` | [Scaling Long Contextual MIL for Histopathology WSI Analysis](https://arxiv.org/abs/2311.12885) | NeurIPS 2024 |
| `DGR_MIL` | [Exploring Diverse Global Representation in MIL for WSI Classification](https://arxiv.org/abs/2407.03575) | ECCV 2024 |
| `CDP_MIL` | [cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process](https://arxiv.org/abs/2407.11448) | ECCV 2024 |
| `CA_MIL` | [Context-Aware Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2305.05314) | ICLR 2024 |
| `AC_MIL` | [Attention-Challenging Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2311.07125) | ECCV 2024 |
| `MAMBA_MIL` | [Enhancing Long Sequence Modeling with Sequence Reordering in CPath](https://arxiv.org/abs/2403.06800) | MICCAI 2024 |
| `RET_MIL` | [Retentive Multiple Instance Learning for Histopathological WSI Classification](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_41) | MICCAI 2024 |
| `SC_MIL` | [Sparse Context-aware MIL for Predicting Cancer Survival Probability Distribution in WSI](https://arxiv.org/abs/2407.00664) | MICCAI 2024 |
| `NCIE_MIL` | [Rethinking Decoupled MIL Framework for Histopathological Slide Classification](https://openreview.net/pdf?id=1GxyidfQzc) | MIDL 2024 |
| `RRT_MIL` | [Towards Foundation Model-Level Performance in Computational Pathology](https://github.com/DearCaat/RRT-MIL) | CVPR 2024 |
| `PA_MIL` | [Dynamic Policy-Driven Adaptive Multi-Instance Learning for WSI Classification](https://ieeexplore.ieee.org/document/10656273) | CVPR 2024 |
| `MICRO_MIL` | [Graph-Based MIL for Context-Aware Diagnosis with Microscopic Images](https://arxiv.org/abs/2407.21604) | MICCAI 2025 |
| `DYHG_MIL` | [Dynamic Hypergraph Representation for Bone Metastasis Cancer Analysis](https://arxiv.org/abs/2501.16787) | CMPB 2025 |
| `MSM_MIL` | [Multi-scan Mamba-based Multiple Instance Learning for WSI classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705125009177) | KBS 2025 |
| `MAMBA2D_MIL` | [2DMamba: Efficient State Space Model for Image Representation](https://github.com/AtlasAnalyticsLab/2DMamba) | CVPR 2025 |
| `FOURIER_MIL` | [Fourier filtering-based multiple instance learning for whole slide image analysis](https://doi.org/10.1007/s11263-025-02679-x) | IJCV 2025 |
| `AEM_MIL` | [Attention Entropy Maximization for MIL based WSI Classification](https://arxiv.org/abs/2406.15303) | MICCAI 2025 |
| `MICO_MIL` | [Multiple Instance Learning with Context-Aware Clustering](https://arxiv.org/abs/2506.18028) | MICCAI 2025 |
| `TDA_MIL` | [Top-Down Attention-based Multiple Instance Learning for Whole Slide Image Analysis](https://link.springer.com/chapter/10.1007/978-3-032-04927-8_62) | MICCAI 2025 |
| `PSA_MIL` | [Probabilistic Spatial Attention-Based MIL for Whole Slide Image Classification](https://arxiv.org/abs/2503.16284) | WACV 2026 |
| `STABLE_MIL` | [Entropy-Stabilized Attention-based MIL for Morphologically Variable WSIs](https://ieeexplore.ieee.org/abstract/document/11477827) | TMI 2026 |
| `GDF_MIL` | [Rethinking Multi-Instance Learning through Graph-Driven Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/40081) | AAAI 2026 |
| `DAG_MIL` | [Deformable attention graph representation learning for histopathology WSI analysis](https://ieeexplore.ieee.org/abstract/document/11464653) | ICASSP 2026 |
| `MO_MIL` | [MoMIL: Multi-order Enhanced Multiple Instance Learning for Computational Pathology](https://www.sciencedirect.com/science/article/abs/pii/S0262885626000247) | IJCV 2026 |

## 推荐使用方式

| 场景 | 推荐路径 |
|---|---|
| 新的 WSI 预处理流程 | 使用集成预处理工作流 |
| 新的病理基础模型特征提取 | 使用内嵌 Trident 风格特征栈 |
| MIL 基准测试或方法扩展 | 使用 YAML 配置配合 `train_mil.py` 与 `test_mil.py` |
| 结果分析与解释 | 使用 `infer_mil.py`、`vis_scripts/` 和 `draw_heatmap/` |

## 说明

- `feature_extractor/` 主要用于兼容旧流程。
- 后续推荐优先使用集成式预处理与特征提取工作流。
- 某些 encoder 仍然需要额外依赖或 gated model 访问权限。

## 致谢

Tri-MIL 受益于计算病理生态中关于 WSI 预处理、MIL benchmarking、弱监督学习与病理基础模型的大量工作。
