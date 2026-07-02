# Tri-MIL

<p align="center">
  English | <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <img src="_readme/tri_mil_logo.png" width="260" alt="Tri-MIL logo">
</p>

<p align="center">
  A unified workflow for WSI preprocessing, feature extraction, MIL training, evaluation, and visualization.
</p>

Tri-MIL is a computational pathology project developed by merging and extending two complementary foundations:

- [Trident](https://github.com/mahmoodlab/trident), which provides modern WSI preprocessing and pathology foundation-model feature extraction
- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE), which provides a broad, unified MIL training and evaluation framework

Tri-MIL does not simply place the two projects side by side.  
Its goal is to turn them into one practical workflow for weakly supervised whole-slide learning: from raw slides, to features, to MIL experiments, to visualization.

## Tri-MIL in One View

| Part | Purpose | Main paths |
|---|---|---|
| Preprocessing | Read WSIs, segment tissue, generate patch coordinates, extract patch/slide features | `trident/`, `run_batch_of_slides.py`, `run_single_slide.py` |
| Training | Train MIL models with a unified config-based interface | `configs/`, `modules/`, `process/`, `train_mil.py` |
| Evaluation | Test checkpoints and export metrics / inference outputs | `test_mil.py`, `infer_mil.py` |
| Utilities | Prepare dataset CSVs, build splits, and visualize learned behavior | `prepare_dataset_csv.py`, `split_scripts/`, `vis_scripts/`, `draw_heatmap/` |

## Why Tri-MIL

Most pathology workflows split these steps across multiple repositories:

- WSI preprocessing
- feature extraction
- dataset split generation
- MIL training
- testing and visualization

Tri-MIL keeps them in one place so experiments are easier to reproduce, extend, and maintain.

## Workflow

| Stage | What happens | Output |
|---|---|---|
| 1. Preprocess WSIs | tissue segmentation and patch coordinate generation | contours, thumbnails, coordinates |
| 2. Extract features | patch embeddings or slide embeddings | feature files |
| 3. Prepare dataset CSVs and splits | dataset CSV construction and train/val/test split generation | standardized csv files |
| 4. Train MIL | fit a selected MIL model from YAML config | checkpoints, logs, metrics |
| 5. Evaluate and visualize | test model behavior and interpret outputs | metrics, ROC, heatmaps, attention maps |

## Core Capabilities

| Area | Summary |
|---|---|
| WSI readers | OpenSlide, CuCIM, SDPC, image files, CZI, OME-Zarr |
| Tissue segmentation | HEST, GrandQC, Otsu |
| Feature extraction | unified patch and slide encoder loading |
| MIL experimentation | many MIL methods under one config-driven interface |
| Dataset splits | user-defined and k-fold split strategies |
| Visualization | feature maps, attention maps, heatmaps, inference summaries |

## Supported Encoders

### Patch Encoders

Tri-MIL currently supports the following patch encoders through the integrated preprocessing stack:

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

### Slide Encoders

Configured slide encoder support currently includes:

| Slide Encoder | Default patch encoder / dependency | Notes |
|---|---|---|
| `chief` | `ctranspath` | requires local CHIEF path configuration |
| `madeleine` | `conch_v1` | requires MADELEINE dependency |
| `gigapath` | `gigapath` | slide-level embedding workflow |

## Quick Start

### 1. Install

```bash
conda create -n tri-mil python=3.10
conda activate tri-mil
pip install -e .
```

### 2. Simplest inference workflow

If you already have extracted features in one folder, you no longer need to manually prepare `test.csv` first.

```bash
python tri_mil.py infer \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --model_weight_path /path/to/model.pth \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --test_log_dir ./infer_out \
  --no_label
```

What this now does for you automatically:

- generates an internal inference CSV from `--feature_dir`
- infers `Model.in_dim` from the first feature file
- picks a safe runtime device with fallback
- avoids manual YAML editing for common inference cases

If you already have a CSV, you can still use the old style:

```bash
python infer_mil.py \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --test_dataset_csv ./test.csv \
  --model_weight_path /path/to/model.pth \
  --test_log_dir ./infer_out \
  --no_label
```

### 3. Simplest heatmap workflow

If you already have precomputed features and coordinates, you no longer need to keep editing `draw_heatmap/heatmap.yaml` for every run.

```bash
python tri_mil.py heatmap \
  --wsi_dir ./wsis \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --coord_dir ./tri_outputs/20x_256px_0px_overlap/patches \
  --model_yaml ./configs/TRANS_MIL.yaml \
  --model_ckpt /path/to/model.pth \
  --job_dir ./heatmap_out \
  --mpp 0.5 \
  --reader_type image \
  --blur
```

This is especially useful for `.png/.jpg` inputs and for the case where features and patch coordinates are already prepared.
If your input slides live under nested folders, Tri-MIL now automatically prefixes each heatmap output with the original relative folder name to avoid collisions between slides with the same basename.

### 4. Preprocess slides

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder uni_v1 --mag 20 --patch_size 256
```

If your WSIs are stored under label subfolders, add `--search_nested`:

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256 --search_nested
```

### 5. Simplest training workflow

You can now train directly from a feature folder plus one label source, without manually creating `train_base.csv`, fold CSVs, or editing YAML dataset paths.

Feature folder + reference CSV:

```bash
python tri_mil.py train \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --reference_csv ./labels.csv \
  --slide_col slide_id \
  --label_col label \
  --dataset_name MY_DATASET \
  --output_dir ./train_workspace \
  --k 3
```

Feature folder + label subfolders in the raw-data directory:

```bash
python tri_mil.py train \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --source_dir ./wsis \
  --dataset_name MY_DATASET \
  --output_dir ./train_workspace \
  --source_recursive \
  --k 3
```

This now happens automatically:

- builds the internal labeled training CSV
- generates k-fold CSVs
- infers `Model.in_dim`
- infers `General.num_classes`
- writes `label_map.json`
- patches the runtime YAML so you do not need to edit dataset paths by hand

### 6. Prepare dataset CSVs manually if needed

Tri-MIL now uses one script, `prepare_dataset_csv.py`, for both training and inference CSV generation.

Flat folder of extracted features + reference CSV:

```bash
python prepare_dataset_csv.py \
  --mode train_flat \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --reference_csv ./labels.csv \
  --slide_col slide_id \
  --label_col label \
  --output_csv ./train_base.csv
```

Raw data folder contains one label subfolder per class, while extracted features are stored flat:

```bash
python prepare_dataset_csv.py \
  --mode train_label_dirs \
  --source_dir ./wsis \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --output_csv ./train_base.csv \
  --source_recursive
```

Feature directory itself already contains label subfolders:

```bash
python prepare_dataset_csv.py \
  --mode train_label_dirs \
  --feature_dir ./labeled_features \
  --output_csv ./train_base.csv
```

Inference from a feature folder without labels:

```bash
python prepare_dataset_csv.py \
  --mode infer \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --output_csv ./test.csv
```

For training, convert `train_base.csv` (`slide_path,label`) into fold files:

```bash
python ./split_scripts/split_datasets_k_fold_train_val.py \
  --csv_path ./train_base.csv \
  --dataset_name MY_DATASET \
  --save_dir ./datasets
```

### 7. Train a MIL model manually

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml
```

### 8. Test a trained model

```bash
python test_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv /path/to/test.csv --model_weight_path /path/to/model.pth --test_log_dir /path/to/test_logs
```

## Repository Structure

| Path | Role |
|---|---|
| `trident/` | embedded preprocessing and feature extraction backend |
| `configs/` | MIL experiment configs |
| `modules/` | MIL model implementations |
| `process/` | training and testing pipelines |
| `datasets/` | example dataset CSVs |
| `prepare_dataset_csv.py` | unified dataset CSV preparation for train / infer |
| `split_scripts/` | dataset split generation |
| `vis_scripts/` | visualization utilities |
| `draw_heatmap/` | heatmap generation |
| `feature_extractor/` | legacy compatibility utilities |

## Supported MIL Models

Tri-MIL currently includes the following MIL model configs and implementations:

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

## Recommended Use

| Scenario | Recommended path |
|---|---|
| new WSI preprocessing pipeline | use the integrated preprocessing workflow |
| new pathology FM feature extraction | use the embedded Trident-based feature stack |
| MIL benchmarking or extension | use YAML configs with `train_mil.py` and `test_mil.py` |
| result inspection | use `infer_mil.py`, `vis_scripts/`, and `draw_heatmap/` |

## Notes

- `feature_extractor/` is kept mainly for backward compatibility.
- the integrated preprocessing workflow is the recommended route going forward.
- some encoders still require optional dependencies or gated model access.

## Acknowledgements

Tri-MIL is built on ideas from the broader computational pathology ecosystem, especially work around WSI preprocessing, MIL benchmarking, weak supervision, and pathology foundation models.
