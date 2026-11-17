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

## 0.4.0 update

- **8 new MIL models:** `AS_MIL`, `ATTRI_MIL`, `CAR_MIL`, `FC_MIL`, `LIN_MIL`,
  `MR_AB_MIL`, `NN_MIL`, and `PNEA_MIL`, with model YAMLs and training routines.
- **TRIDENT 0.3.2:** configurable encoder input resolution, GigaPath Flash,
  Virchow2 CLS, Phaet and Mascaret encoders, BioIO conversion, and updated WSI
  processing/state handling. The existing ViT-S encoder remains available.
- **Survival prediction:** full-cohort Cox loss, right censoring, validation
  C-index checkpoint selection, and unlabeled risk prediction through the same
  training, validation and inference entrypoints.
- Case-level splits are available in `split_by_case/`; function docstrings,
  Ruff formatting and regression tests cover the integrated Python sources.

Upstreams were synchronized from the supplied local snapshots on 2026-11-16.
See [changelog](./CHANGELOG.md), [input hashes](./docs/upstream-sync.json), and
[survival guide / 生存预测说明](./docs/survival.md) for scope and limitations.

### Survival quick start

Prepare a manifest containing `slide_path,time,event,case_id`. `event=1` means
observed event; `event=0` means right censoring. Feature paths refer to PT/HDF5
bags. Cases stay together when splitting:

```bash
python prepare_survival_csv.py --manifest ./datasets/survival_manifest.csv --output ./datasets/survival_splits.csv
# Edit feature dimension and paths in the two SURVIVAL YAMLs first.
python train_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python valid_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
```

Use [SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) and
[SURVIVAL_PIPELINE.yaml](./configs/SURVIVAL_PIPELINE.yaml). Split CSVs contain
`train_slide_path,train_time,train_event` and equivalent `val_` / `test_` columns.
Prediction needs only paths. Larger output `risk` means higher hazard.
C-index gives tied risks half credit; no comparable pairs yields JSON `null`.
The exact full-cohort Cox computation retains all training-bag graphs until
backward, so memory scales with cohort size. Use one bag per patient for
patient-level survival. Specialized multi-stage/coordinate backbones require
additional adapters; see the guide for supported interfaces.

### Development checks

Python 3.10–3.12 is supported. For core tests without downloading model weights:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
python -m pytest -q
python tools/document_functions.py
ruff check .
ruff format --check .
```

Full WSI processing and pretrained-encoder validation additionally require the
native reader dependencies, accessible weights, and representative slide data.

## Tri-MIL in One View

| Part | Purpose | Main paths |
|---|---|---|
| Preprocessing | Read WSIs, segment tissue, generate patch coordinates, extract patch/slide features | `trident/`, `run_batch_of_slides.py`, `run_single_slide.py` |
| Training | Train MIL models with a unified config-based interface | `configs/`, `modules/`, `process/`, `train_mil.py` |
| Evaluation | Validate checkpoints and export metrics / inference outputs | `valid_mil.py`, `infer_mil.py` |
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

### 2. Unified YAML Workflow

Tri-MIL now supports one pipeline YAML for `train_mil.py`, `valid_mil.py`, `infer_mil.py`, and `draw_heatmap.py`.

Use [configs/TRI_MIL_PIPELINE.yaml](./configs/TRI_MIL_PIPELINE.yaml) as the shared entrypoint:

```bash
python train_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python valid_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python draw_heatmap.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
```

Recommended convention:

- `Common.model_yaml_path` points to the MIL model YAML such as `./configs/TRANS_MIL.yaml`
- `Common.model_weight_path`, `Common.wsi_dir`, `Common.feature_dir`, and optionally `Common.coord_dir` are shared by inference and heatmap
- `Train`, `Valid`, `Infer`, and `Heatmap` each store only their own runtime settings
- `draw_heatmap.py` defaults to its built-in heatmap yaml, but if you pass the shared pipeline yaml it will automatically reuse `Common.model_yaml_path`, `Common.model_weight_path`, `Common.wsi_dir`, and precomputed feature paths

For the common precomputed-feature workflow, heatmap usually only needs the shared YAML. A typical setup is:

```yaml
Common:
  model_yaml_path: ./configs/TRANS_MIL.yaml
  model_weight_path: ./infer_out/best.pth
  wsi_dir: ./wsis
  feature_dir: ./trident_processed/20x_256px_0px_overlap/features_vit
```

If `coord_dir` is left empty, `draw_heatmap.py` will try to auto-detect `./patches` next to `feature_dir`.

### 3. Preprocess slides

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder uni_v1 --mag 20 --patch_size 256
```

If your WSIs are stored under label subfolders, add `--search_nested`:

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256 --search_nested
```

### 4. Prepare dataset CSVs manually if needed

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

### 5. Train a MIL model

```bash
python train_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
```

### 6. Validate a trained model

```bash
python valid_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
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

Tri-MIL includes **58 MIL models**, each with a model configuration and implementation. The list below matches the model YAMLs in `configs/`; optional dependencies vary by model.

| Model | Configuration | Paper / Method |
|---|---|---|
| `AB_MIL` | [AB_MIL.yaml](./configs/AB_MIL.yaml) | [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) |
| `AC_MIL` | [AC_MIL.yaml](./configs/AC_MIL.yaml) | [Attention-Challenging Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2311.07125) |
| `ADD_MIL` | [ADD_MIL.yaml](./configs/ADD_MIL.yaml) | [Additive MIL: Intrinsically Interpretable MIL for Pathology](https://arxiv.org/pdf/2206.01794) |
| `AEM_MIL` | [AEM_MIL.yaml](./configs/AEM_MIL.yaml) | [Attention Entropy Maximization for MIL based WSI Classification](https://arxiv.org/abs/2406.15303) |
| `AMD_MIL` | [AMD_MIL.yaml](./configs/AMD_MIL.yaml) | [Agent Aggregator with Mask Denoise Mechanism for Histopathology WSI Analysis](https://dl.acm.org/doi/10.1145/3664647.3681425) |
| `AS_MIL` | [AS_MIL.yaml](./configs/AS_MIL.yaml) | Attention-Stabilized Multiple Instance Learning |
| `ATTRI_MIL` | [ATTRI_MIL.yaml](./configs/ATTRI_MIL.yaml) | Attribute-based Multiple Instance Learning |
| `CA_MIL` | [CA_MIL.yaml](./configs/CA_MIL.yaml) | [Context-Aware Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2305.05314) |
| `CAR_MIL` | [CAR_MIL.yaml](./configs/CAR_MIL.yaml) | Counterfactual Attention Regularization for Multiple Instance Learning |
| `CDP_MIL` | [CDP_MIL.yaml](./configs/CDP_MIL.yaml) | [cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process](https://arxiv.org/abs/2407.11448) |
| `CLAM_MB_MIL` | [CLAM_MB_MIL.yaml](./configs/CLAM_MB_MIL.yaml) | [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) |
| `CLAM_SB_MIL` | [CLAM_SB_MIL.yaml](./configs/CLAM_SB_MIL.yaml) | [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) |
| `DAG_MIL` | [DAG_MIL.yaml](./configs/DAG_MIL.yaml) | [Deformable attention graph representation learning for histopathology WSI analysis](https://ieeexplore.ieee.org/abstract/document/11464653) |
| `DG_MIL` | [DG_MIL.yaml](./configs/DG_MIL.yaml) | [Distribution Guided Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2206.08861) |
| `DGR_MIL` | [DGR_MIL.yaml](./configs/DGR_MIL.yaml) | [Exploring Diverse Global Representation in MIL for WSI Classification](https://arxiv.org/abs/2407.03575) |
| `DS_MIL` | [DS_MIL.yaml](./configs/DS_MIL.yaml) | [Dual-stream MIL Network for WSI Classification with SSL Contrastive Learning](https://arxiv.org/abs/2011.08939) |
| `DT_MIL` | [DT_MIL.yaml](./configs/DT_MIL.yaml) | [Deformable Transformer for Multi-instance Learning on Histopathological Image](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_20) |
| `DTFD_MIL` | [DTFD_MIL.yaml](./configs/DTFD_MIL.yaml) | [Double-Tier Feature Distillation MIL for Histopathology WSI Classification](https://arxiv.org/abs/2203.12081) |
| `DYHG_MIL` | [DYHG_MIL.yaml](./configs/DYHG_MIL.yaml) | [Dynamic Hypergraph Representation for Bone Metastasis Cancer Analysis](https://arxiv.org/abs/2501.16787) |
| `FC_MIL` | [FC_MIL.yaml](./configs/FC_MIL.yaml) | [FC-MIL](https://github.com/7FFDW/FCMIL) |
| `FOURIER_MIL` | [FOURIER_MIL.yaml](./configs/FOURIER_MIL.yaml) | [Fourier filtering-based multiple instance learning for whole slide image analysis](https://doi.org/10.1007/s11263-025-02679-x) |
| `FR_MIL` | [FR_MIL.yaml](./configs/FR_MIL.yaml) | [Distribution Re-calibration based MIL with Transformer for WSI Classification](https://ieeexplore.ieee.org/abstract/document/10640165) |
| `GATE_AB_MIL` | [GATE_AB_MIL.yaml](./configs/GATE_AB_MIL.yaml) | Gated Attention-based Multiple Instance Learning |
| `GDF_MIL` | [GDF_MIL.yaml](./configs/GDF_MIL.yaml) | [Rethinking Multi-Instance Learning through Graph-Driven Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/40081) |
| `IB_MIL` | [IB_MIL.yaml](./configs/IB_MIL.yaml) | [Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images](https://github.com/HHHedo/IBMIL) |
| `IIB_MIL` | [IIB_MIL.yaml](./configs/IIB_MIL.yaml) | [Integrated instance-level and bag-level MIL with label disambiguation](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_54) |
| `ILRA_MIL` | [ILRA_MIL.yaml](./configs/ILRA_MIL.yaml) | [Exploring Low-rank Property in MIL for Whole Slide Image classification](https://openreview.net/pdf?id=01KmhBsEPFO) |
| `INSMIX_MIL` | [INSMIX_MIL.yaml](./configs/INSMIX_MIL.yaml) | Instance Mixup Multiple Instance Learning |
| `LIN_MIL` | [LIN_MIL.yaml](./configs/LIN_MIL.yaml) | Linear-attention Multiple Instance Learning |
| `LONG_MIL` | [LONG_MIL.yaml](./configs/LONG_MIL.yaml) | [Scaling Long Contextual MIL for Histopathology WSI Analysis](https://arxiv.org/abs/2311.12885) |
| `MAMBA2D_MIL` | [MAMBA2D_MIL.yaml](./configs/MAMBA2D_MIL.yaml) | [2DMamba: Efficient State Space Model for Image Representation](https://github.com/AtlasAnalyticsLab/2DMamba) |
| `MAMBA_MIL` | [MAMBA_MIL.yaml](./configs/MAMBA_MIL.yaml) | [Enhancing Long Sequence Modeling with Sequence Reordering in CPath](https://arxiv.org/abs/2403.06800) |
| `MAX_MIL` | [MAX_MIL.yaml](./configs/MAX_MIL.yaml) | Max pooling MIL baseline |
| `MEAN_MIL` | [MEAN_MIL.yaml](./configs/MEAN_MIL.yaml) | Mean pooling MIL baseline |
| `MHIM_MIL` | [MHIM_MIL.yaml](./configs/MHIM_MIL.yaml) | [MIL Framework with Masked Hard Instance Mining for WSI Classification](https://arxiv.org/abs/2307.15254) |
| `MICO_MIL` | [MICO_MIL.yaml](./configs/MICO_MIL.yaml) | [Multiple Instance Learning with Context-Aware Clustering](https://arxiv.org/abs/2506.18028) |
| `MICRO_MIL` | [MICRO_MIL.yaml](./configs/MICRO_MIL.yaml) | [Graph-Based MIL for Context-Aware Diagnosis with Microscopic Images](https://arxiv.org/abs/2407.21604) |
| `MIXUP_MIL` | [MIXUP_MIL.yaml](./configs/MIXUP_MIL.yaml) | [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) |
| `MO_MIL` | [MO_MIL.yaml](./configs/MO_MIL.yaml) | [MoMIL: Multi-order Enhanced Multiple Instance Learning for Computational Pathology](https://www.sciencedirect.com/science/article/abs/pii/S0262885626000247) |
| `MR_AB_MIL` | [MR_AB_MIL.yaml](./configs/MR_AB_MIL.yaml) | [Attention MIL with manifold residual projections](https://github.com/BearCleverProud/MR-Block) |
| `MSM_MIL` | [MSM_MIL.yaml](./configs/MSM_MIL.yaml) | [Multi-scan Mamba-based Multiple Instance Learning for WSI classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705125009177) |
| `NCIE_MIL` | [NCIE_MIL.yaml](./configs/NCIE_MIL.yaml) | [Rethinking Decoupled MIL Framework for Histopathological Slide Classification](https://openreview.net/pdf?id=1GxyidfQzc) |
| `NN_MIL` | [NN_MIL.yaml](./configs/NN_MIL.yaml) | Attention MIL with random feature subspaces and ensemble uncertainty |
| `PA_MIL` | [PA_MIL.yaml](./configs/PA_MIL.yaml) | [Dynamic Policy-Driven Adaptive Multi-Instance Learning for WSI Classification](https://ieeexplore.ieee.org/document/10656273) |
| `PGCN_MIL` | [PGCN_MIL.yaml](./configs/PGCN_MIL.yaml) | [Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks](https://github.com/mahmoodlab/Patch-GCN) |
| `PNEA_MIL` | [PNEA_MIL.yaml](./configs/PNEA_MIL.yaml) | Positive-Negative Evidence Analysis for Multiple Instance Learning |
| `PSA_MIL` | [PSA_MIL.yaml](./configs/PSA_MIL.yaml) | [Probabilistic Spatial Attention-Based MIL for Whole Slide Image Classification](https://arxiv.org/abs/2503.16284) |
| `PSEBMIX_MIL` | [PSEBMIX_MIL.yaml](./configs/PSEBMIX_MIL.yaml) | [Pseudo-Bag Mixup Augmentation for MIL Based Whole Slide Image Classification](https://ieeexplore.ieee.org/abstract/document/10385148) |
| `RANKMIX_MIL` | [RANKMIX_MIL.yaml](./configs/RANKMIX_MIL.yaml) | [Data Augmentation for Classifying WSIs with Diverse Sizes](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_RankMix_Data_Augmentation_for_Weakly_Supervised_Learning_of_Classifying_Whole_CVPR_2023_paper.html) |
| `REMIX_MIL` | [REMIX_MIL.yaml](./configs/REMIX_MIL.yaml) | [A General and Efficient Framework for MIL based WSI Classification](https://arxiv.org/abs/2110.09632) |
| `RET_MIL` | [RET_MIL.yaml](./configs/RET_MIL.yaml) | [Retentive Multiple Instance Learning for Histopathological WSI Classification](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_41) |
| `RRT_MIL` | [RRT_MIL.yaml](./configs/RRT_MIL.yaml) | [Towards Foundation Model-Level Performance in Computational Pathology](https://github.com/DearCaat/RRT-MIL) |
| `S4_MIL` | [S4_MIL.yaml](./configs/S4_MIL.yaml) | [Efficiently Modeling Long Sequences with Structured State Spaces](https://github.com/isyangshu/MambaMIL) |
| `SC_MIL` | [SC_MIL.yaml](./configs/SC_MIL.yaml) | [Sparse Context-aware MIL for Predicting Cancer Survival Probability Distribution in WSI](https://arxiv.org/abs/2407.00664) |
| `STABLE_MIL` | [STABLE_MIL.yaml](./configs/STABLE_MIL.yaml) | [Entropy-Stabilized Attention-based MIL for Morphologically Variable WSIs](https://ieeexplore.ieee.org/abstract/document/11477827) |
| `TDA_MIL` | [TDA_MIL.yaml](./configs/TDA_MIL.yaml) | [Top-Down Attention-based Multiple Instance Learning for Whole Slide Image Analysis](https://link.springer.com/chapter/10.1007/978-3-032-04927-8_62) |
| `TRANS_MIL` | [TRANS_MIL.yaml](./configs/TRANS_MIL.yaml) | [Transformer based Correlated Multiple Instance Learning for WSI Classification](https://arxiv.org/abs/2106.00908) |
| `WIKG_MIL` | [WIKG_MIL.yaml](./configs/WIKG_MIL.yaml) | [Dynamic Graph Representation with Knowledge-aware Attention for WSI Analysis](https://arxiv.org/abs/2403.07719) |

Survival is a separate task using compatible scalar-output backbones. See [SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) and the [survival guide](./docs/survival.md) for supported interfaces and limitations.

## Recommended Use

| Scenario | Recommended path |
|---|---|
| new WSI preprocessing pipeline | use the integrated preprocessing workflow |
| new pathology FM feature extraction | use the embedded Trident-based feature stack |
| MIL benchmarking or extension | use YAML configs with `train_mil.py` and `valid_mil.py` |
| result inspection | use `infer_mil.py`, `vis_scripts/`, and `draw_heatmap/` |

## Notes

- `feature_extractor/` is kept mainly for backward compatibility.
- the integrated preprocessing workflow is the recommended route going forward.
- some encoders still require optional dependencies or gated model access.

## Acknowledgements

Tri-MIL is built on ideas from the broader computational pathology ecosystem, especially work around WSI preprocessing, MIL benchmarking, weak supervision, and pathology foundation models.
