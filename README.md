# Tri-MIL

<p align="center">
  <img src="_readme/trident_crop.jpg" width="220" alt="Tri-MIL">
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
| Utilities | Build dataset splits and visualize learned behavior | `split_scripts/`, `vis_scripts/`, `draw_heatmap/` |

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
| 3. Prepare splits | dataset CSV construction and train/val/test split generation | standardized csv files |
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

## Quick Start

### 1. Install

```bash
conda create -n tri-mil python=3.10
conda activate tri-mil
pip install -e .
```

### 2. Preprocess slides

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder uni_v1 --mag 20 --patch_size 256
```

### 3. Train a MIL model

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml
```

### 4. Test a trained model

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
| `split_scripts/` | dataset split generation |
| `vis_scripts/` | visualization utilities |
| `draw_heatmap/` | heatmap generation |
| `feature_extractor/` | legacy compatibility utilities |

## Model Coverage

Tri-MIL includes a broad collection of MIL methods, covering:

- pooling-based MIL
- transformer and long-context MIL
- graph-based MIL
- context-aware and attention-based MIL
- robust, probabilistic, and distribution-aware MIL

All implemented methods are organized under `configs/`, `modules/`, and `process/`.

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
