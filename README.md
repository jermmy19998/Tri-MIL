# Tri-MIL

<p align="center">
  <a href="https://github.com/jermmy19998/Tri-MIL"><img src="https://img.shields.io/badge/GitHub-Tri--MIL-black?style=flat&logo=github"></a>
  <a href="https://github.com/lingxitong/MIL_BASELINE"><img src="https://img.shields.io/badge/Based%20on-MIL__BASELINE-blue?style=flat"></a>
  <a href="https://github.com/mahmoodlab/trident"><img src="https://img.shields.io/badge/Integrated-Trident-green?style=flat"></a>
</p>

<img src="_readme/trident_crop.jpg" width="260px" align="right" />

Tri-MIL is a computational pathology toolbox for weakly supervised whole-slide image learning.
It keeps the unified training and evaluation philosophy of [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE), while integrating a newer [Trident](https://github.com/mahmoodlab/trident) preprocessing and feature extraction stack for slide segmentation, patching, and foundation-model embeddings.

The goal of this repository is to make the full WSI workflow easier to reproduce:

- Use `trident/` and the Trident entry scripts to preprocess WSIs and extract patch or slide embeddings.
- Use the `configs/`, `modules/`, `process/`, `train_mil.py`, and `test_mil.py` stack to train and evaluate many MIL variants under one interface.
- Use `split_scripts/` and `vis_scripts/` to standardize dataset construction, splitting, and visualization.

## Highlights

- Unified MIL training framework adapted from `MIL_BASELINE`.
- Updated embedded `trident/` library based on `mahmoodlab/trident`.
- Support for WSI preprocessing, segmentation, patch extraction, patch embeddings, and slide embeddings.
- Large collection of MIL methods under a common config-driven interface.
- Standard dataset CSV format and multiple dataset split strategies.
- Built-in visualization utilities for attention maps, feature maps, and heatmaps.

## Repository Structure

- `trident/`: embedded Trident library for WSI IO, segmentation, patching, and encoder loading.
- `run_batch_of_slides.py`: batch preprocessing and feature extraction entry.
- `run_single_slide.py`: single-slide preprocessing and feature extraction entry.
- `configs/`: YAML configs for MIL methods.
- `modules/`: model implementations.
- `process/`: training and evaluation pipelines per method.
- `train_mil.py`: MIL training entry point.
- `test_mil.py`: MIL evaluation entry point.
- `infer_mil.py`: inference and metric visualization helper.
- `feature_extractor/`: legacy feature extraction utilities kept for compatibility.
- `split_scripts/`: dataset split scripts.
- `vis_scripts/`: feature map and attention visualization scripts.
- `datasets/`: dataset CSV examples.
- `docs/`: Trident-oriented documentation pages.
- `tutorials/`: notebooks and usage demos.

## Installation

Tri-MIL currently ships with the Trident package metadata in `pyproject.toml`, so the environment should be prepared with Trident-compatible dependencies first.

```bash
conda create -n tri-mil python=3.10
conda activate tri-mil
pip install -e .
```

Optional Trident extras:

```bash
pip install -e ".[patch-encoders]"
pip install -e ".[slide-encoders]"
pip install -e ".[czi]"
pip install -e ".[omezarr]"
pip install -e ".[full]"
```

For Trident environment checks:

```bash
trident-doctor --profile base
trident-doctor --profile patch-encoders --check-gated
trident-doctor --profile slide-encoders
```

## Workflow Overview

Tri-MIL is easiest to use as a two-stage workflow:

1. Preprocess WSIs and extract embeddings with Trident.
2. Train or evaluate MIL models on the produced features.

### Stage 1: WSI Preprocessing with Trident

Run the full preprocessing pipeline on a directory of WSIs:

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

Or process a single slide:

```bash
python run_single_slide.py --slide_path ./wsis/example.svs --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

Common Trident tasks:

- `--task seg`: tissue segmentation
- `--task coords`: patch coordinate extraction
- `--task feat`: feature extraction
- `--task all`: end-to-end pipeline

Examples:

```bash
python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpus 0 --segmenter hest
python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --slide_encoder titan --mag 20 --patch_size 512
```

Trident in this repository supports the newer upstream capabilities, including:

- multiple WSI readers such as OpenSlide, CuCIM, SDPC, OME-Zarr, CZI, and image files
- smart resume and per-slide run tracking
- multi-GPU dispatch
- newer patch encoder and slide encoder registries
- Otsu segmentation fallback

## Dataset Preparation

Tri-MIL uses CSV files to describe slides and labels.
See:

- `datasets/example_Dataset.csv`
- `split_scripts/README.md`

Supported split strategies include:

- k-fold train/val
- k-fold train/val/test
- k-fold train/val then held-out test
- user-defined train/val/test
- user-defined train/test
- user-defined train/val

Example split command:

```bash
python split_scripts/split_datasets_k_fold_train_val.py --seed 2024 --csv_path ./datasets/example_Dataset.csv --save_dir ./your_split_dir --dataset_name YourDataset --k 5
```

## MIL Training

The training side of Tri-MIL follows the `MIL_BASELINE` style configuration system.
Each method is defined by a YAML file in `configs/`.

Example:

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml
```

You can override config values dynamically:

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml --options General.seed=2024 General.num_epochs=20 Model.in_dim=768
```

## MIL Evaluation

Evaluate a trained model with:

```bash
python test_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv /path/to/test.csv --model_weight_path /path/to/model.pth --test_log_dir /path/to/test_logs
```

You can also use:

```bash
python infer_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv /path/to/test.csv --model_weight_path /path/to/model.pth --test_log_dir /path/to/infer_logs
```

## Implemented MIL Methods

Tri-MIL currently includes configs and implementations for:

- `AB_MIL`
- `AC_MIL`
- `ADD_MIL`
- `AEM_MIL`
- `AMD_MIL`
- `CA_MIL`
- `CDP_MIL`
- `CLAM_MB_MIL`
- `CLAM_SB_MIL`
- `DAG_MIL`
- `DGR_MIL`
- `DG_MIL`
- `DS_MIL`
- `DTFD_MIL`
- `DT_MIL`
- `DYHG_MIL`
- `FOURIER_MIL`
- `FR_MIL`
- `GATE_AB_MIL`
- `GDF_MIL`
- `IB_MIL`
- `IIB_MIL`
- `ILRA_MIL`
- `INSMIX_MIL`
- `LONG_MIL`
- `MAMBA2D_MIL`
- `MAMBA_MIL`
- `MAX_MIL`
- `MEAN_MIL`
- `MHIM_MIL`
- `MICO_MIL`
- `MICRO_MIL`
- `MIXUP_MIL`
- `MO_MIL`
- `MSM_MIL`
- `NCIE_MIL`
- `PA_MIL`
- `PGCN_MIL`
- `PSA_MIL`
- `PSEBMIX_MIL`
- `RANKMIX_MIL`
- `REMIX_MIL`
- `RET_MIL`
- `RRT_MIL`
- `S4_MIL`
- `SC_MIL`
- `STABLE_MIL`
- `TDA_MIL`
- `TRANS_MIL`
- `WIKG_MIL`

## Visualization and Utilities

Visualization helpers are kept in:

- `vis_scripts/draw_attention_map.py`
- `vis_scripts/draw_feature_map.py`
- `draw_heatmap/`

Legacy feature extraction utilities are still available in:

- `feature_extractor/`

For new projects, the recommended preprocessing path is the updated Trident pipeline in this repository.

## Acknowledgements

Tri-MIL is built by combining ideas and code paths from:

- [lingxitong/MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE)
- [mahmoodlab/trident](https://github.com/mahmoodlab/trident/)
- [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)
- [hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF)
- [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

Please also follow the original license terms and model licenses required by upstream projects and pretrained checkpoints.

## Notes

- The Trident portion of this repository is embedded code, not a git submodule.
- Some Trident models still require gated Hugging Face access or extra installs.
- The legacy `feature_extractor/` utilities and the new `trident/` pipeline coexist for compatibility, but the Trident path is the recommended one going forward.
