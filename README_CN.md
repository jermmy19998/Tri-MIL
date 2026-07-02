# Tri-MIL

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

<p align="center">
  <img src="_readme/tri_mil_logo.png" width="260" alt="Tri-MIL logo">
</p>

<p align="center">
  面向 WSI 预处理、特征提取、MIL 训练、评估与可视化的一体化流程。
</p>

Tri-MIL 将两部分能力整合到同一套工作流中：

- [Trident](https://github.com/mahmoodlab/trident)：负责 WSI 预处理与病理基础模型特征提取
- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE)：负责统一的 MIL 训练与评估

目标不是把两个仓库简单拼接，而是把从原始切片到特征、再到 MIL 实验和推理的流程打通。

## 一眼看懂

| 模块 | 作用 | 主要路径 |
|---|---|---|
| 预处理 | 读取 WSI、组织分割、坐标生成、特征提取 | `trident/`, `run_batch_of_slides.py`, `run_single_slide.py` |
| 训练 | 基于配置文件训练 MIL 模型 | `configs/`, `modules/`, `process/`, `train_mil.py` |
| 评估 | 测试模型并导出推理结果 | `test_mil.py`, `infer_mil.py` |
| 工具 | 数据集 CSV 准备、划分与可视化 | `prepare_dataset_csv.py`, `split_scripts/`, `vis_scripts/`, `draw_heatmap/` |

## 工作流

| 阶段 | 内容 | 输出 |
|---|---|---|
| 1. WSI 预处理 | 组织分割与 patch 坐标生成 | contours, thumbnails, coordinates |
| 2. 特征提取 | 提取 patch 或 slide 特征 | feature files |
| 3. 数据准备 | 生成训练/推理 CSV，并按需划分 train/val/test | standardized csv files |
| 4. MIL 训练 | 按 YAML 配置训练模型 | checkpoints, logs, metrics |
| 5. 评估与可视化 | 测试模型并分析结果 | metrics, ROC, heatmaps, attention maps |

## 支持的 Patch Encoder

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

## 快速开始

### 1. 安装

```bash
conda create -n tri-mil python=3.10
conda activate tri-mil
pip install -e .
```

### 2. 最简单的推理流程

如果你已经有一个特征文件夹，现在不需要先手工生成 `test.csv`。

```bash
python tri_mil.py infer \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --model_weight_path /path/to/model.pth \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --test_log_dir ./infer_out \
  --no_label
```

现在这条命令会自动帮你：

- 根据 `--feature_dir` 生成内部推理 CSV
- 从特征文件自动推断 `Model.in_dim`
- 自动选择更安全的运行设备并做回退
- 常见推理场景下不再需要手改 YAML

如果你已经有 CSV，旧用法仍然可用：

```bash
python infer_mil.py \
  --yaml_path ./configs/TRANS_MIL.yaml \
  --test_dataset_csv ./test.csv \
  --model_weight_path /path/to/model.pth \
  --test_log_dir ./infer_out \
  --no_label
```

### 3. 最简单的热图流程

如果你已经有预计算好的特征和坐标，现在不需要每次都去改 `draw_heatmap/heatmap.yaml`。

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

这对 `.png/.jpg` 输入，以及“特征和 patch 坐标已经准备好”的情况尤其方便。
如果原图放在多级子目录下，现在热图输出会自动带上原图的相对文件夹前缀，避免不同子目录里的同名切片互相覆盖。

### 4. 预处理与特征提取

普通目录：

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256
```

如果 `./wsis` 下还有标签子目录或更深层目录，加上 `--search_nested`：

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256 --search_nested
```

### 5. 最简单的训练流程

现在训练也可以直接从“特征目录 + 一种标签来源”开始，不需要你手工生成 `train_base.csv`、fold CSV，也不需要手改 YAML 里的数据路径。

特征目录 + reference CSV：

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

特征目录 + 原始数据标签子文件夹：

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

现在会自动完成：

- 生成内部训练 CSV
- 生成 k-fold CSV
- 自动推断 `Model.in_dim`
- 自动推断 `General.num_classes`
- 写出 `label_map.json`
- 自动补齐运行时 YAML，避免手工改数据路径

### 6. 如有需要再手工生成数据集 CSV

现在统一使用 `prepare_dataset_csv.py`，不再区分旧的 `gen_train_csv.py` 和 `gen_test_csv.py`。

情况一：同一级目录里都是特征文件，另外有一个 `reference csv`

```bash
python prepare_dataset_csv.py \
  --mode train_flat \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --reference_csv ./labels.csv \
  --slide_col slide_id \
  --label_col label \
  --output_csv ./train_base.csv
```

情况二：原始数据目录下是标签子文件夹，特征目录是打平保存的

```bash
python prepare_dataset_csv.py \
  --mode train_label_dirs \
  --source_dir ./wsis \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --output_csv ./train_base.csv \
  --source_recursive
```

情况三：特征目录本身已经按标签子文件夹整理好了

```bash
python prepare_dataset_csv.py \
  --mode train_label_dirs \
  --feature_dir ./labeled_features \
  --output_csv ./train_base.csv
```

情况四：只有一个待推理的特征文件夹，没有标签

```bash
python prepare_dataset_csv.py \
  --mode infer \
  --feature_dir ./tri_outputs/20x_256px_0px_overlap/features_vit \
  --output_csv ./test.csv
```

### 7. 训练前做 fold 划分

训练脚本仍然需要 fold CSV，所以把 `train_base.csv` 再转一次：

```bash
python ./split_scripts/split_datasets_k_fold_train_val.py \
  --csv_path ./train_base.csv \
  --dataset_name MY_DATASET \
  --save_dir ./datasets
```

### 8. 手工训练 MIL 模型

```bash
python train_mil.py --yaml_path ./configs/AB_MIL.yaml
```

### 9. 推理或测试

有标签测试：

```bash
python test_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv /path/to/test.csv --model_weight_path /path/to/model.pth --test_log_dir /path/to/test_logs
```

无标签推理：

```bash
python infer_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv ./test.csv --model_weight_path /path/to/model.pth --test_log_dir ./infer_out --no_label
```

## 仓库结构

| 路径 | 作用 |
|---|---|
| `trident/` | 集成的预处理与特征提取后端 |
| `configs/` | MIL 实验配置 |
| `modules/` | MIL 模型实现 |
| `process/` | 训练和测试流程 |
| `datasets/` | 示例数据集 CSV |
| `prepare_dataset_csv.py` | 训练 / 推理 CSV 统一生成入口 |
| `split_scripts/` | 数据集划分脚本 |
| `vis_scripts/` | 可视化工具 |
| `draw_heatmap/` | 热图生成 |
| `feature_extractor/` | 兼容旧流程的遗留工具 |
