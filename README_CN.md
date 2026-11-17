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

## 0.4.0 更新

- 新增 8 个 MIL 模型：`AS_MIL`、`ATTRI_MIL`、`CAR_MIL`、`FC_MIL`、
  `LIN_MIL`、`MR_AB_MIL`、`NN_MIL`、`PNEA_MIL`，包含 YAML、模型工厂和训练流程。
- 同步 TRIDENT 0.3.2：可配置编码器输入尺寸，新增 GigaPath Flash、Virchow2 CLS、
  Phaet、Mascaret 编码器，更新 BioIO 转换、WSI 读取、处理及状态记录；保留原有 ViT-S。
- 新增生存预测：Cox 损失、右删失处理、验证集 C-index 选模，以及无标签风险推理。
- 接入 `split_by_case/` 病例级划分；为 Python 函数补齐 docstring，统一格式并加入回归测试。

同步依据是 2026-11-16 提供的本地源码快照。详见[更新日志](./CHANGELOG.md)、
[上游输入哈希](./docs/upstream-sync.json)及[生存预测完整说明](./docs/survival.md)。

### 生存预测快速开始

先准备包含 `slide_path,time,event,case_id` 的清单。`time` 是随访时间，
`event=1` 表示已观察到事件，`event=0` 表示右删失；特征支持 PT/HDF5。
病例级划分会把同一患者的切片放在同一组，避免数据泄漏：

```bash
python prepare_survival_csv.py --manifest ./datasets/survival_manifest.csv --output ./datasets/survival_splits.csv
# 先修改两个 SURVIVAL YAML 中的特征维度和路径。
python train_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python valid_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
```

配置示例：[SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) 和
[SURVIVAL_PIPELINE.yaml](./configs/SURVIVAL_PIPELINE.yaml)。划分 CSV 使用
`train_slide_path,train_time,train_event` 及对应的 `val_`、`test_` 列；推理只需路径列。
输出 `risk` 越大表示预测风险越高，它不是生存时间或生存概率。

C-index 对并列风险记 0.5 分，无可比较样本时输出 JSON `null`。训练使用精确的全队列
Cox 风险集，反向传播前保留所有训练 bag 的计算图，显存占用随队列规模增长。
患者级生存预测建议每位患者一个特征 bag；专用多阶段或坐标模型需要额外适配，
支持范围、删失约定及输出字段见完整说明。

### 开发验证

支持 Python 3.10–3.12。以下核心测试无需下载预训练权重：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
python -m pytest -q
python tools/document_functions.py
ruff check .
ruff format --check .
```

完整 WSI 和预训练编码器验证还需要原生读取依赖、模型权重及真实切片数据。

## 一眼看懂

| 模块 | 作用 | 主要路径 |
|---|---|---|
| 预处理 | 读取 WSI、组织分割、坐标生成、特征提取 | `trident/`, `run_batch_of_slides.py`, `run_single_slide.py` |
| 训练 | 基于配置文件训练 MIL 模型 | `configs/`, `modules/`, `process/`, `train_mil.py` |
| 评估 | 验证模型并导出指标或推理结果 | `valid_mil.py`, `infer_mil.py` |
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

### 2. 统一 YAML 工作流

Tri-MIL 现在支持一份统一 pipeline YAML，同时驱动 `train_mil.py`、`valid_mil.py`、`infer_mil.py` 和 `draw_heatmap.py`。

可以直接使用 [configs/TRI_MIL_PIPELINE.yaml](./configs/TRI_MIL_PIPELINE.yaml)：

```bash
python train_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python valid_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python draw_heatmap.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
```

推荐约定：

- `Common.model_yaml_path` 指向具体 MIL 模型 YAML，例如 `./configs/TRANS_MIL.yaml`
- `Train`、`Valid`、`Infer`、`Heatmap` 只放各自阶段需要的运行参数
- `draw_heatmap.py` 默认会读自己的通用 `heatmap.yaml`，如果传统一 pipeline yaml，则会自动合并 `Heatmap` 配置和 `Common.model_yaml_path`

### 3. 预处理与特征提取

普通目录：

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256
```

如果 `./wsis` 下还有标签子目录或更深层目录，加上 `--search_nested`：

```bash
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./tri_outputs --patch_encoder vit --mag 20 --patch_size 256 --search_nested
```

### 4. 如有需要再手工生成数据集 CSV

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

### 5. 训练前做 fold 划分

训练脚本仍然需要 fold CSV，所以把 `train_base.csv` 再转一次：

```bash
python ./split_scripts/split_datasets_k_fold_train_val.py \
  --csv_path ./train_base.csv \
  --dataset_name MY_DATASET \
  --save_dir ./datasets
```

### 6. 训练 MIL 模型

```bash
python train_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
```

### 7. 推理或测试

有标签测试：

```bash
python valid_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
```

无标签推理：

```bash
python infer_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
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

## 支持的 MIL 模型

Tri-MIL 包含 **58 个 MIL 模型**，均提供模型配置和实现。下表与 `configs/` 中的模型 YAML 对应；不同模型可能需要额外依赖。

| 模型 | 配置文件 | 论文 / 方法 |
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

生存预测是独立任务，复用兼容的单标量输出骨干，并非表中所有模型都支持生存训练。详见 [SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) 和[生存预测说明](./docs/survival.md)中的适配范围与限制。
