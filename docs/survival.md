# Survival prediction / 生存预测

Tri-MIL 0.4.0 adds Cox proportional-hazards training and Harrell's C-index for
right-censored outcomes. Classification remains the default task.

## Data / 数据

Prepare a manifest with `slide_path,time,event,case_id` columns. `slide_path`
points to a `.pt` feature tensor/dictionary (`features` or `feats`) or a `.h5`
file with a `features` dataset. Each bag has shape `[patches, feature_dim]`.
Use absolute feature paths. Times must be finite and nonnegative, in one
consistent unit. Set `event=1` for an observed event and `event=0` for right
censoring; a dataset's `censorship=1` must therefore be converted to `event=0`.

清单的 `time` 是随访时间，`event=1` 表示观察到事件，`event=0` 表示右删失。
不要将生存状态当作普通分类标签；不能省略删失信息。

```bash
python prepare_survival_csv.py --manifest ./datasets/survival_manifest.csv --output ./datasets/survival_splits.csv --seed 2024 --val_fraction 0.2 --test_fraction 0.2
```

Splitting uses `case_id`, keeping slides from one case together. This split is
random by case, without event stratification; inspect event counts before
training. For patient-level survival, use one feature bag per patient (for
example, concatenate their slide features before preparing the manifest).
Multiple slide bags per patient otherwise weight that patient multiple times.
All slides of a case must have consistent time and event outcomes.

生成的划分 CSV 包含下列列；不同长度的划分用空行补齐，路径和结局按行对齐：

| Train | Validation | Test |
|---|---|---|
| `train_slide_path` | `val_slide_path` | `test_slide_path` |
| `train_time` | `val_time` | `test_time` |
| `train_event` | `val_event` | `test_event` |
| `train_case_id` | `val_case_id` | `test_case_id` |

You can supply existing splits directly. Case columns are optional for existing
splits, but then only path overlap can be checked. Missing outcomes on a row
with a path are errors. Training needs at least one observed event and validation
needs at least one comparable pair. Test data may be omitted.

## Train, validate, predict / 训练、验证、推理

Edit `Model.in_dim` in `configs/SURVIVAL_AB_MIL.yaml` to match the extracted
features. Edit the paths in `configs/SURVIVAL_PIPELINE.yaml` and run:

```bash
python train_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python valid_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/SURVIVAL_PIPELINE.yaml
```

`General.task: survival` selects survival mode. The unified pipeline accepts
`Common.task`, paths and device, plus `Train.epochs`, `Train.output_dir`,
`Valid.split`, `Valid.output_dir`, `Infer.split`, and `Infer.output_dir`.
`Train.dataset_csv_path`, `Valid.valid_dataset_csv`, and `Infer.infer_dataset_csv`
can override the shared CSV. A model YAML is also accepted for training:

```bash
python train_mil.py --yaml_path ./configs/SURVIVAL_AB_MIL.yaml --options General.num_epochs=10 General.device=cpu
python valid_mil.py --yaml_path ./configs/SURVIVAL_AB_MIL.yaml --valid_dataset_csv ./datasets/survival_splits.csv --model_weight_path ./log_dir/survival/Best_survival.pth --device cpu
python infer_mil.py --yaml_path ./configs/SURVIVAL_AB_MIL.yaml --test_dataset_csv ./datasets/survival_splits.csv --model_weight_path ./log_dir/survival/Best_survival.pth --test_log_dir ./log_dir/survival/infer --device cpu
```

推理 CSV 只需要对应划分的路径列，例如 `test_slide_path`，不需要时间或事件。
输出 `risk` 是相对 log hazard，越大代表预测风险越高、预期生存越短；它不是
生存时间或概率。当前不拟合基线生存函数，也不生成校准生存曲线。

## Outputs and metrics / 输出与指标

Training writes `Best_survival.pth`, `Last_survival.pth`, `survival_log.csv`,
`survival_metrics.json`, `val_predictions.csv`, and, when available,
`test_predictions.csv`. Checkpoints contain backbone name, task, configuration,
epoch and validation C-index. Keep the matching model YAML when reloading.
Validation/prediction write `valid_survival_predictions.csv` /
`infer_survival_predictions.csv` and corresponding metric JSONs.

- The Cox loss uses **Breslow ties** and numerically stable log-sum-exp risk sets.
- Each epoch uses the **full training cohort** as its risk set. Autograd retains
  a graph per bag until backward, so training memory scales with the cohort.
  Choose a smaller backbone or fewer features per bag for large datasets.
- The optimizer settings in the model YAML are used. This initial survival
  workflow uses a constant learning rate; classification scheduler settings
  are not used. Early stopping honors `General.earlystop.use/patience`.
- The best checkpoint maximizes **validation C-index**. Test data is evaluated
  once after selection and is never used to choose a checkpoint.
- Harrell's C-index gives tied risks half credit. An event at an earlier time
  is comparable to a later subject; equal-time event/censor pairs are also
  comparable, while equal-time event/event pairs are excluded.
- No comparable pairs gives `NaN` in the Python metric and JSON `null` in
  exported metrics. All-censored evaluation returns a null Cox loss. This
  implementation is Harrell's C, without IPCW/Uno weighting.

当前提供 AB_MIL 的完整示例。其他能返回单个标量 `logits` 的 MIL 可复用该流程，
加载时自动设 `num_classes=1`；多阶段 DTFD、预处理型 CDP、坐标型 LONG_MIL 和
多头 CLAM_MB 需要专用适配，当前会明确拒绝。生存训练不使用分类专用的辅助损失
或采样机制。生存流程目前读取预先划分的 CSV，不自动执行分类的 k-fold 流程。
