"""Synchronize bilingual README model tables with implemented model configurations."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "AS_MIL": "Attention-Stabilized Multiple Instance Learning",
    "ATTRI_MIL": "Attribute-based Multiple Instance Learning",
    "CAR_MIL": "Counterfactual Attention Regularization for Multiple Instance Learning",
    "FC_MIL": "[FC-MIL](https://github.com/7FFDW/FCMIL)",
    "LIN_MIL": "Linear-attention Multiple Instance Learning",
    "MR_AB_MIL": "[Attention MIL with manifold residual projections](https://github.com/BearCleverProud/MR-Block)",
    "NN_MIL": "Attention MIL with random feature subspaces and ensemble uncertainty",
    "PNEA_MIL": "Positive-Negative Evidence Analysis for Multiple Instance Learning",
    "GATE_AB_MIL": "Gated Attention-based Multiple Instance Learning",
    "INSMIX_MIL": "Instance Mixup Multiple Instance Learning",
}


def main():
    """Generate complete model tables while preserving existing paper references."""
    english_path = ROOT / "README.md"
    english = english_path.read_text(encoding="utf-8")
    start = english.index("## Supported MIL Models")
    end = english.index("\n## ", start + 1)
    old_section = english[start:end]
    references = {}
    method_column = 3 if "| Model | Configuration |" in old_section else 2
    for line in old_section.splitlines():
        if not line.startswith("| `"):
            continue
        fields = line.split("|")
        for model in re.findall(r"`([^`]+)`", fields[1]):
            references[model] = fields[method_column].strip()
    references.update(DESCRIPTIONS)
    models = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if path.stem.startswith("SURVIVAL_") or path.stem == "TRI_MIL_PIPELINE":
            continue
        if not (ROOT / "modules" / path.stem).is_dir():
            raise ValueError(f"Missing implementation directory for {path.stem}")
        models.append(path.stem)
    rows = [
        f"| `{name}` | [{name}.yaml](./configs/{name}.yaml) | {references.get(name, DESCRIPTIONS.get(name, name))} |"
        for name in models
    ]
    table_en = "| Model | Configuration | Paper / Method |\n|---|---|---|\n" + "\n".join(rows)
    section_en = (
        f"## Supported MIL Models\n\nTri-MIL includes **{len(models)} MIL models**, each with a model configuration and implementation. "
        "The list below matches the model YAMLs in `configs/`; optional dependencies vary by model.\n\n"
        + table_en
        + "\n\nSurvival is a separate task using compatible scalar-output backbones. "
        "See [SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) and the "
        "[survival guide](./docs/survival.md) for supported interfaces and limitations.\n"
    )
    english_path.write_text(
        english[:start] + section_en + english[end:], encoding="utf-8", newline="\n"
    )
    chinese_path = ROOT / "README_CN.md"
    chinese = chinese_path.read_text(encoding="utf-8")
    heading = "## 支持的 MIL 模型"
    section_cn = (
        f"{heading}\n\nTri-MIL 包含 **{len(models)} 个 MIL 模型**，均提供模型配置和实现。"
        "下表与 `configs/` 中的模型 YAML 对应；不同模型可能需要额外依赖。\n\n"
        + table_en.replace(
            "| Model | Configuration | Paper / Method |", "| 模型 | 配置文件 | 论文 / 方法 |"
        )
        + "\n\n生存预测是独立任务，复用兼容的单标量输出骨干，并非表中所有模型都支持生存训练。"
        "详见 [SURVIVAL_AB_MIL.yaml](./configs/SURVIVAL_AB_MIL.yaml) 和"
        "[生存预测说明](./docs/survival.md)中的适配范围与限制。\n"
    )
    if heading in chinese:
        start = chinese.index(heading)
        end = chinese.find("\n## ", start + 1)
        chinese = chinese[:start] + section_cn + (chinese[end:] if end >= 0 else "")
    else:
        chinese = chinese.rstrip() + "\n\n" + section_cn
    chinese_path.write_text(chinese, encoding="utf-8", newline="\n")
    print(f"Updated both README tables: {len(models)} models each.")


if __name__ == "__main__":
    main()
