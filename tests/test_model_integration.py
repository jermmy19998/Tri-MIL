"""Exercise newly integrated MIL models and Tri-MIL dataset compatibility."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from utils.model_utils import get_model_from_yaml
from utils.nnmil_utils import fixed_bag_collate
from utils.yaml_utils import read_yaml

NEW_MODELS = [
    "AS_MIL",
    "ATTRI_MIL",
    "CAR_MIL",
    "FC_MIL",
    "LIN_MIL",
    "MR_AB_MIL",
    "NN_MIL",
    "PNEA_MIL",
]


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_new_model_factory_forward_backward(model_name):
    """Construct every added model through YAML and backpropagate classification logits."""
    root = Path(__file__).resolve().parents[1]
    args = read_yaml(root / "configs" / f"{model_name}.yaml")
    args.Model.in_dim = 16
    args.General.num_classes = 2
    args.Model.hidden_dim = 16
    args.Model.latent_dim = 16
    args.Model.attention_dim = 8
    args.Model.num_heads = 2
    args.Model.rank = 4
    args.Model.num_tokens = 4
    args.Model.token_drop = 1
    args.Model.transformer_depth = 1
    model = get_model_from_yaml(args)
    model.train()
    logits = model(torch.randn(1, 12, 16))["logits"]
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 12, 16))["logits"].shape == (1, 2)


def test_nnmil_collate_accepts_tri_mil_paths():
    """Accept the three-field feature, label, path contract used by Tri-MIL."""
    bags, labels, masks = fixed_bag_collate([(torch.randn(3, 8), torch.tensor(1), "slide.pt")], 5)
    assert bags.shape == (1, 5, 8) and labels.tolist() == [1]
    assert masks.sum().item() == 3


@pytest.mark.parametrize("model_name", [name for name in NEW_MODELS if name != "NN_MIL"])
def test_new_classification_training_entrypoint(model_name, tmp_path):
    """Run a complete CPU training epoch through each new model's dedicated process."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8"))
    config["General"].update(num_epochs=1, num_workers=0, device="cpu")
    config["Model"].update(
        in_dim=16,
        hidden_dim=16,
        latent_dim=16,
        attention_dim=8,
        num_heads=2,
        rank=4,
        num_tokens=4,
        token_drop=1,
        transformer_depth=1,
    )
    config["Model"]["scheduler"].update(warmup=1, which="none")
    columns = {}
    for split in ["train", "val", "test"]:
        paths = []
        for index in range(4):
            path = tmp_path / f"{split}_{index}.pt"
            torch.save(torch.randn(12, 16), path)
            paths.append(str(path))
        columns[f"{split}_slide_path"] = paths
        columns[f"{split}_label"] = [0, 1, 0, 1]
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(columns).to_csv(csv, index=False)
    config["Dataset"].update(dataset_csv_path=str(csv), dataset_root_dir={})
    config["Logs"]["log_root_dir"] = str(tmp_path / "logs")
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "train_mil.py", "--yaml_path", str(config_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list((tmp_path / "logs").rglob("Last_EPOCH_1.pth"))
