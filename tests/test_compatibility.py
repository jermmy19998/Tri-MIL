"""Regressions for retained encoder helpers and repaired existing utilities."""

import ast
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from addict import Dict as Config

from modules.AC_MIL.nystrom_attention import NystromAttention
from utils.yaml_utils import read_yaml, update_config_from_options


def test_masked_nystrom_attention_has_finite_backward():
    """Exercise the masked attention branch that previously referenced undefined names."""
    attention = NystromAttention(dim=8, dim_head=4, heads=2, num_landmarks=2)
    features = torch.randn(1, 4, 8, requires_grad=True)
    output = attention(features, mask=torch.tensor([[True, True, True, False]]))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(features.grad).all()


def test_checkpoint_helpers_preserve_vit_backbone_keys(tmp_path):
    """Verify retained checkpoint helpers without loading optional pretrained encoders."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "trident/patch_encoder_models/load.py").read_text(encoding="utf-8"))
    names = {
        "_strip_state_dict_prefixes",
        "_extract_checkpoint_state_dict",
        "_load_local_checkpoint",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == 3
    namespace = {"torch": torch, "Any": Any, "Dict": Dict}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<checkpoint helpers>", "exec"),
        namespace,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                "module.backbone.block.weight": torch.ones(2),
                "module.head.weight": torch.ones(3),
            }
        },
        path,
    )
    loaded = namespace["_load_local_checkpoint"](str(path))
    assert set(loaded) == {"block.weight"}
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "encoder_registry"
            for target in node.targets
        )
    )
    assert {"vit", "gigapath-flash", "virchow2-cls", "phaet", "mascaret"} <= {
        key.value for key in registry.keys
    }


def test_yaml_overrides_and_safe_loading(tmp_path):
    """Parse false booleans, equals in strings and reject Python YAML constructors."""
    config = Config({"General": {"device": 0, "enabled": True}, "Model": {"path": ""}})
    update_config_from_options(
        config, ["General.device=cpu", "General.enabled=false", "Model.path=a=b.pt"]
    )
    assert config.General.device == "cpu" and config.General.enabled is False
    assert config.Model.path == "a=b.pt"
    path = tmp_path / "config.yaml"
    path.write_text("!!python/object/apply:builtins.print [unsafe]", encoding="utf-8")
    with pytest.raises(Exception):
        read_yaml(path)
