import yaml
from addict import Dict
import shutil
import os

try:
    from ruamel.yaml import YAML as RuamelYAML
except ImportError:
    RuamelYAML = None


def read_yaml(fpath=None):
    """Load a UTF-8 YAML mapping without constructing arbitrary Python objects."""
    with open(fpath, mode="r", encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        if not isinstance(yml, dict):
            raise ValueError("The YAML root must be a mapping.")
        return Dict(yml)


def update_config_from_options(config, options):
    """Apply dotted key=value overrides with YAML scalar, list and boolean parsing."""
    for option in options:
        key, value = option.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = yaml.safe_load(value)
    return config


def change_yaml_by_options(yaml_path, options):
    """Change yaml by options."""
    if RuamelYAML is not None:
        yaml_handler = RuamelYAML(typ="rt")
        with open(yaml_path, "r", encoding="utf-8") as file:
            config = yaml_handler.load(file)

        update_config_from_options(config, options)

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml_handler.dump(config, file)
        return

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    update_config_from_options(config, options)

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def save_yaml(args, yaml_path, options):
    """Save yaml."""
    dst_path = os.path.join(args.Logs.now_log_dir, os.path.basename(yaml_path))
    shutil.copyfile(yaml_path, dst_path)
    if options != None:
        change_yaml_by_options(dst_path, options)
