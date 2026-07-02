import yaml
from addict import Dict
import shutil
import os

try:
    from ruamel.yaml import YAML as RuamelYAML
except ImportError:
    RuamelYAML = None


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
def update_config_from_options(config, options):
    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)
    return config

def change_yaml_by_options(yaml_path, options):
    if RuamelYAML is not None:
        yaml_handler = RuamelYAML(typ='rt')
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml_handler.load(file)

        for option in options:
            key, value = option.split('=')
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = type(d[keys[-1]])(value)

        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml_handler.dump(config, file)
        return

    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)



def save_yaml(args,yaml_path,options):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    shutil.copyfile(yaml_path,dst_path)
    if options != None:
        change_yaml_by_options(dst_path,options)
