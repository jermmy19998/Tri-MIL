import argparse
from utils.yaml_utils import read_yaml,update_config_from_options
from process.process_all import process
import warnings
import os
from utils.general_utils import get_time,merge_k_fold_logs
import tempfile
from utils.runtime_utils import (
    build_kfold_dataset_dir,
    build_train_base_csv_from_reference,
    build_train_base_csv_from_source_dirs,
    clone_config_with_updates,
    collect_feature_files,
    get_pipeline_section,
    get_pipeline_section_compat,
    infer_feature_dim,
    is_pipeline_yaml,
    read_plain_yaml,
    resolve_model_yaml_path,
    select_runtime_device,
    write_label_map_json,
)
warnings.filterwarnings('ignore')

def main(arg):
    pipeline_plain_cfg = read_plain_yaml(arg.yaml_path)
    if is_pipeline_yaml(pipeline_plain_cfg):
        common_cfg = get_pipeline_section(pipeline_plain_cfg, "Common")
        train_cfg = get_pipeline_section_compat(pipeline_plain_cfg, "Train")
        model_yaml_path = resolve_model_yaml_path(arg.yaml_path, pipeline_plain_cfg)

        feature_dir = train_cfg.get("feature_dir") or common_cfg.get("feature_dir")
        output_dir = train_cfg.get("output_dir") or common_cfg.get("output_dir")
        dataset_name = train_cfg.get("dataset_name") or common_cfg.get("dataset_name")
        reference_csv = train_cfg.get("reference_csv") or common_cfg.get("reference_csv")
        source_dir = train_cfg.get("source_dir") or common_cfg.get("source_dir")
        slide_col = train_cfg.get("slide_col", common_cfg.get("slide_col", "slide_id"))
        label_col = train_cfg.get("label_col", common_cfg.get("label_col", "label"))
        feature_recursive = bool(train_cfg.get("feature_recursive", common_cfg.get("feature_recursive", False)))
        source_recursive = bool(train_cfg.get("source_recursive", common_cfg.get("source_recursive", False)))
        k = int(train_cfg.get("k", common_cfg.get("k", 3)))
        seed = int(train_cfg.get("seed", common_cfg.get("seed", 42)))
        device_override = train_cfg.get("device", common_cfg.get("device", "auto"))
        epochs = train_cfg.get("epochs")
        log_dir = train_cfg.get("log_dir") or common_cfg.get("log_dir")
        runtime_num_classes = train_cfg.get("num_classes")

        if not feature_dir or not output_dir or not dataset_name:
            raise ValueError("Pipeline YAML train flow requires feature_dir, output_dir, and dataset_name.")
        if reference_csv is None and source_dir is None:
            raise ValueError("Pipeline YAML train flow requires either reference_csv or source_dir.")

        os.makedirs(output_dir, exist_ok=True)
        if reference_csv is not None:
            base_csv_path, label_map = build_train_base_csv_from_reference(
                reference_csv=reference_csv,
                feature_dir=feature_dir,
                output_csv=os.path.join(output_dir, "train_base.csv"),
                slide_col=slide_col,
                label_col=label_col,
                feature_recursive=feature_recursive,
            )
        else:
            base_csv_path, label_map = build_train_base_csv_from_source_dirs(
                source_dir=source_dir,
                feature_dir=feature_dir,
                output_csv=os.path.join(output_dir, "train_base.csv"),
                feature_recursive=feature_recursive,
                source_recursive=source_recursive,
            )

        dataset_root_dir = build_kfold_dataset_dir(
            base_csv_path=base_csv_path,
            output_dir=os.path.join(output_dir, "datasets"),
            dataset_name=dataset_name,
            seed=seed,
            k=k,
        )
        write_label_map_json(label_map, os.path.join(output_dir, "label_map.json"))

        first_feature_path = collect_feature_files(feature_dir, recursive=feature_recursive)[0]
        first_feature_dim = infer_feature_dim(first_feature_path)
        runtime_device, _ = select_runtime_device(device_override)
        yaml_device_value = "cpu" if runtime_device.type == "cpu" else (
            runtime_device.index if runtime_device.index is not None else 0
        )
        runtime_num_classes = int(runtime_num_classes) if runtime_num_classes is not None else len(label_map)
        log_dir = log_dir or os.path.join(output_dir, "logs")
        index_to_label = {index: label for label, index in label_map.items()}

        def update_runtime_config(config):
            config["Dataset"]["DATASET_NAME"] = dataset_name
            config["Dataset"]["dataset_root_dir"] = dataset_root_dir
            config["Dataset"]["dataset_csv_path"] = None
            config["Logs"]["log_root_dir"] = log_dir
            config["General"]["seed"] = seed
            config["General"]["device"] = yaml_device_value
            config["General"]["num_classes"] = runtime_num_classes
            config["Model"]["in_dim"] = first_feature_dim
            if epochs is not None:
                config["General"]["num_epochs"] = int(epochs)
            config["Label"] = {index_to_label[idx]: idx for idx in sorted(index_to_label)}
            return config

        temp_dir = tempfile.mkdtemp(prefix="tri_mil_train_")
        yaml_path = clone_config_with_updates(
            base_config_path=model_yaml_path,
            output_path=os.path.join(temp_dir, "runtime_train.yaml"),
            updater=update_runtime_config,
        )
    else:
        yaml_path = arg.yaml_path

    print(f"MIL-yaml path: {yaml_path}")
    args = read_yaml(yaml_path)
    # dinamically update the config file with the options
    if arg.options:
        args = update_config_from_options(args,arg.options)
    
    if args.Dataset.dataset_root_dir == {} and args.Dataset.dataset_csv_path != None:
        '''
        None-fold split
        '''
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{get_time()}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        process(args,yaml_path,arg.options)

    else:
        '''
        k-fold split
        '''
        dataset_root_dir = args.Dataset.dataset_root_dir
        k_fold_csv_paths = sorted([os.path.join(dataset_root_dir,path) for path in os.listdir(dataset_root_dir)])
        process_time = get_time()
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):
            args.Dataset.dataset_csv_path = k_fold_csv_path
            now_fold = k_idx+1
            args.Dataset.now_fold = now_fold
            log_root_dir = args.Logs.log_root_dir
            os.makedirs(log_root_dir,exist_ok=True)
            sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
            os.makedirs(sub_dir,exist_ok=True)
            if now_fold != None:
                fold_dir = f'fold_{now_fold}'
                args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}/{fold_dir}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            process(args,yaml_path,arg.options)
            print(f'K-Fold:{k_idx+1} Done!')
        fold_total_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/yaml',help='path to MIL-yaml file')
    parser.add_argument('--options',nargs='+',help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the yaml config file')
    arg = parser.parse_args()
    main(arg)
