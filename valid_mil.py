import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
import shutil
import os
from utils.model_utils import get_model_from_yaml,get_criterion
from utils.runtime_utils import (
    get_pipeline_section,
    get_pipeline_section_compat,
    is_pipeline_yaml,
    load_torch_checkpoint,
    read_plain_yaml,
    resolve_model_yaml_path,
    select_runtime_device,
)
warnings.filterwarnings('ignore')

def valid(args):
    plain_cfg = read_plain_yaml(args.yaml_path)
    model_yaml_path = resolve_model_yaml_path(args.yaml_path, plain_cfg if is_pipeline_yaml(plain_cfg) else None)
    valid_cfg = get_pipeline_section_compat(plain_cfg, "Valid", ("Test",)) if is_pipeline_yaml(plain_cfg) else {}
    common_cfg = get_pipeline_section(plain_cfg, "Common") if is_pipeline_yaml(plain_cfg) else {}

    print(f"MIL-model-yaml path: {model_yaml_path}")
    yaml_args = read_yaml(model_yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    valid_dataset_csv = (
        args.valid_dataset_csv
        or args.test_dataset_csv
        or valid_cfg.get("valid_dataset_csv")
        or valid_cfg.get("test_dataset_csv")
        or common_cfg.get("valid_dataset_csv")
        or common_cfg.get("test_dataset_csv")
    )
    if not valid_dataset_csv:
        raise ValueError("valid_dataset_csv is required for valid_mil.")
    print(f"Dataset csv path: {valid_dataset_csv}")
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        test_ds = CDP_MIL_WSI_Dataset(valid_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(valid_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    test_ds = WSI_Dataset(valid_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    model_weight_path = (
        args.model_weight_path
        or valid_cfg.get("model_weight_path")
        or common_cfg.get("model_weight_path")
    )
    if not model_weight_path:
        raise ValueError("model_weight_path is required for valid_mil.")
    print(f"Model weight path: {model_weight_path}")
    runtime_device, device_message = select_runtime_device(
        args.device if args.device is not None else valid_cfg.get("device", common_cfg.get("device", yaml_args.General.device))
    )
    print(f"[INFO] {device_message}")
    device = runtime_device
    criterion = get_criterion(yaml_args.Model.criterion)
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier,attention,dimReduction,attCls = get_model_from_yaml(yaml_args)
        state_dict = load_torch_checkpoint(model_weight_path, map_location=device)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier,attention,dimReduction,attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(load_torch_checkpoint(model_weight_path, map_location=device))

    
    # CLAM_SB_MIL and CLAM_MB_MIL models have different val loop pipeline (has instance loss)
    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        test_loss,test_metrics =  ds_val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        test_loss,test_metrics =  dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,yaml_args.Model.num_Group,yaml_args.Model.grad_clipping,yaml_args.Model.distill,yaml_args.Model.total_instance)
    else:
        test_loss,test_metrics =  val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print('----------------INFO----------------\n')
    print(f'{FAIL}Valid_Loss:{ENDC}{test_loss}\n')
    print(f'{FAIL}Valid_Metrics:  {ENDC}{test_metrics}\n')
    
    valid_log_dir = (
        args.valid_log_dir
        or args.test_log_dir
        or valid_cfg.get("valid_log_dir")
        or valid_cfg.get("test_log_dir")
        or common_cfg.get("valid_log_dir")
        or common_cfg.get("test_log_dir")
    )
    if not valid_log_dir:
        raise ValueError("valid_log_dir is required for valid_mil.")
    os.makedirs(valid_log_dir,exist_ok=True)
    new_yaml_path = os.path.join(valid_log_dir,f'Valid_{model_name}.yaml')
    shutil.copyfile(model_yaml_path,new_yaml_path)
    new_test_dataset_csv_path = os.path.join(valid_log_dir,f'Valid_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
    shutil.copyfile(valid_dataset_csv,new_test_dataset_csv_path)
    test_log_path = os.path.join(valid_log_dir,f'Valid_Log_{model_name}.txt')
    log_to_save = {'test_loss':test_loss,'test_metrics':test_metrics}
    with open(test_log_path,'w') as f:
        f.write(str(log_to_save))
    print(f"Valid log saved at: {test_log_path}")
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/config-yaml',help='path to model yaml or pipeline yaml')
    parser.add_argument('--valid_dataset_csv',type=str,default=None,help='path to validation dataset csv')
    parser.add_argument('--test_dataset_csv',type=str,default=None,help='legacy alias of --valid_dataset_csv')
    parser.add_argument('--model_weight_path',type=str,default='/path/to/your/model-weight',help='path to model weights ')
    parser.add_argument('--valid_log_dir',type=str,default=None,help='path to validation log dir')
    parser.add_argument('--test_log_dir',type=str,default=None,help='legacy alias of --valid_log_dir')
    parser.add_argument('--device',type=str,default=None,help='runtime device override')
    args = parser.parse_args()
    valid(args)

