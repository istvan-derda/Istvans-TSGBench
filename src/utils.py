import os
import pickle
import json
import torch
import numpy as np

PREPROCESSING_PARAS = ['do_preprocessing','original_data_path','output_ori_path','dataset_name','use_ucr_uea_dataset','ucr_uea_dataset_name','seq_length','valid_ratio','do_normalization']
GENERATION_PARAS = ['do_generation','model','dataset_name']

def show_divider():
    print("=" * 20)

def show_with_start_divider(content):
    show_divider()
    print(content)

def show_with_end_divider(content):
    print(content)
    show_divider()
    print()

def make_sure_path_exist(path):
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

def write_json_data(content, path):
    make_sure_path_exist(path)
    with open(path, 'w') as json_file:
        json.dump(content, json_file, indent=4)

def determine_device(cuda_device):
    # Determine device (cpu/gpu)
    if cuda_device == None or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count()>1:
            device = torch.device('cuda', cuda_device)
        else:
            device = torch.device('cuda', 0)
    return device
