import functools
import os
import argparse
from datetime import datetime
import sys
import numpy as np

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, '../data/')

this_model_root =  os.path.abspath(os.path.dirname(sys.argv[0]))
gen_data_dir = os.path.join(this_model_root, 'gen/')

def load_train_data():   
    ori_data_dir = os.path.join(data_dir, 'ori/')
    dataset_name = get_dataset_name()
    dataset_path = f'{ori_data_dir}{dataset_name}/{dataset_name}_train.npy'
    train_data = np.load(dataset_path)
    return train_data


def load_valid_data():   
    ori_data_dir = os.path.join(data_dir, 'ori/')
    dataset_name = get_dataset_name()
    dataset_path = f'{ori_data_dir}{dataset_name}/{dataset_name}_valid.npy'
    valid_data = np.load(dataset_path)
    return valid_data



def persist_gen_data(data):
    now = datetime.now()
    dataset_name = get_dataset_name()
    dir_path = os.path.join(gen_data_dir, dataset_name)
    file_name = f'{dataset_name}_gen_{now}.npy'
    file_path = os.path.join(dir_path, file_name)
    os.makedirs(dir_path, exist_ok=True)
    np.save(file_path, data)

@functools.lru_cache(maxsize=None)
def get_dataset_name():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_no')
    args, _ = parser.parse_known_args()

    dataset_names = {
        '2': 'D2_stock',
        '3': 'D3_stock_long',
        '4': 'D4_exchange',
        '5': 'D5_energy',
        '6': 'D6_energy_long',
        '7': 'D7_eeg'
    }

    if args.dataset_no is None:
        raise ValueError("--dataset_no must be provided (e.g., '--dataset_no 2' or '--dataset_no=2')")

    # Remove both '--dataset_no' and its value safely
    new_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue

        if arg.startswith("--dataset_no"):
            # Handle both '--dataset_no 2' and '--dataset_no=2'
            if arg == "--dataset_no" and i + 1 < len(sys.argv):
                skip_next = True
            continue  # skip this arg (and possibly its value)
        else:
            new_argv.append(arg)

    sys.argv[:] = new_argv  # modify in place so other parsers see clean args


    return dataset_names[args.dataset_no]
