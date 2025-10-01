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

    return dataset_names[args.dataset_no]
