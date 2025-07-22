import pickle
from urllib import request
import zipfile
import gzip
import shutil
import os
import time
import sys
import re

import mgzip
import pandas as pd
import numpy as np
from scipy.io import arff
from pyprojroot import here
from sklearn.preprocessing import MinMaxScaler

raw_data_dir = here('data/ori_raw')
ori_data_dir = here('data/ori')

urls = {
    'D2_D3': 'https://github.com/jsyoon0823/TimeGAN/raw/refs/heads/master/data/stock_data.csv',
    'D4': 'https://github.com/laiguokun/multivariate-time-series-data/raw/refs/heads/master/exchange_rate/exchange_rate.txt.gz',
    'D5_D6': 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip',
    'D7': 'https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip'
}


def main():
    populate_D2_D3_stock()
    populate_D2a_D3a_stock_novolume()
    populate_D4_exchange()
    populate_D5_D6_energy()
    populate_D7_eeg()


def populate_D2_D3_stock():
    print_populate_start('D2_D3_stock')
    temp_path = download_dataset(url=urls['D2_D3'])
    raw_data_path = persist_raw(temp_path, 'D2_D3_stock.csv')

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D2_stock',
        seq_length=24
    )

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D3_stock_long',
        seq_length=128 # TSGBench ori 125
    )


def populate_D4_exchange():
    print_populate_start('D4_exchange')
    temp_path = download_dataset(url=urls['D4'])
    temp_path = ungzip(temp_path)
    raw_data_path = persist_raw(temp_path, 'D4_exchange.csv')

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D4_exchange',
        seq_length=128 # TSGBench ori 125
    )


def populate_D5_D6_energy():
    print_populate_start('D5_D6_energy')
    temp_path = download_dataset(url=urls['D5_D6'])
    temp_path = unzip(temp_path, data_path_in_zip='energydata_complete.csv')
    temp_path = remove_csv_column(temp_path, 'date')
    raw_data_path = persist_raw(temp_path, 'D5_D6_energy.csv')

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D5_energy', 
        seq_length=24
    )
    
    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D6_energy_long', 
        seq_length=128 # TSGBench ori 125
    )


def populate_D7_eeg():
    print_populate_start('D7_eeg')
    temp_path = download_dataset(url=urls['D7'])
    temp_path = unzip(temp_path, data_path_in_zip='EEG Eye State.arff')
    temp_path = arff2csv(temp_path)
    temp_path = remove_csv_column(temp_path, 'eyeDetection')
    raw_data_path = persist_raw(temp_path, 'D7_eeg.csv')

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D7_eeg',
        seq_length=128
    )
   

def print_populate_start(dataset_name):
    print("""
=====================================
          """)
    print(f"Retrieving {dataset_name} dataset")


def download_dataset(url):
    file_path, _ = request.urlretrieve(url, reporthook=urlretrieve_reporthook)
    print("\n")
    return file_path


def persist_raw(temp_path, filename):
    os.makedirs(raw_data_dir, exist_ok=True)
    dest_path = os.path.join(raw_data_dir, filename)
    print(f"Persisting {filename} at {dest_path}")
    shutil.move(temp_path, dest_path)
    return dest_path


def ungzip(path):
    uncompressed_path = path + '.unzip'
    with gzip.open(path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                return uncompressed_path
            

def unzip(path, data_path_in_zip):
    uncompressed_path = path + '.unzip'
    data_path = os.path.join(uncompressed_path, data_path_in_zip)
    with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(uncompressed_path)
            return data_path


def remove_csv_column(file_path, col_name):
    data = pd.read_csv(file_path)
    if not col_name in data.columns:
        return
    print(f'Removing column {col_name}')
    data = data.drop([col_name], axis='columns')
    data.to_csv(file_path, index=False)
    return file_path


def arff2csv(file_path):
    out_path = re.sub('.arff', '', file_path) + '.csv'
    data = arff.loadarff(file_path)
    data = pd.DataFrame(data[0])
    data.to_csv(out_path, index=False)
    return out_path


def urlretrieve_reporthook(count, block_size, _):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_bytes = int(count * block_size)
    kb_s = int(progress_bytes / (1024 * duration))
    sys.stdout.write(f"\r... {round(progress_bytes / (1024 * 1024), 1)} MB, {kb_s} KB/s, {round(duration)} seconds passed")
    sys.stdout.flush()


def preprocess_data(ori_data_path, dataset_name, seq_length, valid_ratio = 0.1):
    print(f"Preprocessing {dataset_name}")

    df = pd.read_csv(ori_data_path)

    # interpolate missing values
    df = df.interpolate(axis=0) # tsgbench uses other axis oO
   
    # scale data to feature range 0..1
    df = (df - df.min().min()) / (df.max().max() - df.min().min())

    windowed_data = sliding_window_view(df.to_numpy(), seq_length)

    # Shuffle
    idx = np.random.permutation(len(windowed_data))
    data = windowed_data[idx]

    valid_len = int(data.shape[0] * (valid_ratio)) 
    valid_data = data[:valid_len]
    train_data = data[valid_len:]

    dest_dir = str(here(f"data/ori/{dataset_name}/"))
    print(f"Persisting preprocessed {dataset_name} to {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    with mgzip.open(f"{dest_dir}/{dataset_name}_train.pkl", 'w') as file:
        pickle.dump(train_data, file)
    with mgzip.open(f"{dest_dir}/{dataset_name}_valid.pkl", 'w') as file:
        pickle.dump(valid_data, file)


def sliding_window_view(data, window_size, step=1):
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L - window_size + 1
    
    # Shape of the output array
    new_shape = (B, window_size, C)
    
    # Calculate strides
    original_strides = data.strides
    new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    return strided_array


if __name__ == "__main__":
    main()
