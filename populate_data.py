from urllib import request
import zipfile
import gzip
import shutil
import os
import time
import sys
import pandas as pd
from scipy.io import arff
import pathlib
import re


from project_root import project_root
from src.preprocess import preprocess_data

raw_data_dir = os.path.join(project_root, 'data/ori_raw')

urls = {
    'D2_D3': 'https://github.com/jsyoon0823/TimeGAN/raw/refs/heads/master/data/stock_data.csv',
    'D4': 'https://github.com/laiguokun/multivariate-time-series-data/raw/refs/heads/master/exchange_rate/exchange_rate.txt.gz',
    'D5_D6': 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip',
    'D7': 'https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip'
}


def main():
    populate_D2_D3_stock()
    populate_D4_exchange()
    populate_D5_D6_energy()
    populate_D7_eeg()


def populate_D2_D3_stock():
    raw_data_path = download_dataset(
        url=urls['D2_D3'],
        data_path_in_zip=None,
        dataset_name='D2_D3_stock'
    )

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D2_stock',
        seq_length=24
    )

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D3_stock_long',
        seq_length=125
    )


def populate_D4_exchange():
    raw_data_path = download_dataset(
        url=urls['D4'],
        data_path_in_zip=None,
        dataset_name='D4_exchange'
    )

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D4_exchange',
        seq_length=125
    )


def populate_D5_D6_energy():
    raw_data_path = download_dataset(
        url=urls['D5_D6'],
        data_path_in_zip='energydata_complete.csv', 
        dataset_name='D5_D6_energy'
    )

    remove_csv_column(raw_data_path, 'date')

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D5_energy', 
        seq_length=24
    )
    

    preprocess_data(
        ori_data_path=raw_data_path,
        dataset_name='D6_energy_long', 
        seq_length=125
    )


def populate_D7_eeg():
    arff_path = download_dataset(
        url=urls['D7'],
        data_path_in_zip='EEG Eye State.arff',
        dataset_name='D7_eeg'
    )

    csv_path = arff2csv(arff_path)

    remove_csv_column(csv_path, 'eyeDetection')

    preprocess_data(
        ori_data_path=csv_path,
        dataset_name='D7_eeg',
        seq_length=128
    )
   

def download_dataset(url, data_path_in_zip, dataset_name):
    dest_path = os.path.join(raw_data_dir, dataset_name)
    if os.path.isfile(dest_path):
        return dest_path
    
    print(f"Downloading {dataset_name} dataset")
    file_path, _ = request.urlretrieve(url, reporthook=urlretrieve_reporthook)
    print("\n")

    download_suffix = pathlib.Path(url).suffix
    if download_suffix != '.csv':
        print(f"Unzipping {dataset_name} dataset")
        file_path = unzip_file(file_path, download_suffix)

    print(f"Moving {dataset_name} dataset to {dest_path}")
    if data_path_in_zip != None:
        file_path = os.path.join(file_path, data_path_in_zip)
    shutil.move(file_path, dest_path)
    return dest_path


def unzip_file(path, suffix):
    out_path = path + '.unzip'
    if suffix == '.zip':
        with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(out_path)
            return out_path
    elif suffix == '.gz':
        with gzip.open(path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                return out_path
    else:
        raise NotImplemented(f"Decompression for {suffix} not implemented")



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


def remove_csv_column(file_path, col_name):
    data = pd.read_csv(file_path)
    if not col_name in data.columns:
        return
    print(f'Removing column {col_name}')
    data = data.drop([col_name], axis='columns')
    data.to_csv(file_path, index=False)


def arff2csv(file_path):
    out_path = re.sub('.arff', '', file_path) + '.csv'
    data = arff.loadarff(file_path)
    data = pd.DataFrame(data[0])
    data.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    main()