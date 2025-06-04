import mgzip
import pickle
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, '../data/')


def load_train_data():   
    ori_data_dir = os.path.join(data_dir, 'ori/')
    dataset_name = get_dataset_name()
    with mgzip.open(f'{ori_data_dir}{dataset_name}/{dataset_name}_train.pkl') as f:
        train_data = pickle.load(f)
    return train_data


def persist_gen_data(data):
    gen_data_dir = os.path.join(data_dir, 'gen/')
    dataset_name = get_dataset_name()
    dir_path = f'{gen_data_dir}{dataset_name}/'
    file_path = dir_path+f'{dataset_name}_gen.pkl'
    os.makedirs(dir_path, exist_ok=True)
    with mgzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)


def get_dataset_name():
    return os.environ['TSGB_USE_DATASET']