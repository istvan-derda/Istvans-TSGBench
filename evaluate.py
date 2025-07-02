from src.evaluation import evaluate_data
import mgzip
import pickle
import argparse
from pyprojroot import here

all_datasets = [2,3,4,5,6,7]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--dataset_no', default=None)

    args = parser.parse_args()

    if args.dataset_no != None:
        dataset_nos = [args.dataset_no]
    else:
        dataset_nos = all_datasets

    for dataset_no in dataset_nos:
        ori_data = ori_load(dataset_no)
        gen_data, gen_data_name = gen_load(args.model, dataset_no)
        evaluate_data(
            ori_data,
            gen_data,
            model_name=args.model,
            dataset_name=gen_data_name
        )


def ori_load(dataset_no):
    paths = list(here('data/ori').glob(f'D{dataset_no}*/D{dataset_no}*train.pkl'))
    assert len(paths) == 1
    path = paths[0]
    with mgzip.open(str(path), 'rb') as f:
        train_data = pickle.load(f)

    paths = list(here('data/ori').glob(f'D{dataset_no}*/D{dataset_no}*valid.pkl'))
    assert len(paths) == 1
    path = paths[0]
    with mgzip.open(str(path), 'rb') as f:
        valid_data = pickle.load(f)
    return train_data, valid_data


def gen_load(model_name, dataset_no):
    paths = list(here('models').glob(f'{model_name}*/gen/D{dataset_no}*/D{dataset_no}*.pkl'))
    assert len(paths) == 1
    path = paths[0]
    with mgzip.open(str(path)) as f:
        data = pickle.load(f)
    gen_filename = path.name
    return data, gen_filename


if __name__ == '__main__':
    main()