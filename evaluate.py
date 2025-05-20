from src.evaluation import evaluate_data
import mgzip
import pickle


def main():
    ori_data = ori_load()
    gen_data = gen_load()
    evaluate_data(
        ori_data,
        gen_data,
        model_name='TimeTransformer',
        dataset_name='stock'
    )


def ori_load():
    with mgzip.open('data/ori/D2_stock/D2_stock_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with mgzip.open('data/ori/D2_stock/D2_stock_valid.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    return train_data, valid_data


def gen_load():
    with mgzip.open('data/gen/D2_stock/D2_stock_gen.pkl') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    main()