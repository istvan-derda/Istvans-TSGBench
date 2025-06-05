import sys
import os

sys.path.append('../') # make tsgb_utils import work
sys.path.append(os.path.join(os.path.dirname(__file__), 'Time_GAN')) # make importing utils in timegan.py work

from tsgb_utils import load_train_data, persist_gen_data
from Time_GAN.timegan import timegan

def main():
    train_data = load_train_data()

    # taken from TimeGAN README.md line 66
    parameters = { 
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': 50000,
        'batch_size': 128,
        'module': 'gru'
    }

    sys.path.append('')
    gen_data = timegan(train_data, parameters)

    persist_gen_data(gen_data)


if __name__ == '__main__':
    main()