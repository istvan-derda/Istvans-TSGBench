import sys
import os
import numpy as np

sys.path.append('../') # make tsgb_utils import work
sys.path.append(os.path.join(os.path.dirname(__file__), 'TransFusion')) # make imports in TransFusion/train.py work

from tsgb_utils import load_train_data, persist_gen_data
from TransFusion.train import train, generate


def main():
    # Load Data
    train_data = load_train_data().astype(np.float32)
    seq_len = train_data.shape[1]
    seq_count = train_data.shape[0]

    # Train Model
    model = train(train_data=train_data, seq_len=seq_len)

    # Generate Synthetic Data
    gen = generate(model=model, len=seq_count)

    # Persist Synthetic Data
    persist_gen_data(gen)


if __name__ == '__main__':
    main()
