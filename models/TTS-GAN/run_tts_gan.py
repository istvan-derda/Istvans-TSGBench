import sys
import os
import numpy as np
import torch

sys.path.append('../') # make tsgb_utils import work
sys.path.append(os.path.join(os.path.dirname(__file__), 'TTS_GAN')) # make imports in TTS_GAN/train_GAN.py work

from tsgb_utils import load_train_data, persist_gen_data
from TTS_GAN.train_GAN import train_tts_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load Data
    train_data = load_train_data()

    # Reshape shape from (BH, length, channel) to (BH, channel, 1, length)
    train_data = np.transpose(train_data, (0, 2, 1))
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1, train_data.shape[2])

    # Train Model
    model = train_tts_gan(train_data)

    # Sample Synthetic Timeseries
    sample_count, _, _, seq_len = train_data.shape
    gen_data = generate(model, sample_count, seq_len)

    # Persist Generated Timeseries
    persist_gen_data(gen_data)


def generate(model, sample_count, seq_len):
    synthetic_data = []
    for _ in range(sample_count):
        random_noise = torch.tensor(np.random.normal(0, 1, (1, seq_len))).to(device, dtype=float32)
        sample = model(random_noise).to('cpu').detach().numpy()
        synthetic_data.append(sample)
    return synthetic_data

if __name__ == '__main__':
    main()
