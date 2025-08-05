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
    ori_shape = train_data.shape

    # Reshape shape from (BH, length, channel) to (BH, channel, 1, length)
    train_data = np.transpose(train_data, (0, 2, 1))
    train_data = train_data.reshape(ori_shape[0], ori_shape[2], 1, ori_shape[1])

    # Train Model
    seq_len = train_data.shape[3]
    if seq_len == 24:
        patch_size = 12
    elif seq_len == 125:
        patch_size = 25
    else:
        raise NotImplementedError(f"No patch_size implemented for timeseries sequence length {seq_len}. train_data.shape: {train_data.shape}")
        
    model = train_tts_gan(train_data, patch_size)

    # Sample Synthetic Timeseries
    sample_count = train_data.shape[0]
    gen_data = generate(model, sample_count, seq_len)
    gen_data = gen_data.reshape(ori_shape[0], ori_shape[2], ori_shape[1])
    gen_data = np.transpose(gen_data, (0, 2, 1))

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
