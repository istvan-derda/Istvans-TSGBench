import sys
import os

sys.path.insert(0,'../') # make tsgb_utils import work
time_vqvae_path = os.path.join(os.path.dirname(__file__), 'Time_VQVAE')
sys.path.insert(0,time_vqvae_path) # make imports in TimeVQVAE/*.py work

preprocessing_init = os.path.join(time_vqvae_path, 'preprocessing', '__init__.py')
if not os.path.exists(preprocessing_init):
    open(preprocessing_init, 'a').close()

import torch
from torch.utils.data import Dataset, DataLoader
from tsgb_utils import load_train_data, load_valid_data, get_dataset_name, persist_gen_data
import Time_VQVAE.stage1 as stage1
import Time_VQVAE.stage2 as stage2
from Time_VQVAE.utils import load_yaml_param_settings
from Time_VQVAE.generators.sample import unconditional_sample
import wandb
import numpy as np
from Time_VQVAE.experiments.exp_stage2 import ExpStage2


class TSGBenchDataset(Dataset):
    def __init__(self, which):
        if which == "train":
            self.data = np.transpose(load_train_data(), (0, 2, 1)) # self.data used?
            self.X = self.data
            self.Y = np.zeros((self.X.shape[0], 1))
        elif which == "valid":
            self.X = np.transpose(load_valid_data(), (0, 2, 1))            
            self.Y = np.zeros((self.X.shape[0], 1))
        else:
            raise ValueError
    
    def __getitem__(self, index):
        x = self.X[index, :]
        y = self.Y[index, :]
        return x, y
    
    def __len__(self):
        return self.X.shape[0]
    

def main():    
    wandb.init(mode="offline")
    wandb.Settings(silent=True)

    train_data = TSGBenchDataset('train')
    valid_data = TSGBenchDataset('valid')

    # Train Stage 1
    args = stage1.load_args()
    config = load_yaml_param_settings(args.config)

    N_SAMPLES, IN_CHANNELS, INPUT_LENGTH = train_data.X.shape
    config['dataset']['in_channels'] = IN_CHANNELS
    N_CLASSES = len(np.unique(train_data.Y))

    batch_size1 = config['dataset']['batch_sizes']['stage1']
    num_workers = config['dataset']['num_workers']

    train_data_loader1 = DataLoader(train_data, batch_size1, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    test_data_loader1 = DataLoader(valid_data, batch_size1, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)

    DATASET_NAME = get_dataset_name()

    stage1.train_stage1(
        config=config,
        dataset_name=DATASET_NAME, 
        train_data_loader=train_data_loader1,
        test_data_loader=test_data_loader1, 
        gpu_device_ind=args.gpu_device_ind
    )

    # Train Stage 2
    wandb.init(mode="offline")
    wandb.Settings(silent=True)
    
    args = stage2.load_args()
    batch_size2 = config['dataset']['batch_sizes']['stage2']

    train_data_loader2 = DataLoader(train_data, batch_size2, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    test_data_loader2 = DataLoader(valid_data, batch_size2, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)

    stage2.train_stage2(
        config=config,
        dataset_name=DATASET_NAME,
        train_data_loader=train_data_loader2,
        test_data_loader=test_data_loader2,
        gpu_device_ind=args.gpu_device_ind,
        feature_extractor_type="rocket",
        use_custom_dataset=True
    )
        
    
    # SAMPLE
    exp_stage2 = ExpStage2.load_from_checkpoint(
        os.path.join('saved_models', f'stage2-{DATASET_NAME}.ckpt'), 
        dataset_name=DATASET_NAME, 
        in_channels=IN_CHANNELS,
        input_length=INPUT_LENGTH, 
        config=config,
        n_classes=N_CLASSES,
        feature_extractor_type='rocket',
        use_custom_dataset=True,
        map_location='cpu',
        strict=False
        )

    maskgit = exp_stage2.maskgit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _x_new_l, _x_new_h, x_new = unconditional_sample(
        maskgit=maskgit, 
        n_samples=N_SAMPLES, 
        device=device, 
        batch_size=config['evaluation']['batch_size']
    )

    print(x_new.shape)
    x_new = np.transpose(x_new, (0, 2, 1))
    
    persist_gen_data(x_new)



if __name__ == '__main__':
    main()