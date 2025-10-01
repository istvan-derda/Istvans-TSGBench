from torch.utils.data import Dataset, DataLoader
from tsgb_utils import load_train_data, load_valid_data, get_dataset_name
from TimeVQVAE.stage1 import load_args, train_stage1
from TimeVQVAE.stage2 import train_stage2
from TimeVQVAE.utils import load_yaml_param_settings

class TSGBenchDataset(Dataset):
    def __init__(self, which):
        if which == "train":
            self.data = load_train_data()
        elif which == "test":
            self.data = load_valid_data()
        else:
            raise ValueError
    
    def __getitem__(self, index):
        X = self.data[index]
        Y = 0
        return X, Y
    
    def __len__(self):
        return self.data.shape[0]
    

def main():    
    args = load_args()
    config = load_yaml_param_settings(args.config)


    train_data = TSGBenchDataset('train')
    valid_data = TSGBenchDataset('valid')

    config['dataset']['in_channels'] = train_data.shape[2]

    batch_size1 = config['dataset']['batch_sizes']['stage1']
    num_workers = config['dataset']['num_workers']

    train_data_loader1 = DataLoader(train_data, batch_size1, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    test_data_loader1 = DataLoader(valid_data, batch_size1, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)

    train_stage1(
        config=config,
        dataset_name=get_dataset_name(), 
        train_data_loader=train_data_loader1,
        test_data_loader=test_data_loader1, 
        gpu_device_ind=args.gpu_device_ind
    )
    
    batch_size2 = config['dataset']['batch_sizes']['stage2']

    train_data_loader2 = DataLoader(train_data, batch_size2, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    test_data_loader2 = DataLoader(valid_data, batch_size2, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)

    train_stage2(
        config=config,
        dataset_name=get_dataset_name(),
        train_data_loader=train_data_loader2,
        test_data_loader=test_data_loader2,
        gpu_device_ind=args.gpu_device_ind,
        feature_extractor_type=args.feature_extractor_type,
        use_custom_dataset=True
    )