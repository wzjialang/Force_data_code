import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import os
import pickle
from utils import *


class force_data(Dataset):
    def __init__(self, data_dir, splits, fold, is_train=None):

        self.data_dir = data_dir
        self.is_train = is_train

        with open(splits, 'rb') as file:
            data = pickle.load(file)

        # All folds
        fold_ids = ['1','2','3','4','5','6']
        # Remove validation fold
        fold_ids.remove(fold)

        if self.is_train:
            self.attempts_set = []
            for fold_id in fold_ids:
                fold_list = data['fold_' + fold_id]
                for attempt in fold_list:
                    self.attempts_set.append(attempt)
        else:
            self.attempts_set = data['fold_' + fold]
            
        self.scaler = normalize_data(data_dir) # scaler for force data

    def __len__(self):
        return len(self.attempts_set)
    
    def __getitem__(self, idx):

        # Pick an attempt
        attempt = self.attempts_set[idx]

        ############# Force Data ##############
        attempt_force_data_path = os.path.join(self.data_dir,attempt)
        data = pd.read_csv(attempt_force_data_path, header=None)
        force_data = data.iloc[:, 1] # take only the second column which contains the force data
        force_data = self.scaler.transform(force_data.values.reshape(-1, 1))
        force_data = torch.tensor(np.asarray(force_data))
        force_data = torch.permute(force_data,(1,0)) # features,T

        # For training if sequence is < 300 we pad with zeros. If > 300 we pick randomly 300 consecutive samples
        if self.is_train:
            if force_data.shape[1] < 300:
                force_data = torch.cat((force_data,torch.zeros(force_data.shape[0],300-force_data.shape[1])),1)
            else:
                index = np.random.randint(0,force_data.shape[1] - 300)
                force_data = force_data[:,index:index+300]
        
        
        ############### Labels ################
        label = 1 if 'expert' in attempt else 0 # 1 for expert, 0 for novice
        label = torch.tensor(label).unsqueeze(dim=0)

        return force_data, label


# Dataset
def dataset(scheme,fold,batch_size_train,batch_size_val):
    # Paths for videos and annotations (besed on the cross-validation scheme)
    
    scheme = scheme
    fold = fold
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'all_attempts')
    splits = os.path.join(cwd,scheme + '.pkl')

    # Data Splits
    train_set = force_data(data_dir, splits, fold, is_train=True)
    val_set = force_data(data_dir, splits, fold, is_train=False)

    # Dataloaders
    train_loader = DataLoader(dataset=train_set,batch_size=batch_size_train,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(dataset=val_set,batch_size=batch_size_val,shuffle=False,num_workers=4,pin_memory=True)

    return train_loader, val_loader



if __name__ == '__main__':

    # Test that everything is working fine
    train, val = dataset(scheme='louo',fold='1',batch_size_train=batch_size_train,batch_size_val=batch_size_val)

    inputs, labels = next(iter(train))

    print(inputs.shape)
    print(labels.shape)