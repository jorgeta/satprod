from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from satprod.data_handlers.num_data import NumericalDataHandler
from satprod.configs.config_utils import ImgType
from satprod.data_handlers.img_data import Img, ImgDataset
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

class WindDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_end: datetime=None, test_start: datetime=None):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.formatted_data_path = os.path.join(self.root, 'data', 'formatted')
        
        self.train_end = train_end
        self.test_start = test_start
        
        if self.train_end is None: self.train_end = datetime(2019, 4, 30, 23)
        if self.test_start is None: self.test_start = datetime(2020, 5, 1, 0)
        
        self.img_features = []#['grid']
        self.target_labels = ['production_bess', 'production_skom', 'production_vals', 'production_yvik']
        
        num = NumericalDataHandler()
        
        num_data = num.read_formatted_data(nan=False)
        
        try:
            # get img data from file
            img_data = pd.read_csv(f'{self.formatted_data_path}/img_data.csv', index_col='time')
            img_data.index = pd.to_datetime(img_data.index)
        except:
            # create img data dataset, write to file, and return set
            img_data = self.update_image_indices(num_data)
        
        self.img_datasets = {}
        for col in img_data.columns:
            if col not in self.img_features:
                img_data = img_data.drop(columns=[col])
            else:
                self.img_datasets[col] = ImgDataset(ImgType(col), normalize=True)
        
        # must be called in init
        self.__split_and_scale_data(num_data, img_data)
        
    def update_image_indices(self, num_data):
        
        # all possible image features
        img_features = ['grid', 'sat', 'fb_dense', 'lk_dense', 'dtvl1_dense', 'rlof_dense', 'lk_sparse', 'lk_sparsemask']
        img_data = pd.DataFrame(index=num_data.index, columns=img_features)
        
        img_datasets = {}
        for col in img_data.columns:
            try:
                img_datasets[col] = ImgDataset(ImgType(col), normalize=True)
        
                img_idx_list = []
                for index in img_data.index:
                    img_idx_list.append(
                        img_datasets[col].getDateIdx(
                            datetime.strptime(str(index), '%Y-%m-%d %H:%M:%S')
                        )
                    )
                
                img_data[col] = img_idx_list
            except:
                img_data = img_data.drop(columns=[col])
        
        img_data.to_csv(f'{self.formatted_data_path}/img_data.csv')
        return img_data
        
    def __split_and_scale_data(self, num_data, img_data):
        # num_data_train_scaled, num_data_valid_scaled, test sets
        num_data_train_scaled = num_data[num_data.index <= self.train_end]
        #self.img_data_train = self.img_data[self.img_data.index <= self.train_end]
        
        self.train_end_index = len(num_data_train_scaled)-1
        
        num_data_valid_scaled = num_data[num_data.index > self.train_end]
        #img_data_valid = self.img_data[self.img_data.index > self.train_end]
        num_data_valid_scaled = num_data_valid_scaled[num_data_valid_scaled.index < self.test_start]
        #self.img_data_valid = img_data_valid[img_data_valid.index < self.test_start]
        #test = num_data[num_data.index >= test_start]
        
        self.test_start_index = self.train_end_index+1 + len(num_data_valid_scaled)
        
        # find scaling factors from num_data_train_scaled set, and scale num_data_valid_scaled and test set accordingly
        self.scaler = StandardScaler()
        
        train_without_direction = pd.concat([get_columns(num_data_train_scaled, 'speed'), get_columns(num_data_train_scaled, 'production')], axis=1)
        valid_without_direction = pd.concat([get_columns(num_data_valid_scaled, 'speed'), get_columns(num_data_valid_scaled, 'production')], axis=1)
        #test_without_direction = pd.concat([get_columns(test, 'speed'), get_columns(test, 'production')], axis=1)
        
        self.scaler.fit(train_without_direction.values)
        train_sc = self.scaler.transform(train_without_direction.values)
        valid_sc = self.scaler.transform(valid_without_direction.values)
        #test_sc = self.scaler.transform(test_without_direction.values)
        
        num_data_train_scaled[train_without_direction.columns] = train_sc
        num_data_valid_scaled[valid_without_direction.columns] = valid_sc
        #test[test_without_direction.columns] = test_sc
        
        num_data_scaled = pd.concat([num_data_train_scaled, num_data_valid_scaled], axis=0)
        #num_data_scaled = pd.concat([num_data_train_scaled, num_data_valid_scaled, test], axis=0)
        
        self.data = pd.concat([num_data_scaled, img_data], axis=1)
        
    def __getitem__(self, idx: int):
        return self.data.iloc[idx]

    def __len__(self):
        return len(self.data)
        
if __name__ =='__main__':
    wind_dataset = WindDataset()
    
    i = 0
    for img_feature in wind_dataset.img_features:
        idx = int(wind_dataset[i][img_feature])
        img_dataset = wind_dataset.img_datasets[img_feature]
        #print(img_handler[idx].img)
    
    sequence_length = 5
    pred_sequence_length = 2
    
    train_indices = np.arange(sequence_length, wind_dataset.train_end_index+1-pred_sequence_length)
    print(train_indices)
    np.random.seed(10)
    #np.random.shuffle(train_indices)
    #np.random.shuffle(train_indices)
    print(train_indices)
    i = 10000
    input_data = wind_dataset[train_indices[i]-sequence_length:train_indices[i]]
    target_data = wind_dataset[train_indices[i]:train_indices[i]+pred_sequence_length][wind_dataset.target_labels]
    print(input_data)
    print(target_data)
    print(len(input_data.dropna(axis=0)))
    
    
    
    