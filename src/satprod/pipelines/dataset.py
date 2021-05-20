from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from satprod.data_handlers.num_data import NumericalDataHandler
from satprod.configs.config_utils import ImgType
from satprod.configs.job_configs import TrainConfig
from satprod.data_handlers.img_data import Img, ImgDataset
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

class WindDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_config: TrainConfig):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        
        self.img_extraction_method = train_config.img_extraction_method
        self.parks=train_config.parks
        self.num_feature_types=train_config.num_feature_types
        self.img_features=train_config.img_features
        self.valid_start=train_config.valid_start
        self.test_start=train_config.test_start
        
        # get numerical data
        num = NumericalDataHandler()
        self.num_data = num.read_formatted_data()
        self.formatted_data_path = num.formatted_data_path
        
        # select wanted features
        park_data = []
        for park in self.parks:
            if park=='skom' and 'bess' not in self.parks:
                park_data.append(get_columns(self.num_data, 'bess'))
                park_data[-1] = park_data[-1].drop(columns='production_bess')
                park_data.append(get_columns(self.num_data, park))
            else:
                park_data.append(get_columns(self.num_data, park))
        self.num_data = pd.concat(park_data, axis=1)
        for feature_type in ['speed', 'direction']:
            if feature_type not in self.num_feature_types:
                self.num_data = self.num_data.drop(columns=get_columns(self.num_data, feature_type).columns.values)
        if 'forecast' not in self.num_feature_types:
            self.num_data = self.num_data.drop(columns=get_columns(self.num_data,'+'))
        else:
            # remove forecasts that go beyond the prediction sequence length
            forecasts_removed = False
            hour = train_config.pred_sequence_length+1
            while not forecasts_removed:
                identifier = '+' + str(hour)
                columns_to_drop = get_columns(self.num_data,identifier).columns
                if len(columns_to_drop)==0:
                    forecasts_removed = True
                else:
                    self.num_data = self.num_data.drop(columns=columns_to_drop)
                    hour += 1
                
        
        # target labels
        self.target_labels = list(get_columns(self.num_data,'production').columns)
        
        # image data
        try:
            # get img data from file
            img_data = pd.read_csv(f'{self.formatted_data_path}/img_data.csv', index_col='time')
            img_data.index = pd.to_datetime(img_data.index)
        except:
            # create img data dataset, write to file, and return set
            img_data = self.update_image_indices(self.num_data)
        
        # image datasets
        self.img_datasets = {}
        for col in img_data.columns:
            if col not in self.img_features:
                img_data = img_data.drop(columns=[col])
            else:
                if self.img_extraction_method=='resnet':
                    self.img_datasets[col] = ImgDataset(ImgType(col), normalize=True, upscale=True) #[62, 12, 224, 224, 3]
                else:
                    self.img_datasets[col] = ImgDataset(ImgType(col), normalize=True, grayscale=True)
        
        if len(self.img_features)>0:
            self.img_height = self.img_datasets[self.img_features[0]][0].height
            self.img_width = self.img_datasets[self.img_features[0]][0].width
        else:
            self.img_height = 0
            self.img_width = 0
        
        # save unscaled data
        self.data_unscaled = pd.concat([self.num_data, img_data], axis=1)
        
        # must be called in init
        self.__split_and_scale_data(self.num_data, img_data)
        
        # some useful measurements
        self.n_image_features = len(img_data.columns)
        self.n_forecast_features = len(get_columns(self.data, '+').columns)
        self.n_past_features = len(self.data.columns)-self.n_forecast_features-self.n_image_features
        self.n_output_features = len(self.target_labels)
        
        
    def update_image_indices(self):
        
        # all possible image features
        img_features = [
            'grid', 
            'sat', 
            'fb_dense', 
            'lk_dense', 
            'dtvl1_dense', 
            'rlof_dense', 
            'lk_sparse', 
            'lk_sparsemask'
        ]
        img_data = pd.DataFrame(index=self.num_data.index, columns=img_features)
        
        img_datasets = {}
        for col in img_data.columns:
            logging.info(f'Updating image indices in column: {col}')
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
        
        num_data_train_scaled = num_data[num_data.index < self.valid_start]
        #self.img_data_train = self.img_data[self.img_data.index < self.valid_start]
        
        self.valid_start_index = len(num_data_train_scaled)
        
        num_data_valid_scaled = num_data[num_data.index >= self.valid_start]
        #img_data_valid = self.img_data[self.img_data.index > self.valid_start]
        num_data_valid_scaled = num_data_valid_scaled[num_data_valid_scaled.index < self.test_start]
        #self.img_data_valid = img_data_valid[img_data_valid.index < self.test_start]
        #test = num_data[num_data.index >= test_start]
        
        self.test_start_index = self.valid_start_index + len(num_data_valid_scaled)
        
        # find scaling factors from num_data_train_scaled set, and scale num_data_valid_scaled and test set accordingly
        self.scaler = StandardScaler()
        
        train_without_direction = pd.concat([
            get_columns(num_data_train_scaled, 'speed'), 
            get_columns(num_data_train_scaled, 'production'), 
            ], axis=1)
        valid_without_direction = pd.concat([
            get_columns(num_data_valid_scaled, 'speed'), 
            get_columns(num_data_valid_scaled, 'production'),
            ], axis=1)
        #test_without_direction = pd.concat([get_columns(test, 'speed'), get_columns(test, 'production')], axis=1)
        
        # find indices of columns of target values
        self.target_label_indices = []
        for target_label in self.target_labels:
            self.target_label_indices.append(list(train_without_direction.columns).index(target_label))
        
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
        self.data = self.data.dropna(axis=0)
        self.data = self.data.asfreq('H')
        
    def __getitem__(self, idx: int):
        return self.data.iloc[idx]

    def __len__(self):
        return len(self.data)
        
if __name__ =='__main__':
    parks = ['bess']#,'vals','skom','yvik']
    num_feature_types = ['production']
    img_features = ['grid']
    
    train_config = TrainConfig(
        batch_size = 64,
        num_epochs = 15,
        learning_rate = 4e-3,
        scheduler_step_size = 5,
        scheduler_gamma = 0.8,
        train_valid_splits = 1,
        pred_sequence_length = 5,
        random_seed = 0,
        parks = parks,
        num_feature_types = num_feature_types,
        img_features = img_features
    )
    wind_dataset = WindDataset(train_config)
    
    print(wind_dataset.data.dropna(axis=0))
    '''i = 0
    for img_feature in wind_dataset.img_features:
        idx = int(wind_dataset[i][img_feature])
        img_dataset = wind_dataset.img_datasets[img_feature]
        #print(img_handler[idx].img)
    
    sequence_length = 5
    pred_sequence_length = 2
    
    train_indices = np.arange(sequence_length, wind_dataset.valid_start_index+1-pred_sequence_length)
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
    print(len(input_data.dropna(axis=0)))'''
    
    
    
    