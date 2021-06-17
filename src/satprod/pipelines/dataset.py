from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from satprod.data_handlers.num_data import NumericalDataHandler
from satprod.configs.config_utils import ImgType
from satprod.configs.job_configs import DataConfig
from satprod.data_handlers.img_data import Img, ImgDataset
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

class WindDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_config: DataConfig):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        
        self.img_extraction_method = data_config.img_extraction_method
        self.parks = data_config.parks
        self.numerical_features = data_config.numerical_features
        self.use_numerical_forecasts = data_config.use_numerical_forecasts
        self.use_img_forecasts = data_config.use_img_forecasts
        self.use_img_features = data_config.use_img_features
        self.img_features = data_config.img_features
        self.valid_start = data_config.valid_start
        self.test_start = data_config.test_start
        self.test_end = data_config.test_end
        
        self.image_indices_recently_updated = False
        
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
        for feature_type in ['speed', 'direction', 'temporal']:
            if feature_type not in self.numerical_features:
                self.num_data = self.num_data.drop(columns=get_columns(self.num_data, feature_type).columns.values)
        if not self.use_numerical_forecasts:
            self.num_data = self.num_data.drop(columns=get_columns(self.num_data,'+'))
        else:
            # remove forecasts that go beyond the prediction sequence length
            forecasts_removed = False
            hour = data_config.pred_sequence_length+1
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
            self.update_image_indices(self.num_data)
            self.image_indices_recently_updated = True
            img_data = pd.read_csv(f'{self.formatted_data_path}/img_data.csv', index_col='time')
        
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
        
        if self.use_img_forecasts:
            for col in img_data.columns:
                for i in range(data_config.pred_sequence_length):
                    img_data[f'{col}+{i+1}h'] = img_data[col].shift(-i-1)
        
        if self.use_img_features>0:
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
        self.n_image_features = len(self.img_features)
        self.n_forecast_features = len(get_columns(self.num_data, '+').columns)
        self.n_unique_forecast_features = len(get_columns(self.num_data, '+1h').columns)
        self.n_past_features = len(self.num_data.columns)-self.n_forecast_features
        self.n_output_features = len(self.target_labels)
        
        '''print(self.data.columns)
        print(self.n_image_features)
        print(self.n_forecast_features)
        print(self.n_unique_forecast_features)
        print(self.n_past_features)
        print(self.n_output_features)'''
        
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
        
    def __split_and_scale_data(self, num_data, img_data):
        # num_data_train_scaled, num_data_valid_scaled, test sets
        
        # drop irrelevant data before production data is available
        num_data = num_data.dropna(axis=0).asfreq('H')
        img_data = img_data.dropna(axis=0).asfreq('H')
        
        num_data_train_scaled = num_data[num_data.index < self.valid_start]
        
        num_data_valid_scaled = num_data[num_data.index >= self.valid_start]
        num_data_valid_scaled = num_data_valid_scaled[num_data_valid_scaled.index < self.test_start]
        
        num_data_test_scaled = num_data[num_data.index >= self.test_start]
        num_data_test_scaled = num_data_test_scaled[num_data_test_scaled.index <= self.test_end]
        
        self.valid_start_index = len(num_data_train_scaled)
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
        test_without_direction = pd.concat([
            get_columns(num_data_test_scaled, 'speed'), 
            get_columns(num_data_test_scaled, 'production'),
            ], axis=1)
        
        # find indices of columns of target values
        self.target_label_indices = []
        for target_label in self.target_labels:
            self.target_label_indices.append(list(train_without_direction.columns).index(target_label))
        
        self.scaler.fit(train_without_direction.values)
        train_sc = self.scaler.transform(train_without_direction.values)
        valid_sc = self.scaler.transform(valid_without_direction.values)
        test_sc = self.scaler.transform(test_without_direction.values)
        
        num_data_train_scaled[train_without_direction.columns] = train_sc
        num_data_valid_scaled[valid_without_direction.columns] = valid_sc
        num_data_test_scaled[test_without_direction.columns] = test_sc
        
        num_data_scaled = pd.concat([num_data_train_scaled, num_data_valid_scaled, num_data_test_scaled], axis=0)
        
        self.data = pd.concat([num_data_scaled, img_data], axis=1)
        self.data = self.data.dropna(axis=0).asfreq('H')
    
    def __getitem__(self, idx: int):
        return self.data.iloc[idx]

    def __len__(self):
        return len(self.data)