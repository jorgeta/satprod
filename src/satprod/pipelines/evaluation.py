from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pickle

from satprod.configs.config_utils import TimeInterval
from satprod.configs.job_configs import TrainConfig
from satprod.ml.models import LSTM_net
from satprod.pipelines.dataset import WindDataset
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

@dataclass
class Results():
    params_in_network: int
    epoch: int
    lowest_valid_mae: float
    corr_train_mae: float
    train_mae: [float]
    valid_mae: [float]
    best_val_preds: [float]
    val_targs: [float]
    corr_train_preds: [float]
    train_targs: [float]

def store_results(
    timestamp: str, 
    model, 
    train_config: TrainConfig, 
    results: Results, 
    scaler,
    target_label_indices: [int]):
    
    if len(train_config.parks)==1: park = train_config.parks[0]
    else: park = 'all'
    
    cd = str(os.path.dirname(os.path.abspath(__file__)))
    root = f'{cd}/../../..'
    path = f'{root}/storage/{park}/{model.name}/{timestamp}'
    os.makedirs(path, exist_ok=True)
    
    info = (model, results, train_config, scaler, target_label_indices)
    with open(f'{path}/model_results_config.pickle', 'wb') as storage_file:
        pickle.dump(info, storage_file)

class Evaluate():
    
    def __init__(self, timestamp: str, model_name: str, park: str):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.timestamp = timestamp
        self.model_name = model_name
        self.park = park
        
        self.get_stored_results()
        self.info()
        self.unscale_val_predictions()
        self.create_error_matrix()
    
    def get_stored_results(self):
        self.path = f'{self.root}/storage/{self.park}/{self.model_name}/{self.timestamp}'
        with open(f'{self.path}/model_results_config.pickle', 'rb') as storage_file:
            net, results, train_config, scaler, target_label_indices = pickle.load(storage_file)
            
        self.net = net
        self.results = results
        self.train_config = train_config
        self.scaler = scaler
        self.target_label_indices = target_label_indices
        self.wind_dataset = WindDataset(self.train_config)
    
    def info(self, to_console: bool=False):
        info_str = f'\nTimestamp: {self.timestamp}\nPark: {self.park}\nModel: {self.model_name}'
        info_str += f'\nParams in network: {self.results.params_in_network}'
        info_str += f'\nLowest validation MAE: {self.results.lowest_valid_mae}'
        info_str += f'\nCorresponding train MAE: {self.results.corr_train_mae}'
        info_str += f'\nEpoch of lowest validation MAE: {self.results.epoch}'
        info_str += f'\n{self.net}'
        for key, value in vars(self.net).items():
            if not key.startswith('_') and key!='training':
                info_str += f'\n{key}: {value}'
        for key, value in vars(self.train_config).items():
            info_str += f'\n{key}: {value}'
        
        if to_console: logging.info(info_str)
        
        with open(f'{self.path}/info.txt', 'w') as info_file:
            info_file.write(info_str)
    
    def unscale_val_predictions(self):
        val_preds = np.array(self.results.best_val_preds)
        val_targs = np.array(self.results.val_targs)
        #train_preds = np.array(self.results.corr_train_preds)
        #train_targs = np.array(self.results.train_targs)
        
        means = [self.scaler.mean_[x] for x in self.target_label_indices]
        stds = [self.scaler.scale_[x] for x in self.target_label_indices]
        
        for step in range(self.train_config.pred_sequence_length):
            
            preds = val_preds[:, step, :]
            targs = val_targs[:, step, :]
            
            for park_idx in range(preds.shape[1]):
                preds[:,park_idx] = preds[:,park_idx]*stds[park_idx]+means[park_idx]
                targs[:,park_idx] = targs[:,park_idx]*stds[park_idx]+means[park_idx]
            
            val_preds[:, step, :] = preds
            val_targs[:, step, :] = targs
        
        self.val_targs_unscaled = val_targs
        self.val_preds_unscaled = val_preds
        self.val_prediction_interval = TimeInterval(
            start=self.wind_dataset.valid_start+timedelta(hours=self.net.sequence_len), 
            stop=self.wind_dataset.valid_start+timedelta(hours=self.net.sequence_len+len(val_preds)-1)
        )
        
        #(384, 5, 4)
        
        print(f'Shape of validation predictions: {self.val_preds_unscaled.shape}')
        
    def create_error_matrix(self):
        total_mae = mean_absolute_error(np.ravel(self.val_targs_unscaled), np.ravel(self.val_preds_unscaled))
        
        error_matrix = np.zeros((self.val_targs_unscaled.shape[1], self.val_targs_unscaled.shape[2]))
        for i in range(error_matrix.shape[0]):
            for j in range(error_matrix.shape[1]):
                error_matrix[i][j] = mean_absolute_error(self.val_targs_unscaled[:,i,j], self.val_preds_unscaled[:,i,j])
        
        self.error_matrix = error_matrix
        
    def plot_errors(self):
        fig, axs = plt.subplots(self.error_matrix.shape[1], 1, constrained_layout=True)
        
        if self.error_matrix.shape[1]==1:
            axis = []
            axis.append(axs)
        else:
            axis = axs
        
        fig.suptitle('MAE')
        for i in range(self.error_matrix.shape[1]):
            axis[i].bar(range(self.error_matrix.shape[0]), self.error_matrix.T[i], align='center')
            axis[i].set_title(self.train_config.parks[i])
            axis[i].set_xlabel('hours ahead')
            axis[i].set_ylabel('MAE')
            axis[i].set_xticks(range(self.error_matrix.shape[0]))
            axis[i].set_xticklabels([f'{h+1}' for h in range(self.error_matrix.shape[0])])
        plt.savefig(f'{self.path}/validation_maes.png')
        #plt.show()
        
    
    def compare_to_baselines(self):
        # persistence
        prod = get_columns(self.wind_dataset.data_unscaled, 'production')
        prod = prod[prod.index >= self.wind_dataset.valid_start]
        prod = prod[prod.index < self.wind_dataset.test_start]
        prod_columns = prod.columns
        
        self.persistence_error_matrix = np.zeros_like(self.error_matrix) #(5,4)
        for i in range(self.persistence_error_matrix.shape[0]):
            for j, col in enumerate(prod_columns):
                preds = prod[col].shift(1).loc[self.val_prediction_interval.start:self.val_prediction_interval.stop]
                targs = prod[col].shift(-i).loc[self.val_prediction_interval.start:self.val_prediction_interval.stop]
                self.persistence_error_matrix[i,j] = mean_absolute_error(preds, targs)
        
        # linear
        speed = get_columns(self.wind_dataset.data_unscaled, 'speed')
        wind_speed_forecasts = get_columns(get_columns(self.wind_dataset.data_unscaled,'speed'), '+')
        
        val_wind_forecasts = wind_speed_forecasts.loc[self.val_prediction_interval.start:self.val_prediction_interval.stop].values
        
        prod = get_columns(self.wind_dataset.data_unscaled, 'production')
        speed = speed.drop(columns=wind_speed_forecasts.columns)
        
        def linear_prediction(x, speed, prod):
            if x==0:
                return 0
            # linear model
            # Create linear regression object
            from sklearn import linear_model
            regr = linear_model.LinearRegression()
            
            regression_data = pd.concat([speed, prod], axis=1)
            regression_data = regression_data.dropna(axis=0)
            regression_data = regression_data[regression_data.index < self.wind_dataset.valid_start]
            
            regression_data = regression_data[regression_data['wind_speed_bess'] > 0.9*x].copy()
            regression_data = regression_data[regression_data['wind_speed_bess'] < 1.1*x]
            
            regr.fit(regression_data['wind_speed_bess'].values.reshape(-1, 1), regression_data['production_bess'].values.reshape(-1, 1))
            
            return regr.predict(np.array(x).reshape(-1, 1))[0][0]
        
        linear_val_preds = np.zeros_like(val_wind_forecasts)
        for i in range(linear_val_preds.shape[0]):
            for j in range(linear_val_preds.shape[1]):
                linear_val_preds[i, j] = linear_prediction(val_wind_forecasts[i][j], speed, prod)
        
        linear_val_preds = linear_val_preds.reshape(self.val_targs_unscaled.shape)
        
        self.linear_model_error_matrix = np.zeros((self.val_targs_unscaled.shape[1], self.val_targs_unscaled.shape[2]))
        for i in range(self.linear_model_error_matrix.shape[0]):
            for j in range(self.linear_model_error_matrix.shape[1]):
                self.linear_model_error_matrix[i][j] = mean_absolute_error(self.val_targs_unscaled[:,i,j], linear_val_preds[:,i,j])
        
        print(self.persistence_error_matrix)
        print(self.linear_model_error_matrix)
        print(self.error_matrix)
        
        labels = ['1', '2', '3', '4', '5']
        per = np.ravel(self.persistence_error_matrix)
        lin = np.ravel(self.linear_model_error_matrix)
        lstm = np.ravel(self.error_matrix)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, per, width, label='Persistence')
        rects2 = ax.bar(x, lin, width, label='Linear')
        rects3 = ax.bar(x + width/2, lstm, width, label='LSTM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE')
        ax.set_xlabel('Hours ahead')
        ax.set_title('MAE of three models')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.savefig(f'{self.path}/error_comparison.png')
        plt.show()
        
        
        '''
        plt.plot(np.ravel(self.results.train_targs), label='train_targs')
        plt.plot(np.ravel(train_preds), label='train_preds')
        plt.legend()
        plt.savefig(f'storage/train.png')
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()
        
        plt.plot(np.ravel(val_targs), label='val_targs')
        plt.plot(np.ravel(val_preds), label='val_preds')
        plt.legend()
        plt.savefig(f'storage/val_{now}.png')
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()
        
        epochs = np.arange(len(train_mae))
        plt.figure()
        plt.plot(epochs, train_mae, 'r', epochs, valid_mae, 'b')
        plt.legend(['train MAE','validation MAE'])
        plt.xlabel('epochs')
        plt.ylabel('MAE')
        plt.savefig(f'storage/maes_{now}.png')
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()
        
        '''