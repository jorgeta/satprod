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
from satprod.pipelines.dataset import WindDataset
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

@dataclass
class Results():
    params_in_network: int
    trainable_params_in_network: int
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
        
        self.park_name = {
            'bess': 'Bessakerfjellet', 
            'skom': 'Skomakerfjellet', 
            'vals': 'Valsneset', 
            'yvik': 'Ytre Vikna'
        }
        
        self.get_stored_results()
        self.info()
        self.unscale_val_predictions()
        self.create_error_matrix()
        self.baseline_comparisons()
        self.plot_fitting_example()
        self.plot_training_curve()
    
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
        info_str = f'\nTrainable parameters in network: {self.results.trainable_params_in_network}'
        info_str += f'\nParameters in network: {self.results.params_in_network}'
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
        
        #(384, 5, 4)
        
        logging.info(f'Shape of validation predictions: {self.val_preds_unscaled.shape}')
        
    def create_error_matrix(self):
        self.total_mae = mean_absolute_error(np.ravel(self.val_targs_unscaled), np.ravel(self.val_preds_unscaled))
        
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
        plt.close()
        #plt.show()
    
    def persistence(self):
        
        # get relevant production
        prod = get_columns(self.wind_dataset.data_unscaled, 'production')
        prod_columns = prod.columns
        
        self.persistence_error_matrix = np.zeros_like(self.error_matrix) #(5,4)
        
        for i in range(self.train_config.pred_sequence_length):
            for col in prod_columns:
                prod[f'{col}_shift({i+1})'] = prod[col].shift(i+1).values
        prod = prod.loc[
            self.train_config.valid_start:self.train_config.test_start-timedelta(hours=1)
        ].dropna(axis=0)
        
        for i in range(self.train_config.pred_sequence_length):
            for j, col in enumerate(prod_columns):
                self.persistence_error_matrix[i,j] = mean_absolute_error(
                    prod[col].values, prod[f'{col}_shift({i+1})'].values)
        
        self.persistence_total_mae = np.mean(np.ravel(self.persistence_error_matrix))
    
    '''def avg_of_history_model(self):
        upper_percent_standard = 1.1
        lower_percent_standard = 0.9
        
        upper_percent = upper_percent_standard
        lower_percent = lower_percent_standard
        
        # get wind_speeds and wind speed forecasts
        speed = get_columns(self.wind_dataset.data_unscaled, 'speed')
        #print(speed)
        wind_speed_forecasts = get_columns(speed, '+')
        if wind_speed_forecasts.empty:
            self.avg_of_history_error_matrix = None
        else:
            val_wind_forecasts = wind_speed_forecasts.loc[self.train_config.valid_start:self.train_config.test_start-timedelta(hours=1)].values
            
            while val_wind_forecasts.ndim < 3:
                val_wind_forecasts = val_wind_forecasts[..., np.newaxis]
            speed = speed.drop(columns=wind_speed_forecasts.columns)
            prod = get_columns(self.wind_dataset.data_unscaled, 'production')
            
            speed_prod = pd.concat([speed, prod], axis=1).dropna(axis=0)
            speed_prod = speed_prod[speed_prod.index < self.wind_dataset.valid_start]
            
            preds = np.zeros_like(val_wind_forecasts)
            
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    for k in range(preds.shape[2]):
                        wind_forecast = val_wind_forecasts[i][j][k]
                        park = self.train_config.parks[k]
                        
                        similar_situation_production_avg = []
                        while len(similar_situation_production_avg)==0:
                            if park=='skom':
                                similar_situation_production_avg = speed_prod[
                                    (speed_prod[f'wind_speed_bess'] > lower_percent*wind_forecast) & (speed_prod[f'wind_speed_bess'] < upper_percent*wind_forecast)
                                    ][f'production_skom'].values
                            else:
                                similar_situation_production_avg = speed_prod[
                                    (speed_prod[f'wind_speed_{park}'] > lower_percent*wind_forecast) & (speed_prod[f'wind_speed_{park}'] < upper_percent*wind_forecast)
                                    ][f'production_{park}'].values
                            lower_percent -= 0.1
                            upper_percent += 0.1
                        
                        lower_percent = lower_percent_standard
                        upper_percent = upper_percent_standard
                        preds[i, j, k] = np.mean(similar_situation_production_avg)
            
            self.avg_of_history_error_matrix = np.zeros((self.val_targs_unscaled.shape[1], self.val_targs_unscaled.shape[2]))
            for i in range(self.avg_of_history_error_matrix.shape[0]):
                for j in range(self.avg_of_history_error_matrix.shape[1]):
                    self.avg_of_history_error_matrix[i][j] = mean_absolute_error(self.val_targs_unscaled[:,i,j], preds[:,i,j])'''
    
    def baseline_comparisons(self):
        self.persistence()
        self.avg_of_history_error_matrix = None
        
        logging.info(f'Persistence MAEs:\n{self.persistence_error_matrix}')
        if self.avg_of_history_error_matrix is not None:
            logging.info(f'Average history based model MAEs:\n{self.avg_of_history_error_matrix}')
        logging.info(f'{self.model_name} MAEs:\n{self.error_matrix}')
        
        labels = ['1', '2', '3', '4', '5']
        per = np.ravel(self.persistence_error_matrix)
        if self.avg_of_history_error_matrix is not None:
            lin = np.ravel(self.avg_of_history_error_matrix)
        else: lin = None
        lstm = np.ravel(self.error_matrix)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, per, width, label='Persistence')
        if lin is not None:
            rects2 = ax.bar(x, lin, width, label='Linear')
        rects3 = ax.bar(x + width/2, lstm, width, label=f'{self.model_name}')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE')
        ax.set_xlabel('Hours ahead')
        ax.set_title('MAE of three models')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        fig.tight_layout()
        plt.savefig(f'{self.path}/error_comparison.png')
        plt.close()
    
    def plot_fitting_example(self):
        
        plt.figure()
        plt.ylabel('production (kWh/h)')
        plt.xlabel('hours')
        index = 1
        n_samples_shown = np.minimum(50, self.val_targs_unscaled.shape[0])
        n_plots = self.train_config.pred_sequence_length*len(self.train_config.parks)
        for i in range(self.train_config.pred_sequence_length):
            for j in range(len(self.train_config.parks)):
                plt.subplot(self.train_config.pred_sequence_length, len(self.train_config.parks), index) #(nrows, ncols, index)
                plt.plot(np.ravel(np.array(self.val_targs_unscaled)[:n_samples_shown, i, j]), label='targets')
                plt.plot(np.ravel(np.array(self.val_preds_unscaled)[:n_samples_shown, i, j]), label='predicitions')
                
                
                plt.title(f'{i+1}h ahead at {self.park_name[self.train_config.parks[j]]}', fontsize=8)
                if index==n_plots:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                else:
                    plt.xticks([])
                plt.tight_layout()
                index += 1
        plt.tight_layout()
        plt.savefig(f'{self.path}/fitting_examples.png')
        plt.close()
        
    def plot_training_curve(self):
        
        epochs = np.arange(len(self.results.train_mae))
        plt.plot(epochs, self.results.train_mae, 'r', epochs, self.results.valid_mae, 'b')
        plt.legend(['train MAE','validation MAE'])
        plt.xlabel('epochs')
        plt.ylabel('MAE')
        plt.axvline(x=self.results.epoch)
        
        plt.savefig(f'{self.path}/training_curve.png')
        plt.close()