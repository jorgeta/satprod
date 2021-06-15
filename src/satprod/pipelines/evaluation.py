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
from satprod.configs.job_configs import TrainConfig, DataConfig
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
    test_mae: float
    test_preds: [float]
    test_targs: [float]

def store_results(
    model, 
    train_config: TrainConfig, 
    data_config: DataConfig, 
    results: Results, 
    scaler,
    target_label_indices: [int],
    use_img_features: bool,
    train_on_one_batch: bool,
    ):
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    if len(data_config.parks)==1: park = data_config.parks[0]
    else: park = 'all'
    
    if train_on_one_batch:
        sorting='test'
    else:
        sorting='img' if use_img_features else 'num'
    
    cd = str(os.path.dirname(os.path.abspath(__file__)))
    root = f'{cd}/../../..'
    path = f'{root}/storage/{park}/{model.name}/{sorting}/{timestamp}'
    os.makedirs(path, exist_ok=True)
    
    info = (model, results, train_config, data_config, scaler, target_label_indices)
    with open(f'{path}/model_results_config.pickle', 'wb') as storage_file:
        pickle.dump(info, storage_file)
        
    # test evaluation and write plots
    
    modelEval = ModelEvaluation(timestamp, model.name, park, sorting)

class ModelEvaluation():
    
    def __init__(self, timestamp: str, model_name: str, park: str, sorting: str):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.timestamp = timestamp
        self.model_name = model_name
        self.park = park
        self.sorting = sorting
        
        self.park_name = {
            'bess': 'Bessakerfjellet', 
            'skom': 'Skomakerfjellet', 
            'vals': 'Valsneset', 
            'yvik': 'Ytre Vikna'
        }
        
        self.get_stored_results()
        
        self.train_preds_unscaled, self.train_targs_unscaled = self.unscale_predictions(
            self.results.corr_train_preds, 
            self.results.train_targs
        )
        
        self.valid_preds_unscaled, self.valid_targs_unscaled = self.unscale_predictions(
            self.results.best_val_preds, 
            self.results.val_targs
        )
        
        try:
            self.test_preds_unscaled, self.test_targs_unscaled = self.unscale_predictions(
                self.results.test_preds, 
                self.results.test_targs
            )
        except:
            raise Exception(f'The following model does not conform with the update system: {self.park}, {self.timestamp}, {self.model_name}, {self.sorting}.')
        
        self.train_mae, self.train_error_matrix = self.get_errors(self.train_preds_unscaled, self.train_targs_unscaled)
        self.valid_mae, self.valid_error_matrix = self.get_errors(self.valid_preds_unscaled, self.valid_targs_unscaled)
        self.test_mae, self.test_error_matrix = self.get_errors(self.test_preds_unscaled, self.test_targs_unscaled)
        
        logging.info(f'{self.model_name} Valid MAEs:\n{self.valid_error_matrix}')
        
        self.plot_training_curve()
        
        self.info()
        
        self.plot_errors(self.train_error_matrix, 'train', self.train_mae)
        self.plot_errors(self.valid_error_matrix, 'valid', self.valid_mae)
        self.plot_errors(self.test_error_matrix, 'test', self.test_mae)
        
        self.plot_fitting_example(self.train_preds_unscaled, self.train_targs_unscaled, 'train')
        self.plot_fitting_example(self.valid_preds_unscaled, self.valid_targs_unscaled, 'valid')
        self.plot_fitting_example(self.test_preds_unscaled, self.test_targs_unscaled, 'test')
        
        
        
        '''
        self.persistence_train_mae, self.persistence_train_error_matrix = self.persistence('train')
        self.persistence_valid_mae, self.persistence_valid_error_matrix = self.persistence('valid')
        self.persistence_test_mae, self.persistence_test_error_matrix = self.persistence('test')
        
        logging.info(f'Persistence Valid MAEs:\n{self.persistence_valid_error_matrix}')
        
        self.baseline_comparisons(self.train_error_matrix, self.persistence_train_error_matrix, 'train')
        self.baseline_comparisons(self.valid_error_matrix, self.persistence_valid_error_matrix, 'valid')
        self.baseline_comparisons(self.test_error_matrix, self.persistence_test_error_matrix, 'test')
        '''
    
    def get_stored_results(self):
        self.path = f'{self.root}/storage/{self.park}/{self.model_name}/{self.sorting}/{self.timestamp}'
        with open(f'{self.path}/model_results_config.pickle', 'rb') as storage_file:
            net, results, train_config, data_config, scaler, target_label_indices = pickle.load(storage_file)
            
        self.net = net
        self.results = results
        self.train_config = train_config
        self.data_config = data_config
        self.scaler = scaler
        self.target_label_indices = target_label_indices
        self.wind_dataset = WindDataset(self.data_config)
    
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
        
        info_str += f'\nTrain MAEs:\n{self.train_error_matrix}'
        info_str += f'\nValid MAEs:\n{self.valid_error_matrix}'
        info_str += f'\nTest MAEs:\n{self.test_error_matrix}'
        
        try:
            info_str += f'\nPersistence Train MAEs:\n{self.persistence_train_error_matrix}'
            info_str += f'\nPersistence Valid MAEs:\n{self.persistence_valid_error_matrix}'
            info_str += f'\nPersistence Test MAEs:\n{self.persistence_test_error_matrix}'
        except:
            pass
        
        if to_console: logging.info(info_str)
        
        with open(f'{self.path}/info.txt', 'w') as info_file:
            info_file.write(info_str)
    
    def unscale_predictions(self, best_preds, targs):
        preds = np.array(best_preds)
        targs = np.array(targs)
        
        means = [self.scaler.mean_[x] for x in self.target_label_indices]
        stds = [self.scaler.scale_[x] for x in self.target_label_indices]
        
        for step in range(self.data_config.pred_sequence_length):
            
            p = preds[:, step, :]
            t = targs[:, step, :]
            
            for park_idx in range(p.shape[1]):
                p[:,park_idx] = p[:,park_idx]*stds[park_idx]+means[park_idx]
                t[:,park_idx] = t[:,park_idx]*stds[park_idx]+means[park_idx]
            
            preds[:, step, :] = p
            targs[:, step, :] = t
        
        return preds, targs
    
    def get_errors(self, preds_unscaled, targs_unscaled):
        mae = mean_absolute_error(np.ravel(targs_unscaled), np.ravel(preds_unscaled))
        
        error_matrix = np.zeros((targs_unscaled.shape[1], targs_unscaled.shape[2]))
        for i in range(error_matrix.shape[0]):
            for j in range(error_matrix.shape[1]):
                error_matrix[i][j] = mean_absolute_error(targs_unscaled[:,i,j], preds_unscaled[:,i,j])
        
        return mae, error_matrix
    
    def plot_errors(self, error_matrix: [[float]], dataset_name: str, total_mae: float):
        fig, axs = plt.subplots(error_matrix.shape[1], 1, constrained_layout=True)
        
        if error_matrix.shape[1]==1:
            axis = []
            axis.append(axs)
        else:
            axis = axs
        
        fig.suptitle('MAE')
        for i in range(error_matrix.shape[1]):
            axis[i].bar(range(error_matrix.shape[0]), error_matrix.T[i], align='center')
            axis[i].set_title(f'{self.data_config.parks[i]} - total {dataset_name} MAE: {total_mae}')
            axis[i].set_xlabel('hours ahead')
            axis[i].set_ylabel('MAE')
            axis[i].set_xticks(range(error_matrix.shape[0]))
            axis[i].set_xticklabels([f'{h+1}' for h in range(error_matrix.shape[0])])
        plt.savefig(f'{self.path}/{dataset_name}_maes.png')
        plt.close()
    
    def plot_fitting_example(self, preds_unscaled, targs_unscaled, dataset_name: str):
        
        plt.figure()
        plt.ylabel('production (kWh/h)')
        plt.xlabel('hours')
        index = 1
        n_samples_shown = np.minimum(50, targs_unscaled.shape[0])
        n_plots = self.data_config.pred_sequence_length*len(self.data_config.parks)
        for i in range(self.data_config.pred_sequence_length):
            for j in range(len(self.data_config.parks)):
                plt.subplot(self.data_config.pred_sequence_length, len(self.data_config.parks), index) #(nrows, ncols, index)
                plt.plot(np.ravel(np.array(targs_unscaled)[:n_samples_shown, i, j]), label='targets')
                plt.plot(np.ravel(np.array(preds_unscaled)[:n_samples_shown, i, j]), label='predicitions')
                
                
                plt.title(f'{i+1}h ahead at {self.park_name[self.data_config.parks[j]]}, {dataset_name} set', fontsize=8)
                if index==n_plots:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                else:
                    plt.xticks([])
                plt.tight_layout()
                index += 1
        plt.tight_layout()
        plt.savefig(f'{self.path}/{dataset_name}_fitting_examples.png')
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
        
        
        
        
    def persistence(self, dataset_name: str):
        
        # get relevant production
        prod = get_columns(self.wind_dataset.data_unscaled, 'production')
        prod_columns = prod.columns
        
        persistence_error_matrix = np.zeros_like(self.train_error_matrix) #(5,4)
        
        for i in range(self.data_config.pred_sequence_length):
            for col in prod_columns:
                prod[f'{col}_shift({i+1})'] = prod[col].shift(i+1).values
        if dataset_name=='train':
            prod = prod.loc[:self.data_config.valid_start-timedelta(hours=1)].dropna(axis=0)
        elif dataset_name=='valid':
            prod = prod.loc[
                self.data_config.valid_start:self.data_config.test_start-timedelta(hours=1)
            ].dropna(axis=0)
        elif dataset_name=='test':
            prod = prod.loc[self.data_config.test_start:].dropna(axis=0)
        else:
            raise Exception(f'The dataset name should be either "train", "valid" or "test", not {dataset_name}.')
        
        for i in range(self.data_config.pred_sequence_length):
            for j, col in enumerate(prod_columns):
                persistence_error_matrix[i,j] = mean_absolute_error(
                    prod[col].values, prod[f'{col}_shift({i+1})'].values)
        
        persistence_mae = np.mean(np.ravel(persistence_error_matrix))
        
        return persistence_mae, persistence_error_matrix
        
    def baseline_comparisons(self, error_matrix, persistence_error_matrix, dataset_name: str):
        self.avg_of_history_error_matrix = None
        
        if self.avg_of_history_error_matrix is not None:
            logging.info(f'Average history based model MAEs:\n{self.avg_of_history_error_matrix}')
        
        labels = ['+1h', '+2h', '+3h', '+4h', '+5h']
        per = np.ravel(persistence_error_matrix)
        '''if self.avg_of_history_error_matrix is not None:
            lin = np.ravel(self.avg_of_history_error_matrix)'''
        lin = None
        lstm = np.ravel(error_matrix)

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
        ax.set_title(f'MAE comparison on the {dataset_name} set')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        fig.tight_layout()
        plt.savefig(f'{self.path}/{dataset_name}_error_comparison.png')
        plt.close()