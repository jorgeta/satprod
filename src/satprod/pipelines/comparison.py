from satprod.pipelines.evaluation import ModelEvaluation
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from satprod.data_handlers.data_utils import get_columns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from tasklog.tasklogger import logging

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray', labelsize=20)
plt.rc('ytick', direction='out', color='gray', labelsize=20)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class ModelComparison():
    
    def __init__(self, park: str, config):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.park = park
        self.config = config[park]
        
        self.TE_data_path = f'{self.root}/storage/{park}/TE'
        
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.comparison_storage_path = f'{self.root}/storage/{park}/comparison/{timestamp}'
        os.makedirs(self.comparison_storage_path, exist_ok=True)
        
        self.eval_dict = {}
        for key, value in self.config.items():
            if value is None: 
                if key=='model1':
                    raise Exception('The "model1" slot must be filled, as it is used as the main model of the comparison.')
                continue
            value['park']=self.park
            
            try:
                self.eval_dict[key] = ModelEvaluation(**value)
            except Exception as e:
                logging.info(e)
                logging.info(f'Unable to evaluate {value}.')
                continue
        
        self.wind_dataset = self.eval_dict['model1'].wind_dataset
        self.pred_sequence_length = self.eval_dict['model1'].data_config.pred_sequence_length
        self.valid_start = self.eval_dict['model1'].data_config.valid_start
        self.test_start = self.eval_dict['model1'].data_config.test_start
        
        self.benchmark_test_mae, self.benchmark_test_error_matrix = self.benchmark()
        
        self.persistence_train_mae, self.persistence_train_error_matrix = self.persistence('train')
        self.persistence_valid_mae, self.persistence_valid_error_matrix = self.persistence('valid')
        self.persistence_test_mae, self.persistence_test_error_matrix = self.persistence('test')
        
        logging.info(f'Benchmark Test MAEs:\n{self.benchmark_test_error_matrix}')
        logging.info(f'Persistence Test MAEs:\n{self.persistence_test_error_matrix}')
        
        self.baseline_comparisons('train')
        self.baseline_comparisons('valid')
        self.baseline_comparisons('test')
        
        self.compare_LSTM_information()
        self.compare_TCN_information()
        
    def benchmark(self):
        with open(f'{self.TE_data_path}/TE_predictions.pickle', 'rb') as model_file:
            TE_predictions = pickle.load(model_file).asfreq('H').copy()
        
        prod = get_columns(self.wind_dataset.data_unscaled, 'production').asfreq('H')
        prod_columns = prod.columns
        for i in range(0, self.pred_sequence_length+1):
            prod[f'y_{i}'] = prod[f'production_{self.park}'].shift(-i).copy()
        prod = prod.drop(columns=[f'production_{self.park}'])
        df = pd.concat([TE_predictions, prod], axis=1)
        df = df.dropna(axis=0)
        
        if self.pred_sequence_length==6:
            benchmark_test_error_matrix = np.zeros_like(self.eval_dict['model1'].train_error_matrix[1:,:]) #(5,4)
        else:
            benchmark_test_error_matrix = np.zeros_like(self.eval_dict['model1'].train_error_matrix) #(5,4)
        
        for i in range(self.pred_sequence_length):
            for j in range(len(prod_columns)):
                benchmark_test_error_matrix[i,j] = mean_absolute_error(
                    df[f'{i}'].values, df[f'y_{i}'].values)
        
        benchmark_test_mae = np.mean(np.ravel(benchmark_test_error_matrix))
        
        return benchmark_test_mae, benchmark_test_error_matrix
    
    def persistence(self, dataset_name: str):
        
        # get relevant production
        prod = get_columns(self.wind_dataset.data_unscaled, 'production')
        prod_columns = prod.columns
        
        if self.pred_sequence_length==6:
            persistence_error_matrix = np.zeros_like(self.eval_dict['model1'].train_error_matrix[1:,:]) #(5,4)
        else:
            persistence_error_matrix = np.zeros_like(self.eval_dict['model1'].train_error_matrix) #(5,4)
        
        for i in range(self.pred_sequence_length):
            for col in prod_columns:
                prod[f'{col}_shift({i+1})'] = prod[col].shift(i+1).values
        if dataset_name=='train':
            prod = prod.loc[:self.valid_start-timedelta(hours=1)].dropna(axis=0)
        elif dataset_name=='valid':
            prod = prod.loc[
                self.valid_start:self.test_start-timedelta(hours=1)
            ].dropna(axis=0)
        elif dataset_name=='test':
            prod = prod.loc[self.test_start:].dropna(axis=0)
        else:
            raise Exception(f'The dataset name should be either "train", "valid" or "test", not {dataset_name}.')
        
        for i in range(self.pred_sequence_length):
            for j, col in enumerate(prod_columns):
                persistence_error_matrix[i,j] = mean_absolute_error(
                    prod[col].values, prod[f'{col}_shift({i+1})'].values)
        
        persistence_mae = np.mean(np.ravel(persistence_error_matrix))
        
        return persistence_mae, persistence_error_matrix
        
    def baseline_comparisons(self, dataset_name: str):
        
        # gather models to plot depending on the part of the dataset that is plotted
        models = {}
        if dataset_name=='train':
            models['Persistence'] = self.persistence_train_error_matrix
            for key, model in self.eval_dict.items():
                if self.pred_sequence_length==6:
                    models[f'{model.model_name}'+f'{model.timestamp}'] = model.train_error_matrix[1:,:]
                else:
                    models[f'{model.model_name}'+f'{model.timestamp}'] = model.train_error_matrix
        elif dataset_name=='valid':
            models['Persistence'] = self.persistence_valid_error_matrix
            for key, model in self.eval_dict.items():
                if self.pred_sequence_length==6:
                    models[f'{model.model_name}'+f'{model.timestamp}'] = model.train_error_matrix[1:,:]
                else:
                    models[f'{model.model_name}'+f'{model.timestamp}'] = model.valid_error_matrix
        else:
            models['TE (benchmark)'] = self.benchmark_test_error_matrix
            models['Persistence'] = self.persistence_test_error_matrix
            for key, model in self.eval_dict.items():
                if self.pred_sequence_length==6:
                    models[f'{model.model_name}'+f'{model.timestamp}'] = model.train_error_matrix[1:,:]
                else:
                    models[f'{model.model_name} ({model.timestamp})'] = model.test_error_matrix
        
        width = 0.05  # the width of the bars
        labels = ['+1h', '+2h', '+3h', '+4h', '+5h']

        x_axis = []
        x_axis.append(np.arange(len(labels)))  # the label locations
        for i in range(len(models.keys())-1):
            x_axis.append([r + width for r in x_axis[-1]])
        
        fig, ax = plt.subplots()
        rects = []
        for i, (key, value) in enumerate(models.items()):
            rects.append(ax.bar(x_axis[i], np.ravel(value), width, label=key))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE')
        ax.set_xlabel('hours ahead')
        dataset_name_name = dataset_name
        if dataset_name_name=='valid':
            dataset_name_name = 'validation'
        ax.set_title(f'MAE comparison on the {dataset_name_name} set')
        ax.set_xticks(x_axis[-1])
        ax.set_xticklabels(labels)
        #ax.legend()
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.comparison_storage_path}/{dataset_name}_comparison.png')
        
    def compare_LSTM_information(self):
        df = pd.DataFrame()
        
        lowest_valid_maes = []
        features_used = []
        params_in_networks = []
        trainable_params_in_networks = []
        linear_sizes = []
        hidden_sizes = []
        sequence_lengths = []
        
        for model_id, model in self.eval_dict.items():
            if model.model_name!='LSTM': continue
            features_used.append(model.data_config.numerical_features)
            trainable_params_in_networks.append(model.results.trainable_params_in_network)
            params_in_networks.append(model.results.params_in_network)
            lowest_valid_maes.append(model.results.lowest_valid_mae)
            for key, value in vars(model.net).items():
                if not key.startswith('_') and key!='training':
                    if key=='linear_size':
                        linear_sizes.append(value)
                    if key=='hidden_size':
                        hidden_sizes.append(value)
                    if key=='sequence_length':
                        sequence_lengths.append(value)
        
        df['lowest_valid_mae'] = lowest_valid_maes
        df['features'] = features_used
        df['params_in_network'] = params_in_networks
        df['trainable_params_in_network'] = trainable_params_in_networks
        df['linear_size'] = linear_sizes
        df['hidden_size'] = hidden_sizes
        df['sequence_length'] = sequence_lengths
        
        logging.info(df)
        
        df.to_csv(f'{self.comparison_storage_path}/lstm_comparison.csv')
        
    def compare_TCN_information(self):
        df = pd.DataFrame()
        
        lowest_valid_maes = []
        features_used = []
        params_in_networks = []
        trainable_params_in_networks = []
        channel_list = []
        kernel_sizes = []
        dropouts = []
        sequence_lengths = []
        
        for model_id, model in self.eval_dict.items():
            if model.model_name!='TCN': continue
            features_used.append(model.data_config.numerical_features)
            trainable_params_in_networks.append(model.results.trainable_params_in_network)
            params_in_networks.append(model.results.params_in_network)
            lowest_valid_maes.append(model.results.lowest_valid_mae)
            for key, value in vars(model.net).items():
                if not key.startswith('_') and key!='training':
                    if key=='channels':
                        channel_list.append(value)
                    if key=='kernel_size':
                        kernel_sizes.append(value)
                    if key=='dropout':
                        dropouts.append(value)
                    if key=='sequence_length':
                        sequence_lengths.append(value)
        
        df['lowest_valid_mae'] = lowest_valid_maes
        df['features'] = features_used
        df['params_in_network'] = params_in_networks
        df['trainable_params_in_network'] = trainable_params_in_networks
        df['channels'] = channel_list
        df['kernel_size'] = kernel_sizes
        df['dropout'] = dropouts
        df['sequence_length'] = sequence_lengths
        
        logging.info(df)
        
        df.to_csv(f'{self.comparison_storage_path}/tcn_comparison.csv')