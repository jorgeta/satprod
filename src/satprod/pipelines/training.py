from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
#import pickle

from satprod.pipelines.dataset import WindDataset

from satprod.ml.lstm import LSTM
from satprod.ml.agru import AGRU
from satprod.ml.tcn import TCN

from satprod.data_handlers.data_utils import get_columns
from satprod.configs.job_configs import TrainConfig
from satprod.pipelines.evaluation import Results, Evaluate, store_results

from tasklog.tasklogger import logging
from tqdm import tqdm

def train_model():
    
    parks = ['skom']#, ['bess', 'vals','skom','yvik']
    num_feature_types = ['production', 'speed', 'direction'] #['forecast', 'direction', 'speed', 'production']
    use_numerical_forecasts = True
    use_img_forecasts = True
    img_features = []#['grid'] # [] or ['grid']
    img_extraction_method = 'lenet' # lenet, deepsense, resnet
    train_on_small_subset = False
    
    model = 'TCN' #'LSTM' # 'TCN', 'AGRU'
    
    lenet_channels = [1, 16, 32]
    if len(img_features)==0:
        img_extraction_method = None
        lenet_channels[-1] = 0
    if img_extraction_method is None:
        img_features = []
        lenet_channels[-1] = 0
        
    assert len(set(parks))==len(parks), 'The list of parks contains duplicates'
    
    train_config = TrainConfig(
        model = model,
        batch_size = 64,
        num_epochs = 15,
        learning_rate = 4e-3,
        scheduler_step_size = 2,
        scheduler_gamma = 0.9,
        pred_sequence_length = 5,
        random_seed = 0,
        parks = parks,
        num_feature_types = num_feature_types,
        img_features = img_features,
        img_extraction_method = img_extraction_method,
        train_on_small_subset = train_on_small_subset,
        use_numerical_forecasts = use_numerical_forecasts,
        use_img_forecasts = use_img_forecasts
    )
    
    # get data
    wind_dataset = WindDataset(train_config)
    
    # define image extraction parameters
    lenet_params = {
        'channels' : lenet_channels,
        'kernel_size_conv' : [8,4],
        'stride_conv' : [4,4],
        'padding_conv' : [0,0],
        'kernel_size_pool' : [3,2],
        'stride_pool' : [3,1],
        'padding_pool' : [0,0],
        'height' : wind_dataset.img_height,
        'width' : wind_dataset.img_width
    }
    
    resnet_params = {
        'output_size' : 32
    }
    
    deepsense_params = {
        
    }
    
    # define model structure
    lstm_params = {
        'sequence_length' : 12,
        'hidden_size' : 32,
        'linear_size' : 64,
        'num_layers' : 1,
        'input_size' : wind_dataset.n_past_features,
        'num_forecast_features' : wind_dataset.n_forecast_features,
        'num_output_features' : wind_dataset.n_output_features,
        'output_size' : train_config.pred_sequence_length,
        'initialization' : 'xavier',
        'activation' : nn.Tanh(),
        'img_extraction_method': train_config.img_extraction_method,
        'lenet_params' : lenet_params,
        'resnet_params' : resnet_params,
        'deepsense_params' : deepsense_params
    }
    
    tcn_params = {
        'channels' : [1, 32, 32, 1],
        'num_past_features': wind_dataset.n_past_features,
        'num_unique_forecast_features': wind_dataset.n_unique_forecast_features,
        'output_size': wind_dataset.n_output_features,
        'pred_sequence_length': train_config.pred_sequence_length,
        'kernel_size' : 3,
        'dilations' : [1, 2, 4],
        'img_extraction_method': train_config.img_extraction_method,
        'lenet_params' : lenet_params,
        'resnet_params' : resnet_params,
        'deepsense_params' : deepsense_params
    }
    
    if model=='TCN':
        net = TCN(**tcn_params)
    elif model=='LSTM':
        net = LSTM(**lstm_params)
    else:
        exit()

    # train the model and return the model with the lowest validation error
    best_model, results = train_loop(net, train_config, wind_dataset)
    
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    store_results(
        timestamp=now,
        model=best_model,
        train_config=train_config,
        results=results,
        scaler=wind_dataset.scaler,
        target_label_indices=wind_dataset.target_label_indices
    )
    
    if len(train_config.parks)==1: park = train_config.parks[0]
    else: park = 'all'
    
    evaluate = Evaluate(timestamp=now, model_name=best_model.name, park=park)
    evaluate.plot_errors()

def train_loop(net, train_config: TrainConfig, data: WindDataset):
    sequence_length = net.sequence_length
    pred_sequence_length = train_config.pred_sequence_length
    batch_size = train_config.batch_size
    num_epochs = train_config.num_epochs
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #logging.info(net)
    params_in_network = 0
    trainable_params_in_network = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            trainable_params_in_network += len(np.ravel(param.data.numpy()))
            #print(name, param.data.numpy().shape)
        params_in_network += len(np.ravel(param.data.numpy()))
        
    logging.info(f'Trainable parameters in network: {trainable_params_in_network}.')
    logging.info(f'Parameters in network: {params_in_network}.')
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=train_config.learning_rate)
    
    # get train and valid indices
    train_indices = np.arange(sequence_length, data.valid_start_index - pred_sequence_length)
    valid_indices = np.arange(data.valid_start_index + sequence_length, data.test_start_index - pred_sequence_length)
    
    # set seed for shuffling train indices
    np.random.seed(train_config.random_seed)

    results = Results(
        trainable_params_in_network = trainable_params_in_network,
        params_in_network = params_in_network,
        epoch = 0,
        lowest_valid_mae = np.inf,
        corr_train_mae = np.inf,
        train_mae = [], 
        valid_mae = [],
        best_val_preds = [],
        val_targs = [],
        corr_train_preds = [],
        train_targs = []
    )
    
    num_samples_train = len(train_indices)
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = len(valid_indices)
    num_batches_valid = num_samples_valid // batch_size
    
    if train_config.train_on_small_subset:
        num_batches_train = 2
        train_indices = train_indices[:(num_batches_train+1)*batch_size]
        num_samples_train = len(train_indices)
        num_batches_valid = 2
        valid_indices = valid_indices[:(num_batches_valid+1)*batch_size]
        num_samples_valid = len(valid_indices)
    
    logging.info(f'Number of train and valid samples: {num_samples_train}, {num_samples_valid}')
    logging.info(f'Number of train and valid batches: {num_batches_train}, {num_batches_valid}')
    
    scheduler = StepLR(
        optimizer,
        step_size=train_config.scheduler_step_size, 
        gamma=train_config.scheduler_gamma
    )
    
    for epoch in range(num_epochs):
        # shuffle
        np.random.shuffle(train_indices)
        
        ## Train
        net.train()
        for i in tqdm(range(num_batches_train+1), desc="Training"):
            train_batch_indices = train_indices[i*batch_size:(i+1)*batch_size]
            
            optimizer.zero_grad()
            
            X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts, y_batch = get_sequenced_data(
                batch_indices = train_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data = data,
                device = device
            )
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            
            if train_config.recursive:
                # train only on one step ahead
                output = net(X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts, y_batch)
            else:
                output = net(X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts)
            
            # compute gradients given loss
            batch_loss = criterion(output, y_batch)
            
            batch_loss.backward()
            optimizer.step()
        
        net.eval()
        ## Evaluate training
        train_preds, train_targs = [], []
        for i in tqdm(range(num_batches_train+1), desc="Train evaluation"):
            train_batch_indices = np.sort(train_indices)[i*batch_size:(i+1)*batch_size]
            
            X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts, y_batch = get_sequenced_data(
                batch_indices = train_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data = data,
                device = device
            )
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            
            output = net(X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts)
            
            train_targs += list(y_batch.data.numpy())
            train_preds += list(output.data.numpy())
        
        ## Evaluate validation
        val_preds, val_targs = [], []
        for i in tqdm(range(num_batches_valid+1), desc="Valid evaluation"):
            valid_batch_indices = valid_indices[i*batch_size:(i+1)*batch_size]
            
            X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts, y_batch = get_sequenced_data(
                batch_indices = valid_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data = data,
                device = device
            )
            
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            
            output = net(X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts)
            
            val_targs += list(y_batch.data.numpy())
            val_preds += list(output.data.numpy())
        
        train_mae_cur = mean_absolute_error(np.ravel(train_targs), np.ravel(train_preds))
        valid_mae_cur = mean_absolute_error(np.ravel(val_targs), np.ravel(val_preds))
        
        results.train_mae.append(train_mae_cur)
        results.valid_mae.append(valid_mae_cur)
        
        ## Store best Parameters
        if valid_mae_cur < results.lowest_valid_mae:
            results.lowest_valid_mae = valid_mae_cur
            results.corr_train_mae = train_mae_cur
            results.epoch = epoch
            results.best_val_preds = val_preds
            results.val_targs = val_targs
            results.corr_train_preds = train_preds
            results.train_targs = train_targs
            best_model = deepcopy(net)
        
        scheduler.step()
        
        # plot the current state of the training
        temporary_plot(results, val_targs, val_preds)
        
        logging.info('Epoch %2i : Train MAE %f, Valid MAE %f' % (epoch+1, train_mae_cur, valid_mae_cur))
    
    return best_model, results

def get_sequenced_data(batch_indices: [int], sequence_length: int, train_config: TrainConfig, data: WindDataset, device):
    input_data_array = []
    target_data_array = []
    forecast_data_array = []
    img_data_array = []
    img_forecast_data_array = []
    last_production_data_array = []
    
    for j in batch_indices:
        input_data = data[j-sequence_length:j].copy()
        target_data = data[j:j+train_config.pred_sequence_length][data.target_labels].copy()

        if len(input_data.dropna(axis=0)) < sequence_length: continue
        if len(target_data.dropna(axis=0)) < train_config.pred_sequence_length: continue
        
        if len(train_config.img_features) > 0:
            img_data = input_data[train_config.img_features].copy()
            input_data = input_data.drop(columns=train_config.img_features)
            
            img_data = img_data.astype(int)
            
            image_sequence = []
            img_indices = np.ravel(img_data[train_config.img_features].values)
            for index in img_indices:
                image_sequence.append(data.img_datasets[train_config.img_features[0]][index].img)
                
            img_data = torch.from_numpy(np.array(image_sequence))
            img_data_array.append(img_data)
            
            if train_config.use_img_forecasts:
                img_forecast_data = get_columns(input_data, train_config.img_features[0]).astype(int)
                input_data = input_data.drop(columns=img_forecast_data.columns)
                
                image_sequence = []
                img_indices = np.ravel(img_forecast_data.iloc[-1].values[:train_config.pred_sequence_length-1])
                
                for index in img_indices:
                    image_sequence.append(data.img_datasets[train_config.img_features[0]][index].img)
                
                img_forecast_data = torch.from_numpy(np.array(image_sequence))
                img_forecast_data_array.append(img_forecast_data)
        
        #print(get_columns(input_data, f'+{1}h').iloc[-1])
        if train_config.use_numerical_forecasts:
            forecast_data = []
            for i in range(train_config.pred_sequence_length):
                forecast_data.append(get_columns(input_data, f'+{i+1}h').iloc[-1].values)
            
            input_data = input_data.drop(columns=get_columns(input_data, '+').columns)
            forecast_data = torch.from_numpy(np.array(forecast_data))
            forecast_data_array.append(forecast_data)
        #print(input_data)
        #print(forecast_data)
        last_production_data = torch.from_numpy(get_columns(input_data, 'production').iloc[-1].values)
        input_data = torch.from_numpy(input_data.values)
        target_data = torch.from_numpy(target_data.values)
        #print(input_data)
        last_production_data_array.append(last_production_data)
        input_data_array.append(input_data)
        target_data_array.append(target_data)
        #exit()
    
    if len(input_data_array)==0:
        return None, None, None, None, None, None
    if len(target_data_array)==0:
        return None, None, None, None, None, None
    
    X_batch = torch.stack(input_data_array)
    y_batch = torch.stack(target_data_array)
    X_last_production = torch.stack(last_production_data_array)
    
    if data.n_image_features > 0:
        X_batch_img = torch.stack(img_data_array)
        X_batch_img = Variable(X_batch_img).float().to(device)
        if train_config.use_img_forecasts:
            X_batch_img_forecasts = torch.stack(img_forecast_data_array)
            X_batch_img_forecasts = Variable(X_batch_img_forecasts).float().to(device)
        else:
            X_batch_img_forecasts = None
    else:
        X_batch_img = None
        X_batch_img_forecasts = None
    if len(forecast_data_array) > 0:
        X_batch_forecasts = torch.stack(forecast_data_array)
        X_batch_forecasts = Variable(X_batch_forecasts).float().to(device)
    else:
        X_batch_forecasts = None
        
    X_batch = Variable(X_batch).float().to(device)
    y_batch = Variable(y_batch).float().to(device)
    X_last_production = Variable(X_last_production).float().to(device)
    
    return X_batch, X_batch_forecasts, X_batch_img, X_last_production, X_batch_img_forecasts, y_batch

def temporary_plot(results: Results, val_targs, val_preds):
    plt.figure(figsize=(16,7))
    plt.subplot(1, 2, 1) #(nrows, ncols, index)
    plt.plot(np.ravel(np.array(val_targs)[:, 0, 0])[:100], label='targs')
    plt.plot(np.ravel(np.array(val_preds)[:, 0, 0])[:100], label='preds')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    epochs = np.arange(len(results.train_mae))
    plt.plot(epochs, results.train_mae, 'r', epochs, results.valid_mae, 'b')
    plt.legend(['train MAE','validation MAE'])
    plt.xlabel('epochs')
    plt.ylabel('MAE')
    
    plt.show(block=False)
    plt.pause(0.25)
    plt.close()
    
