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
import pickle

from satprod.pipelines.dataset import WindDataset

from satprod.ml.lstm import LSTM
from satprod.ml.tcn import TCN
from satprod.ml.tcn_bai import TCN_Bai
from satprod.ml.mlr import MLR
from satprod.ml.sin import SIN

from satprod.data_handlers.data_utils import get_columns
from satprod.configs.job_configs import TrainConfig, DataConfig
from satprod.pipelines.evaluation import Results, ModelEvaluation, store_results

from tasklog.tasklogger import logging
from tqdm import tqdm

def train_model(config):
    """Builds a dataset and a model that fit each other,
    trains the model on the train data,
    saves the trained model with the lowest error on the validation data,
    and runs this model on the test set. The model, configs, and predictions
    on each split of the set is stored in the 'storage/' folder,
    tagged with the name of the park, model name, and the point in time
    when the model was stored.

    Args:
        config (munch.Munch): the contents of 'config.yaml', located in the project root directory.
    
    レットウンス　ノチング
    """
    
    # initialize dataset and model based on the configuration dict
    net, train_config, data_config, dataset = init_data_and_model(config)
    
    # train the model and return the model with the lowest validation error
    best_model, results = train_loop(net, train_config, data_config, dataset)
    
    # store results tagged with the current time and the model type
    store_results(
        model=best_model.cpu(),
        train_config=train_config,
        data_config=data_config,
        results=results,
        scaler=dataset.scaler,
        target_label_indices=dataset.target_label_indices,
        use_img_features=data_config.use_img_features,
        train_on_one_batch=config.train_on_one_batch
    )

def train_loop(net, train_config: TrainConfig, data_config: DataConfig, data: WindDataset):
    """Train the input network on the given dataset, using parameters from he train and data configs.

    Args:
        net: the network to train
        train_config (TrainConfig): a dataclass containing parameters specific to training
        data_config (DataConfig): a dataclass containing which model and which features and splits to use
        data (WindDataset): the dataset structured to fit the model

    Returns:
        best_model: the trained network with the lowest validation error
        results (Results): a dataclass containing results and predictions
    """
    
    sequence_length = net.sequence_length
    logging.info(f'Model: {net.name}')
    logging.info(f'Sequence length: {sequence_length}')
    
    pred_sequence_length = data_config.pred_sequence_length
    batch_size = train_config.batch_size
    num_epochs = train_config.num_epochs
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    net.to(device)
    
    #logging.info(net)
    params_in_network = 0
    trainable_params_in_network = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            trainable_params_in_network += len(np.ravel(param.data.cpu().numpy()))
            #print(name, param.data.numpy().shape)
        params_in_network += len(np.ravel(param.data.cpu().numpy()))
    
    logging.info(f'Trainable parameters in network: {trainable_params_in_network}.')
    logging.info(f'Parameters in network: {params_in_network}.')
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=train_config.learning_rate)
    
    # get train and valid indices
    train_indices = np.arange(sequence_length, data.valid_start_index - pred_sequence_length)
    valid_indices = np.arange(data.valid_start_index + sequence_length, data.test_start_index - pred_sequence_length)
    test_indices = np.arange(data.test_start_index + sequence_length, len(data) - pred_sequence_length)
    
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
        train_targs = [],
        test_preds = [],
        test_targs = [],
        test_mae = np.inf
    )
    lowest_train_mae = np.inf
    
    num_samples_train = len(train_indices)
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = len(valid_indices)
    num_batches_valid = num_samples_valid // batch_size
    num_samples_test = len(test_indices)
    num_batches_test = num_samples_test // batch_size
    
    if train_config.train_on_one_batch:
        num_batches_train = 2
        train_indices = train_indices[:(num_batches_train+1)*batch_size]
        num_samples_train = len(train_indices)
        num_batches_valid = 2
        valid_indices = valid_indices[:(num_batches_valid+1)*batch_size]
        num_samples_valid = len(valid_indices)
        num_batches_test = 2
        test_indices = test_indices[:(num_batches_test+1)*batch_size]
        num_samples_test = len(test_indices)
    
    logging.info(f'Number of train, valid and test samples: {num_samples_train}, {num_samples_valid}, {num_samples_test}')
    logging.info(f'Number of train, valid and test batches: {num_batches_train}, {num_batches_valid}, {num_batches_test}')
    
    scheduler = StepLR(
        optimizer,
        step_size=train_config.scheduler_step_size, 
        gamma=train_config.scheduler_gamma
    )
    
    # temporary storage path
    now = datetime.now()
    path = f'storage'
    os.makedirs(path, exist_ok=True)
    
    for epoch in range(num_epochs):
        # shuffle
        np.random.shuffle(train_indices)
        
        ## Train
        net.train()
        for i in tqdm(range(num_batches_train+1), desc="Training"):
            train_batch_indices = train_indices[i*batch_size:(i+1)*batch_size]
            
            optimizer.zero_grad()
            
            data_dict = get_sequenced_data(
                batch_indices = train_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data_config = data_config,
                data = data,
                device = device
            )
            
            if data_dict['X_prod'] is None and data_dict['X_img'] is None: continue
            if data_config.use_img_features > 0:
                current_batch_size = data_dict['X_img'].shape[0]
            else:
                current_batch_size = data_dict['X_prod'].shape[0]
            
            X_prod = data_dict['X_prod']
            if 'production' not in data_config.numerical_features:
                data_dict['X_prod'] = None
            
            output = net(data_dict)
            
            # compute gradients given loss
            y_prod = data_dict['y_prod']
            if data_config.model=='TCN_Bai' or data_config.model=='TCN':
                if not net.only_predict_future_values:
                    y_prod = torch.cat([X_prod[:, pred_sequence_length:, :], data_dict['y_prod']], dim=1)
            
            batch_loss = criterion(output, y_prod)
            
            batch_loss.backward()
            optimizer.step()
        
        net.eval()
        ## Evaluate training
        train_preds, train_targs = [], []
        for i in tqdm(range(num_batches_train+1), desc="Train evaluation"):
            train_batch_indices = np.sort(train_indices)[i*batch_size:(i+1)*batch_size]
            
            data_dict = get_sequenced_data(
                batch_indices = train_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data_config = data_config,
                data = data,
                device = device
            )
            
            if data_dict['X_prod'] is None and data_dict['X_img'] is None: continue
            if data_config.use_img_features:
                current_batch_size = data_dict['X_img'].shape[0]
            else:
                current_batch_size = data_dict['X_prod'].shape[0]
            
            if 'production' not in data_config.numerical_features:
                data_dict['X_prod'] = None
            
            output = net(data_dict)[:, -pred_sequence_length:, :]
            
            train_targs += list(data_dict['y_prod'].data.cpu().numpy())
            train_preds += list(output.data.cpu().numpy())
        
        ## Evaluate validation
        val_preds, val_targs = [], []
        for i in tqdm(range(num_batches_valid+1), desc="Valid evaluation"):
            valid_batch_indices = valid_indices[i*batch_size:(i+1)*batch_size]
            
            data_dict = get_sequenced_data(
                batch_indices = valid_batch_indices, 
                sequence_length = sequence_length, 
                train_config = train_config, 
                data_config = data_config,
                data = data,
                device = device
            )
            
            if data_dict['X_prod'] is None and data_dict['X_img'] is None: continue
            if data_config.use_img_features:
                current_batch_size = data_dict['X_img'].shape[0]
            else:
                current_batch_size = data_dict['X_prod'].shape[0]
            
            if 'production' not in data_config.numerical_features:
                data_dict['X_prod'] = None
            
            output = net(data_dict)[:, -pred_sequence_length:, :]
            
            val_targs += list(data_dict['y_prod'].data.cpu().numpy())
            val_preds += list(output.data.cpu().numpy())
        
        def fill_nans_and_clip_infs(arr):
            '''Fills in invalid values in an array if required at runtime.
            '''
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        
        try:
            train_mae_cur = mean_absolute_error(np.ravel(train_targs), np.ravel(train_preds))
            valid_mae_cur = mean_absolute_error(np.ravel(val_targs), np.ravel(val_preds))
        except:
            logging.warning('Predictions included invalid values. Fixed with forward fill.')
            
            train_targs = fill_nans_and_clip_infs(train_targs)
            train_preds = fill_nans_and_clip_infs(train_preds)
            val_targs = fill_nans_and_clip_infs(val_targs)
            val_preds = fill_nans_and_clip_infs(val_preds)
            
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
            
            # save the model with the lowest valid error to file
            
            with open(f'{path}/best_model_{now}.pickle', 'wb') as model_file:
                pickle.dump(net, model_file)
        
        if train_mae_cur < lowest_train_mae:
            lowest_train_mae = train_mae_cur
            results.corr_train_mae = train_mae_cur
            results.corr_train_preds = train_preds
            results.train_targs = train_targs
            
        
        scheduler.step()
        
        # plot the current state of the training
        temporary_plot(results, val_targs, val_preds)
        
        logging.info('Epoch %2i : Train MAE %f, Valid MAE %f' % (epoch+1, train_mae_cur, valid_mae_cur))
    
    with open(f'{path}/best_model_{now}.pickle', 'rb') as model_file:
        best_model = pickle.load(model_file)
    
    os.remove(f'{path}/best_model_{now}.pickle')
    
    ## Evaluate test
    test_preds, test_targs = [], []
    for i in tqdm(range(num_batches_test+1), desc="Test evaluation"):
        test_batch_indices = test_indices[i*batch_size:(i+1)*batch_size]
        
        data_dict = get_sequenced_data(
            batch_indices = test_batch_indices, 
            sequence_length = sequence_length, 
            train_config = train_config, 
            data_config = data_config,
            data = data,
            device = device
        )
        
        if data_dict['X_prod'] is None and data_dict['X_img'] is None: continue
        if data_config.use_img_features:
            current_batch_size = data_dict['X_img'].shape[0]
        else:
            current_batch_size = data_dict['X_prod'].shape[0]
            
        if 'production' not in data_config.numerical_features:
            data_dict['X_prod'] = None
        
        output = best_model(data_dict)[:, -pred_sequence_length:, :]
        
        test_targs += list(data_dict['y_prod'].data.cpu().numpy())
        test_preds += list(output.data.cpu().numpy())
    
    test_mae = mean_absolute_error(np.ravel(test_targs), np.ravel(test_preds))
    results.test_mae = test_mae
    results.test_preds = test_preds
    results.test_targs = test_targs
    
    return best_model, results
    
def get_sequenced_data(
    batch_indices: [int], 
    sequence_length: int,
    train_config: TrainConfig, 
    data_config: DataConfig, 
    data: WindDataset, 
    device):
    
    data_dict = {
        'X_prod': None, 
        'X_weather': None, 
        'X_weather_forecasts': None, 
        'X_img': None, 
        'X_img_forecasts': None, 
        'y_prod': None
    }
    
    # production
    input_prod_array = []
    target_prod_array = []
    
    # weather
    input_weather_array = []
    forecast_weather_array = []
    
    # images
    input_img_array = []
    forecast_img_array = []
    
    for j in batch_indices:
        input_data = data[j-sequence_length:j].copy()
        target_data = data[j:j+data_config.pred_sequence_length][data.target_labels].copy()

        # the input and target data must not contain missing values
        if len(input_data.dropna(axis=0)) < sequence_length: continue
        if len(target_data.dropna(axis=0)) < data_config.pred_sequence_length: continue
        
        # image features
        if data_config.use_img_features > 0:
            img_data = input_data[data_config.img_features].copy()
            input_data = input_data.drop(columns=data_config.img_features)
            
            img_data = img_data.astype(int)
            
            image_sequence = []
            img_indices = np.ravel(img_data[data_config.img_features].values)
            
            for index in img_indices:
                image_sequence.append(data.img_datasets[data_config.img_features[0]][index].img)
            
            img_data = torch.stack(image_sequence)
            
            # add to array
            input_img_array.append(img_data)
            
            # image forecast features
            if data_config.use_img_forecasts:
                img_forecast_data = get_columns(input_data, data_config.img_features[0]).astype(int)
                input_data = input_data.drop(columns=img_forecast_data.columns)
                
                image_sequence = []
                img_indices = np.ravel(img_forecast_data.iloc[-1].values[:data_config.pred_sequence_length])
                
                for index in img_indices:
                    image_sequence.append(data.img_datasets[data_config.img_features[0]][index].img)
                
                img_forecast_data = torch.stack(image_sequence)
                
                # add to array
                forecast_img_array.append(img_forecast_data)
        
        # weather forecasts
        if data_config.use_numerical_forecasts:
            forecast_data = []
            for i in range(data_config.pred_sequence_length):
                forecast_data.append(get_columns(input_data, f'+{i+1}h').iloc[-1].values)
            
            input_data = input_data.drop(columns=get_columns(input_data, '+').columns)
            
            forecast_data = torch.from_numpy(np.array(forecast_data))
            
            # add to array
            forecast_weather_array.append(forecast_data)
        
        input_prod = torch.from_numpy(get_columns(input_data, 'production').values)
        
        input_data = input_data.drop(columns=get_columns(input_data, 'production').columns)
        if not input_data.empty:
            input_data = torch.from_numpy(input_data.values)
            input_weather_array.append(input_data)
        target_data = torch.from_numpy(target_data.values)
        
        # add to array
        input_prod_array.append(input_prod)
        target_prod_array.append(target_data)
    
    # if there are missing values in all the sequences in the batch, return data_dict of None
    if len(input_prod_array)==0 or len(target_prod_array)==0: return data_dict
    
    X_prod = torch.stack(input_prod_array)
    if len(input_weather_array) > 0:
        X_weather = torch.stack(input_weather_array)
        data_dict['X_weather'] = Variable(X_weather).float().to(device)
    
    y_prod = torch.stack(target_prod_array)
    
    data_dict['X_prod'] = Variable(X_prod).float().to(device)
    data_dict['y_prod'] = Variable(y_prod).float().to(device)
    
    if data.n_image_features > 0:
        X_img = torch.stack(input_img_array)
        data_dict['X_img'] = Variable(X_img).float().to(device)
        if data_config.use_img_forecasts:
            X_img_forecasts = torch.stack(forecast_img_array)
            data_dict['X_img_forecasts'] = Variable(X_img_forecasts).float().to(device)
    
    if len(forecast_weather_array) > 0 and len(input_weather_array) > 0:
        X_weather_forecasts = torch.stack(forecast_weather_array)
        data_dict['X_weather_forecasts'] = Variable(X_weather_forecasts).float().to(device)
    
    return data_dict

def init_data_and_model(config):
    """Uses the input config to initialize config objects, the dataset and the model.

    Args:
        config (munch.Munch): [description]

    Raises:
        Exception: checks that the model name fits any of the available models

    Returns:
        net
        train_config
        data_config
        wind_dataset
    """
    train_config = TrainConfig(
        batch_size = config.train_config.batch_size,
        num_epochs = config.train_config.num_epochs,
        learning_rate = config.train_config.learning_rate,
        scheduler_gamma = config.train_config.scheduler_gamma,
        random_seed = config.train_config.random_seed,
        train_on_one_batch = config.train_on_one_batch
    )
    
    logging.info(vars(train_config))
    
    data_config = DataConfig(
        model = config.model, 
        parks = config.data_config.parks,
        numerical_features = config.data_config.numerical_features, 
        use_img_features = config.data_config.use_img_features, 
        img_extraction_method = config.data_config.img_extraction_method,
        pred_sequence_length = config.data_config.pred_sequence_length,
        use_numerical_forecasts = config.data_config.use_numerical_forecasts,
        use_img_forecasts = config.data_config.use_img_forecasts,
        crop_image = config.data_config.crop_image,
        valid_start = config.data_config.valid_start,
        test_start = config.data_config.test_start,
        test_end = config.data_config.test_end
    )
    
    logging.info(vars(data_config))
    
    # get data
    wind_dataset = WindDataset(data_config)
    
    # define image extraction parameters
    lenet_params = {
        'channels' : config.img_extraction_methods.lenet.channels,
        'kernel_size_conv' : config.img_extraction_methods.lenet.kernel_size_conv,
        'stride_conv' : config.img_extraction_methods.lenet.stride_conv,
        'padding_conv' : config.img_extraction_methods.lenet.padding_conv,
        'kernel_size_pool' : config.img_extraction_methods.lenet.kernel_size_pool,
        'stride_pool' : config.img_extraction_methods.lenet.stride_pool,
        'padding_pool' : config.img_extraction_methods.lenet.padding_pool,
        'height' : wind_dataset.img_height,
        'width' : wind_dataset.img_width
    }
    
    resnet_params = {
        'output_size' : config.img_extraction_methods.resnet.output_size,
        'freeze_all_but_last_layer' : config.img_extraction_methods.resnet.freeze_all_but_last_layer
    }
    
    vgg_params = {
        'output_size' : config.img_extraction_methods.vgg.output_size,
        'freeze_all_but_last_layer' : config.img_extraction_methods.vgg.freeze_all_but_last_layer
    }
    
    # define model structure
    lstm_params = {
        'sequence_length' : config.models.lstm.sequence_length,
        'hidden_size' : config.models.lstm.hidden_size,
        'linear_size' : config.models.lstm.linear_size,
        'num_layers' : config.models.lstm.num_layers,
        'input_size' : wind_dataset.n_past_features,
        'num_forecast_features' : wind_dataset.n_forecast_features,
        'num_img_forecast_features' : wind_dataset.n_img_forecast_features,
        'num_output_features' : wind_dataset.n_output_features,
        'output_size' : data_config.pred_sequence_length,
        'img_extraction_method': data_config.img_extraction_method,
        'lenet_params' : lenet_params,
        'resnet_params' : resnet_params,
        'vgg_params' : vgg_params,
        'dropout': config.models.lstm.dropout
    }
    
    tcn_params = {
        'num_past_features': wind_dataset.n_past_features,
        'output_size': wind_dataset.n_output_features,
        'channels': config.models.tcn.channels,
        'kernel_size': config.models.tcn.kernel_size,
        'dilation_base': config.models.tcn.dilation_base,
        'pred_sequence_length': data_config.pred_sequence_length,
        'img_extraction_method': data_config.img_extraction_method,
        'lenet_params' : lenet_params,
        'resnet_params' : resnet_params,
        'vgg_params' : vgg_params,
        'dropout': config.models.tcn.dropout,
        'only_predict_future_values': config.models.tcn.only_predict_future_values
    }
    
    tcn_bai_params = {
        'input_size': wind_dataset.n_past_features, 
        'output_size': wind_dataset.n_output_features,
        'channels': config.models.tcn_bai.channels, 
        'kernel_size': config.models.tcn_bai.kernel_size, 
        'dropout': config.models.tcn_bai.dropout,
        'pred_sequence_length': data_config.pred_sequence_length
    }
    
    mlr_params = {
        'num_past_features': wind_dataset.n_past_features, 
        'output_size': wind_dataset.n_output_features,
        'sequence_length': config.models.mlr.sequence_length,
        'num_forecast_features': wind_dataset.n_forecast_features, 
        'pred_sequence_length': data_config.pred_sequence_length
    }
    
    sin_params = {
        'output_size': data_config.pred_sequence_length,
        'num_output_features': wind_dataset.n_output_features,
        'img_extraction_method': data_config.img_extraction_method,
        'lenet_params' : lenet_params,
        'resnet_params' : resnet_params,
        'vgg_params' : vgg_params,
        'dropout' : config.models.sin.dropout,
        'sequence_length' : config.models.sin.sequence_length
    }
    
    if config.model=='TCN':
        net = TCN(**tcn_params)
    elif config.model=='LSTM':
        net = LSTM(**lstm_params)
    elif config.model=='TCN_Bai':
        net = TCN_Bai(**tcn_bai_params)
    elif config.model=='MLR':
        net = MLR(**mlr_params)
    elif config.model=='SIN':
        net = SIN(**sin_params)
    else:
        raise Exception(f'The model "{config.model}" does not exist.')
    
    return net, train_config, data_config, wind_dataset
    
def temporary_plot(results: Results, val_targs, val_preds):
    """Shows how the training is progressing during training.

    Args:
        results (Results): Contains MAEs per epoch of the validation set predictions
        val_targs ([float]): The validation set targets
        val_preds ([float]): The validation set predictions made in the current epoch
    """
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