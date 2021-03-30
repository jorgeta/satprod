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
from satprod.ml.models import LSTM_net
from satprod.data_handlers.data_utils import get_columns

from tasklog.tasklogger import logging

def test_forward_pass():
    raise NotImplementedError

def train_model(
    sequence_length: int=None,
    pred_sequence_length: int=None,
    ):
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get data
    wind_dataset = WindDataset()
    #print(wind_dataset.data.columns)
    
    # define hyperparameters, network, criterion and optimizer
    if sequence_length is None: sequence_length  = 12
    if pred_sequence_length is None: pred_sequence_length = 5
    
    hidden_size = 16
    linear_size = 64
    num_layers = 1
    num_epochs = 150
    batch_size = 64
    num_input_features = len(wind_dataset.data.columns)
    num_output_features = len(wind_dataset.target_labels)
    
    net = LSTM_net(
        input_size=num_input_features, 
        hidden_size=hidden_size, 
        linear_size=linear_size,
        output_size=pred_sequence_length,
        num_output_features=num_output_features, 
        num_layers=num_layers, 
        sequence_len=sequence_length
    )
    logging.info(net)
    params_in_network = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            params_in_network += len(np.ravel(param.data.numpy()))
    print(f'Parameters in network: {params_in_network}.')
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    
    # get train and valid indices
    train_indices = np.arange(sequence_length, wind_dataset.train_end_index+1 - pred_sequence_length)
    valid_indices = np.arange(wind_dataset.train_end_index+1 + sequence_length, wind_dataset.test_start_index - pred_sequence_length)
    
    # set seed for shuffling train indices
    np.random.seed(11)
    
    # setting up lists for handling loss/accuracy
    train_mae = []
    valid_mae = []

    lowest_valid_mae = np.inf
    
    num_samples_train = len(train_indices)
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = len(valid_indices)
    num_batches_valid = num_samples_valid // batch_size
    
    print(num_samples_train, num_samples_valid)
    print(num_batches_train, num_batches_valid)
    
    scheduler = StepLR(optimizer, step_size=6, gamma=0.9)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
    
    '''num_batches_train = 0
    train_indices = train_indices[:(num_batches_train+1)*batch_size]
    num_batches_valid = 0
    valid_indices = valid_indices[:(num_batches_valid+1)*batch_size]'''
    
    def get_sequenced_data(batch_indices: [int]):
        input_data_array = []
        target_data_array = []
        for j in batch_indices:
            input_data = wind_dataset[j-sequence_length:j].copy()
            target_data = wind_dataset[j:j+pred_sequence_length][wind_dataset.target_labels].copy()
            
            #input_data = input_data.drop(columns=get_columns(input_data, '+'))
            #input_data = get_columns(input_data, 'bess').copy()
            #target_data = get_columns(target_data, 'bess').copy()
            
            if len(input_data.dropna(axis=0)) < sequence_length: 
                continue
            if len(target_data.dropna(axis=0)) < pred_sequence_length: 
                continue
            
            input_data = torch.from_numpy(input_data.values)
            target_data = torch.from_numpy(target_data.values)
            
            input_data_array.append(input_data)
            target_data_array.append(target_data)
            
        if len(input_data_array)==0:
            return None, None
        if len(target_data_array)==0:
            return None, None
        
        X_batch = torch.stack(input_data_array)
        y_batch = torch.stack(target_data_array)
        
        return X_batch, y_batch
    
    for epoch in range(num_epochs):
        # shuffle
        np.random.shuffle(train_indices)
        
        ## Train
        net.train()
        for i in range(num_batches_train+1):
            train_batch_indices = train_indices[i*batch_size:(i+1)*batch_size]
            #print(train_batch_indices)
            #np.random.shuffle(train_batch_indices)
            
            optimizer.zero_grad()
            
            X_batch, y_batch = get_sequenced_data(train_batch_indices)
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            X_batch = Variable(X_batch).float().to(device)
            y_batch = Variable(y_batch).float().to(device)
            
            output = net(X_batch)
            
            # compute gradients given loss
            batch_loss = criterion(output, y_batch)
            
            batch_loss.backward()
            optimizer.step()
            
        net.eval()
        ## Evaluate training
        train_preds, train_targs = [], []
        for i in range(num_batches_train+1):
            train_batch_indices = np.sort(train_indices)[i*batch_size:(i+1)*batch_size]
            
            X_batch, y_batch = get_sequenced_data(train_batch_indices)
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            
            X_batch = Variable(X_batch).float().to(device)
            y_batch = Variable(y_batch).float().to(device)
            
            output = net(X_batch)
            
            train_targs += list(y_batch.data.numpy())
            train_preds += list(output.data.numpy())
        
        
        ## Evaluate validation
        val_preds, val_targs = [], []
        for i in range(num_batches_valid+1):
            valid_batch_indices = valid_indices[i*batch_size:(i+1)*batch_size]
            
            X_batch, y_batch = get_sequenced_data(valid_batch_indices)
            if X_batch is None: continue
            current_batch_size = X_batch.shape[0]
            
            X_batch = Variable(X_batch).float().to(device)
            y_batch = Variable(y_batch).float().to(device)
            
            output = net(X_batch)
            
            val_targs += list(y_batch.data.numpy())
            val_preds += list(output.data.numpy())
        
        train_mae_cur = mean_absolute_error(np.ravel(train_targs), np.ravel(train_preds))
        valid_mae_cur = mean_absolute_error(np.ravel(val_targs), np.ravel(val_preds))
        
        train_mae.append(train_mae_cur)
        valid_mae.append(valid_mae_cur)
        
        plt.figure(figsize=(16,7))
        plt.subplot(1, 2, 1)
        plt.plot(np.ravel(np.array(train_targs))[:100], label='targs')
        plt.plot(np.ravel(np.array(train_preds))[:100], label='preds')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        epochs = np.arange(len(train_mae))
        plt.plot(epochs, train_mae, 'r', epochs, valid_mae, 'b')
        plt.legend(['train MAE','validation MAE'])
        plt.xlabel('epochs')
        plt.ylabel('MAE')
        
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()
        
        ## Store best Parameters
        if valid_mae_cur < lowest_valid_mae:
            lowest_valid_mae = valid_mae_cur
            best_model = deepcopy(net)
            best_val_targs = deepcopy(val_targs)
            best_val_preds = deepcopy(val_preds)
            best_train_targs = deepcopy(train_targs)
            best_train_preds = deepcopy(train_preds)
        
        scheduler.step()
        #if epoch%5==4 or epoch==num_epochs-1:
        print("Epoch %2i : Train MAE %f, Valid MAE %f" % (epoch+1, train_mae_cur, valid_mae_cur))
        #print("Epoch %2i : Train MAE %f" % (epoch+1, train_mae_cur))
    
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    os.makedirs('storage', exist_ok=True)
    
    plt.plot(np.ravel(train_targs), label='train_targs')
    plt.plot(np.ravel(train_preds), label='train_preds')
    plt.legend()
    plt.savefig(f'storage/train_{now}.png')
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
    
    all_info = (lowest_valid_mae, sequence_length, pred_sequence_length, hidden_size, linear_size, num_layers, num_epochs, batch_size, num_input_features, num_output_features)
    with open(f'storage/model_{now}.pickle', 'wb') as models_file:
        pickle.dump(best_model, models_file)
    with open(f'storage/info_{now}.pickle', 'wb') as info_file:
        pickle.dump(all_info, info_file)


if __name__ == '__main__':
    train()


'''
i = 0
    for img_feature in wind_dataset.img_features:
        idx = int(wind_dataset[i][img_feature])
        img_dataset = wind_dataset.img_datasets[img_feature]
        #print(img_handler[idx].img)
'''


'''if epoch==num_epochs-1:
                for h in range(len(X_batch)):
                    plt.scatter(np.arange(len(X_batch[h].numpy()[:,3])), X_batch[h].numpy()[:,3], label='X')
                    plt.scatter(len(X_batch[h].numpy()[:,3]), y_batch.data[h].numpy(), label='y_targ')
                    plt.scatter(len(X_batch[h].numpy()[:,3]), output.data[h].numpy(), label='y_pred')
                    plt.legend()
                    plt.show(block=False)
                    plt.pause(0.05)
                    plt.close()'''