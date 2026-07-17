#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For a particular output length, Train ForecastNet models using different input lag sizes.
Created on Thu Mar 17 12:46:18 2022

@author: khaled
"""

# fixed randomization seed for reproducible results, if needed
import numpy as np
np.random.seed(25)
import torch
torch.manual_seed(25) # https://pytorch.org/docs/stable/notes/randomness.html
import random
random.seed(25)

import os
import pandas  as pd
from datetime import datetime
import time
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force execution in cpu, even if gpu is available

import sys
# sys.path.append(os.path.expanduser('~/Dropbox/Water/IterativeForecastV3/codes/libs')) # libs - utils, evals
# sys.path.append(os.path.expanduser('~/Dropbox/Water/IterativeForecastV3/codes/fnet/pytorch/'))

sys.path.append('/lscratch/s5084400/Water/IterativeForecastV3/codes/libs')
sys.path.append('/lscratch/s5084400/Water/IterativeForecastV3/codes/fnet/pytorch/')

from forecastNet import forecastNet
from train import train
from evaluate import evaluate, get_prediction_only
from dataHelpers import *
from calculateError_custom import *

import util # custom functions

#model_type = 'dense' #'dense2' or 'conv', or 'conv2'
model_type = 'conv2'


#%% Globals
dataset_name = 'SB17'
target_series_name = 'do'
ds_name = 'SbDO'  # short name to represent the data series in result analysis
forecast_length = 48
# outseq_length = 12   # number of timesteps to be predicted (part of actual forecasting horizon to be predicted)
outseq_length = forecast_length


#%% Data Paths

# data_dir = os.path.expanduser('~/Water/IterativeForecastV3')
data_dir = '/lscratch/s5084400/Water/IterativeForecastV3'
# data_dir = '/lscratch/s5084397/Water/IterativeForecastV3'

models_dir = os.path.join(data_dir, 'models_basetuning', dataset_name, target_series_name, 'fnet_'+ model_type+'_models', 'model_outws'+str(outseq_length))
if not os.path.exists(models_dir): os.makedirs(models_dir)


TRAIN_FILENAME = 'trainset.tsv'
TEST_FILENAME = 'testset.tsv'


print('fNET run starting:  ' + model_type + '  ' + dataset_name + ' - ' + target_series_name)
#%% Main
train_set = pd.read_csv(os.path.join(data_dir, 'datasets', dataset_name, target_series_name, TRAIN_FILENAME), sep='\t', parse_dates=['datetime'], infer_datetime_format=True, index_col=['datetime'], low_memory=False,)
test_set = pd.read_csv(os.path.join(data_dir, 'datasets', dataset_name, target_series_name, TEST_FILENAME), sep='\t', parse_dates=['datetime'], infer_datetime_format=True, index_col=['datetime'], low_memory=False,)


dataset = pd.concat([train_set, test_set])[target_series_name].to_numpy().reshape(-1, 1) # making univariate time series
test_start_idx = dataset.shape[0] - len(test_set) # starting index of test set in combine dataset

data_scale = {'data_max' : np.max(dataset), 'data_min' : np.min(dataset)} # to be used in data scaling and reverse data scaling

# normalize data - scale in range [0,1]
dataset = util.scale_dataset(data_scale,dataset) # dataset: entire univariate time series - train & test set
test_set = util.scale_dataset(data_scale,test_set[target_series_name].to_numpy()) # entire test set: single vector, not sliding windowed.
train_set = util.scale_dataset(data_scale,train_set[target_series_name].to_numpy())


# For each potential inseq-length,  prepare multistepped input-ouput samples from the data series using sliding windowing, train model on these windowed samples 
# and check prediction accuracy on train, test and validation set.

candidate_inseq_lengths = [outseq_length * 3, outseq_length * 2, outseq_length] # explore suitable lag input-size in this search space for particular outseq_length


for inseq_length in candidate_inseq_lengths:
    
    avg_error_scores = pd.DataFrame() # separate error file for each inseq length
    
    samples_df = util.create_samples_from_timeseries(dataset, inseq_length, outseq_length)
    
    x_train, y_train, x_test, y_test, x_valid, y_valid = util.divide_samples_in_predefined_train_test(samples_df, test_start_idx, get_validation = True)
    
    # Format data for ForecastNet (i.e. make Time major in 3d: [timesteps, n_batch, input_dim])
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
    y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    
    x_train = np.transpose(x_train, (1, 0, 2))
    y_train = np.transpose(y_train, (1, 0, 2))
    x_valid = np.transpose(x_valid, (1, 0, 2))
    y_valid = np.transpose(y_valid, (1, 0, 2))
    x_test = np.transpose(x_test, (1, 0, 2))
    y_test = np.transpose(y_test, (1, 0, 2))
    
    
    # build and train ForecastNet model
    
    # Model parameters
    #model_type = 'dense2' #'dense' or 'conv', 'dense2' or 'conv2'
    hidden_dim = 24
    input_dim = 1
    output_dim = 1
    learning_rate = 0.0001 # tunned in 10^-i (i=2,...,6)
    n_epochs = 100 #200
    batch_size = 16
    
    
    # Initialise model
    
    model_file = os.path.join(models_dir, ds_name + '_fnet_'+ model_type + '_inws' + str(inseq_length) + '_outws' + str(outseq_length) + '.pt')
    
    fcstnet = forecastNet(in_seq_length=inseq_length, out_seq_length=outseq_length, input_dim=input_dim,
                            hidden_dim=hidden_dim, output_dim=output_dim, model_type = model_type, batch_size = batch_size,
                            n_epochs = n_epochs, learning_rate = learning_rate, save_file = model_file)
    
    # train
    start_time = time.time()
    training_costs, validation_costs = train(fcstnet, x_train, y_train, x_valid, y_valid, restore_session=False)
    end_time = time.time()
    training_time = end_time-start_time
    
    print('Time (minutes) taken to model training: ', training_time/60)
    
    # save training loss and validation loss
    train_losses = pd.DataFrame({'train_loss': np.array(training_costs), 'val_loss': np.array(validation_costs)})
    train_losses.to_csv(os.path.join(models_dir, ds_name + '_fnet_'+ model_type + '_inws' + str(inseq_length) + '_outws' + str(outseq_length) + '_training_losses.tsv'), sep='\t', index=False, float_format='%.7f')
    
    
    
    #%% Evaluate the model on Training, Test and validation set
    
    ## On Train set
    
    y_pred_train = get_prediction_only(fcstnet, x_train, y_train)
    
    # forecastnet's default input-outputs shape [out_seq_length, n_samples, n_features=1]
    # reshape ground truth and forecastnet's predictions in shape [n_samples, out_seq_length] for easier metric calculation
    
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    y_train = np.transpose(y_train, (1, 0))
    y_pred_train = y_pred_train.reshape(y_pred_train.shape[0], y_pred_train.shape[1])
    y_pred_train = np.transpose(y_pred_train, (1, 0))
    
    
    mase_train, smape_train, _ = calculate_error_stepaheadwise(y_train, y_pred_train) #metric value for each step ahead
    mae_train, rmse_train, mape_train = calculate_scale_dependent_error(y_train, y_pred_train, data_scale) # these metrics not dependent on samplewise or stepwise
    

    avg_error_scores = avg_error_scores.append({
                            'error_on': 'train',
                            'train_time' : training_time,
                            'mae' : np.mean(mae_train),
                            'rmse' : np.mean(rmse_train),
                            'mape' : np.mean(mape_train),
                            'mase' : np.mean(mase_train),
                            'smape' : np.mean(smape_train),
                            }, ignore_index = True)
    
    ## On Test set
    
    y_pred_test = get_prediction_only(fcstnet, x_test, y_test)
    
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
    y_test = np.transpose(y_test, (1, 0))
    y_pred_test = y_pred_test.reshape(y_pred_test.shape[0], y_pred_test.shape[1])
    y_pred_test = np.transpose(y_pred_test, (1, 0))
    
    
    mase_test, smape_test, _ = calculate_error_stepaheadwise(y_test, y_pred_test) #metric value for each step ahead
    mae_test, rmse_test, mape_test = calculate_scale_dependent_error(y_test, y_pred_test, data_scale) # these metrics not dependent on samplewise or stepwise
    

    avg_error_scores = avg_error_scores.append({
                            'error_on': 'test',                    
                            'mae' : np.mean(mae_test),
                            'rmse' : np.mean(rmse_test),
                            'mape' : np.mean(mape_test),
                            'mase' : np.mean(mase_test),
                            'smape' : np.mean(smape_test),
                            }, ignore_index = True)
    
    ## On validation set
    
    y_pred_val = get_prediction_only(fcstnet, x_valid, y_valid)

    y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1])
    y_valid = np.transpose(y_valid, (1, 0))
    y_pred_val = y_pred_val.reshape(y_pred_val.shape[0], y_pred_val.shape[1])
    y_pred_val = np.transpose(y_pred_val, (1, 0))
    
    
    mase_val, smape_val, _ = calculate_error_stepaheadwise(y_valid, y_pred_val) #metric value for each step ahead
    mae_val, rmse_val, mape_val = calculate_scale_dependent_error(y_valid, y_pred_val, data_scale) # these metrics not dependent on samplewise or stepwise
    

    avg_error_scores = avg_error_scores.append({
                            'error_on': 'valid',
                            'mae' : np.mean(mae_val),
                            'rmse' : np.mean(rmse_val),
                            'mape' : np.mean(mape_val),
                            'mase' : np.mean(mase_val),
                            'smape' : np.mean(smape_val),
                            }, ignore_index = True)

    avg_error_scores.to_csv(os.path.join(models_dir, ds_name + '_fnet_'+ model_type + '_inws' + str(inseq_length) + '_outws' + str(outseq_length) + '_train_time_errors.tsv'), sep='\t', index=False, float_format='%.5f')
    print(f'Training complete for outseq_length: {outseq_length}, inseq_length: {inseq_length}')

print('fNET run complete:  ' + ds_name + '  ' + model_type + '  ' + '_inws' + str(inseq_length) + '_outws' + str(outseq_length))
