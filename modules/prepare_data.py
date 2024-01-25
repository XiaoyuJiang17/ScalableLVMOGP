# This piece of code is used to store all the functions for preparing dataset.
import numpy as np
import torch
from torch import Tensor
import pandas as pd
import random

def prepare_synthetic_regression_data(config):
    '''
    Prepare data
    Args:
        config: containing all information used to construct the data (ready for model training)
    Return:
        data_inputs: Tensor, of shape n_inputs
        data_Y_squeezed: Tensor, of shape n_input * n_outputs
        ls_of_ls_train_input: list (list of list), n_outputs outer list, n_input_train inner list
        ls_of_ls_test_input: list (list of list), n_outputs outer list, n_input_test inner list
        train_sample_idx_ls: np.array, of shape (n_input_train * n_outputs, )
        test_sample_idx_ls: np.array, of shape (n_input_test * n_outputs, )
    '''

    data_Y_squeezed = Tensor(pd.read_csv(config['data_Y_squeezed_path']).to_numpy()).reshape(-1)
    data_inputs = Tensor(pd.read_csv(config['data_inputs_path']).to_numpy()).reshape(-1) 
    assert data_inputs.shape[0] == config['n_input']
    assert data_Y_squeezed.shape[0] == (config['n_input'] * config['n_outputs'])

    np.random.seed(config['random_seed'])
    list_expri_random_seeds = np.random.randn(config['n_outputs'])

    ls_of_ls_train_input = []
    ls_of_ls_test_input = []
    
    train_sample_idx_ls, test_sample_idx_ls = [], []

    for i in range(config['n_outputs']):
        # iterate across different output functions
        random.seed(list_expri_random_seeds[i])
        train_index = random.sample(range(config['n_input']), config['n_input_train'])
        test_index = [index for index in range(config['n_input']) if index not in train_index]
        ls_of_ls_train_input.append(train_index)
        ls_of_ls_test_input.append(test_index)

        train_sample_idx_ls = np.concatenate((train_sample_idx_ls, list(np.array(train_index) + config['n_input']*i)))
        test_sample_idx_ls = np.concatenate((test_sample_idx_ls, list(np.array(test_index) + config['n_input']*i)))

    return data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls