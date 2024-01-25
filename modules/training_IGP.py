import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import torch
from torch.optim.lr_scheduler import StepLR
from models_.variational_elbo import VariationalELBO
from tqdm import trange
import time
import numpy as np
import pandas as pd
import random

def mini_batching_sampling_func(num_inputs, batch_size):
    assert batch_size <= num_inputs
    idx_list = random.sample(range(num_inputs), batch_size)
    return idx_list


def save_model_and_likelihoods(multi_variational_igp, filename):
    state_dicts = {
        'models': [model.state_dict() for model in multi_variational_igp.models],
        'likelihoods': [likelihood.state_dict() for likelihood in multi_variational_igp.likelihoods]
    }
    torch.save(state_dicts, filename)


def train_multi_IGP(data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls, my_models_and_likelihoods, config):
    '''
    First 6 arguments are returned from functions in prepare_data.py, these data are naturally designed for MOGP.
    To train multiple IGPs, we need to reconstruct datasets based on them.
    Return:
        list of training times (for all outputs). 
    '''
    # The following lists consist of datasets for all outputs.
    list_train_X, list_train_Y = [], [] 
    list_test_X, list_test_Y = [], []

    # split data_Y_squeezed into train/test part. NOTE: that's train/test target data for all outputs.
    data_Y_train_squeezed = data_Y_squeezed[train_sample_idx_ls]
    data_Y_test_squeezed = data_Y_squeezed[test_sample_idx_ls]

    n_input_test = config['n_input'] - config['n_input_train']
    ##### ------------------------------------------------------------------------
    for i in range(config['n_outputs']):
        # start and end for current output, idx used to pick data for only current output
        idgp_train_start = i * config['n_input_train']
        idgp_train_end = idgp_train_start + config['n_input_train']

        idgp_test_start = i * n_input_test
        idgp_test_end = idgp_test_start + n_input_test

        # training data for current output
        train_X = data_inputs[ls_of_ls_train_input[i]]
        train_Y = data_Y_train_squeezed[idgp_train_start:idgp_train_end]
        assert train_X.shape ==  train_Y.shape == torch.Size([config['n_input_train']])
        list_train_X.append(train_X)
        list_train_Y.append(train_Y)

        # testing data for current output
        test_X = data_inputs[ls_of_ls_test_input[i]]
        test_Y = data_Y_test_squeezed[idgp_test_start:idgp_test_end]
        assert test_X.shape ==  test_Y.shape == torch.Size([n_input_test])
        list_test_X.append(test_X)
        list_test_Y.append(test_Y)

    ##### ------------------------------------------------------------------------
    if config['store_all_loss'] == True:
        all_train_loss = np.zeros((config['n_outputs'], config['n_iterations'])) 

    ls_training_time = []
    ##### Train IGPs one by one
    for j in range(config['n_outputs']):

        # current train_X and train_Y
        train_X = list_train_X[j]
        train_Y = list_train_Y[j]

        curr_model = my_models_and_likelihoods.get_model(j)
        curr_likelihood = my_models_and_likelihoods.get_likelihood(j)

        curr_model.train()
        curr_likelihood.train()
        
        curr_optimizer = torch.optim.Adam([
            {'params': curr_model.parameters()},
            {'params': curr_likelihood.parameters()},
        ], lr=config['lr'])

        curr_scheduler = StepLR(curr_optimizer, step_size=config['step_size'], gamma=config['gamma'])

        mll = VariationalELBO(curr_likelihood, curr_model, num_data=train_Y.size(0))

        # start training!
        ls_train_loss = []
        iterator = trange(config['n_iterations'], leave=True)

        curr_start_time = time.time()
        for i in iterator:
            curr_optimizer.zero_grad()
            mini_batch_idx = mini_batching_sampling_func(num_inputs=train_X.shape[0], batch_size=config['batch_size_input'])
            output_pred = curr_model(train_X[mini_batch_idx])
            loss = -mll(output_pred, train_Y[mini_batch_idx])
            ls_train_loss.append(loss.item())
            iterator.set_description( 'Training '+ str(j) + 'th Model; '+ 'Loss: ' + str(float(np.round(loss.item(),3))) + ", iter no: " + str(i))
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(curr_model.parameters(), config['model_max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(curr_likelihood.parameters(), config['likeli_max_grad_norm'])

            curr_optimizer.step()
            curr_scheduler.step()

        curr_end_time = time.time()
        curr_total_training_time = curr_end_time - curr_start_time

        if config['store_all_loss'] == True:
            all_train_loss[j,:] = np.array(ls_train_loss)

        ls_training_time.append(curr_total_training_time)

    save_model_and_likelihoods(my_models_and_likelihoods, config['model_and_likelihood_path'])
    pd.DataFrame(all_train_loss).to_csv(config['all_training_loss_file'])
    return ls_training_time