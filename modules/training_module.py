# Apply the model on the data.
# Assume data is given as dataframe, D rows (D outputs), N columns (N data points per output). The available data index (for training) is given as a list (D) of list (N_d). 
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
import time
from util_functions import *
from models_.variational_elbo import VariationalELBO

def train_the_model(data_Y_squeezed, data_inputs, idx_ls_of_ls, my_model, my_likelihood, config):

    '''
    Args:
        data_Y_squeezed: tensor, of length D *  N.
        data_inputs: tensor of inputs: length N. 
        idx_ls_of_ls: 
        my_model: instance of class LVMOGP_SVI
        my_likelihood: 
        config: 
    '''
    number_all = data_Y_squeezed.shape[0]
    my_mll = VariationalELBO(my_likelihood, my_model, num_data=number_all)

    # optimizer and scheduler
    optimizer = torch.optim.Adam([
        {'params': my_model.parameters()},
        {'params': my_likelihood.parameters()}
    ], lr=config['lr'])

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma']) 

    loss_list = []
    iterator = trange(config['n_iterations'], leave=True)

    my_model.train()
    my_likelihood.train()
    start_time = time.time()
    for i in iterator: 

        batch_index_latent, batch_index_input = sample_index_X_and_C_from_list(idx_ls_of_ls, batch_size_X = config['batch_size_latent'], batch_size_C = config['batch_size_input'])
        optimizer.zero_grad()
        
        ### computing loss = negative variational elbo = - (log_likelihood - kl_divergence - added_loss)
        loss_value = 0.0
        for _ in range(config['num_latent_MC']):
            sample_batch_X = my_model.sample_latent_variable(batch_index_latent)
            sample_batch_C = data_inputs[batch_index_input]
            output_batch = my_model(sample_batch_X, sample_batch_C) # q(f)
            batch_index_Y = inhomogeneous_index_of_batch_Y(batch_index_latent, batch_index_input, config['n_latent'], config['n_input'])
            loss = -my_mll(output_batch, data_Y_squeezed[batch_index_Y]).sum()
            loss_value += loss.item()
            loss.backward()
        
            '''
            ## log-likelihood term
            log_likelihood_batch = my_likelihood.expected_log_prob(input=output_batch, target=data_Y_squeezed[batch_index_Y]).sum(-1).div(output_batch.event_shape[0])
            loss += -log_likelihood_batch

            ## x_kl term
            added_loss = torch.zeros_like(log_likelihood_batch)
            for added_loss_term in my_model.added_loss_terms():
                # ONLY one added loss here, which is KL in latent space
                added_loss.add_(config['alpha'] * added_loss_term.loss())
            loss += added_loss

        ## KL divergence term
        kl_divergence = my_model.variational_strategy.kl_divergence().div(number_all / config['beta'])
        loss = loss / config['num_latent_MC'] + kl_divergence
        loss.backward()
            '''
        loss_value /= config['num_latent_MC']

        loss_list.append(loss_value)
        iterator.set_description('Loss: ' + str(float(np.round(loss_value, 3))) + ", iter no: " + str(i))

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config['model_max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(my_likelihood.parameters(), config['likeli_max_grad_norm'])

        optimizer.step()
        scheduler.step()

    end_time = time.time()
    total_training_time = end_time - start_time

    # plot training losses
    _loss_list = list(np.array(loss_list)[np.array(loss_list) < 3])
    plt.plot(_loss_list)
    plt.savefig(config['training_losses_figure_path'])

    # save model
    torch.save(my_model.state_dict(), config['model_path'])
    torch.save(my_likelihood.state_dict(), config['likelihood_path'])    

    return None