import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import numpy as np
import random
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Tensor
from torch.distributions import MultivariateNormal
from models_.lvmogp_svi import LVMOGP_SVI
from models_.gaussian_likelihood import GaussianLikelihood
from models_.variational_elbo import VariationalELBO
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
from util_functions import *
import yaml
import time

expri_random_seed =  12  # 13, 78, 912, 73, 269

# Double Check this with data folder title! Make sure import the correct one.
w_n_C_total = 50 # totally 700 points for C
w_n_outputs = 50 # 100, 300, 500, 1000, 2500(20), 1500, 2000

synth_data_path = f'/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/data/synth_regression/smartly_generated/ninputs_{w_n_C_total}_nlatents_{w_n_outputs}'
w_C_total = Tensor(pd.read_csv(f'{synth_data_path}/inputs.csv').to_numpy()).reshape(-1)
w_X_true = Tensor(pd.read_csv(f'{synth_data_path}/latents.csv').to_numpy()).reshape(-1, 2)
w_sample_total_data = Tensor(pd.read_csv(f'{synth_data_path}/target_data.csv').to_numpy()).reshape(-1)

w_n_C_train = 25 # the number of training data points per output
w_n_C_test = w_n_C_total - w_n_C_train

np.random.seed(expri_random_seed)
torch.manual_seed(expri_random_seed)
list_expri_random_seeds = np.random.randn(w_n_outputs)

# different from the previous case, C_train and C_test no longer a single set, but every output has different values.
w_ls_of_ls_train_C = []
w_ls_of_ls_test_C = []

w_sample_train_index, w_sample_test_index = [], []

for i in range(w_n_outputs):
    # iterate across different output functions
    random.seed(list_expri_random_seeds[i])
    train_index = random.sample(range(w_n_C_total), w_n_C_train)
    test_index = [index for index in range(w_n_C_total) if index not in train_index]
    w_ls_of_ls_train_C.append(train_index)
    w_ls_of_ls_test_C.append(test_index)

    w_sample_train_index = np.concatenate((w_sample_train_index, list(np.array(train_index) + w_n_C_total*i)))
    w_sample_test_index = np.concatenate((w_sample_test_index, list(np.array(test_index) + w_n_C_total*i)))

w_sample_train_data = w_sample_total_data[w_sample_train_index]
w_sample_test_data = w_sample_total_data[w_sample_test_index]

assert w_sample_train_data.shape[0] == w_n_C_train * w_n_outputs
assert w_sample_test_data.shape[0] == w_n_C_test * w_n_outputs

def only_train_variational_params_fix_others(true_hyperparams, my_model, my_likelihood):
    
    # assign true values to model hyper-parameters
    my_model.covar_module_latent.raw_outputscale.data = torch.tensor(true_hyperparams['X_raw_outputscale'])
    my_model.covar_module_input.raw_outputscale.data = torch.tensor(true_hyperparams['C_raw_outputscale'])
    my_model.covar_module_latent.base_kernel.raw_lengthscale.data = torch.tensor([true_hyperparams['X_raw_lengthscale']])
    my_model.covar_module_input.base_kernel.raw_lengthscale.data = torch.tensor([true_hyperparams['C_raw_lengthscale']])
    my_likelihood.noise = torch.tensor(true_hyperparams['likelihood_noise']) # NOTE: not .data !

    # fix gradient updates for hyperparameters
    my_model.covar_module_latent.raw_outputscale.requires_grad = False
    my_model.covar_module_input.raw_outputscale.requires_grad = False
    my_model.covar_module_latent.base_kernel.raw_lengthscale.requires_grad = False
    my_model.covar_module_input.base_kernel.raw_lengthscale.requires_grad = False
    my_likelihood.raw_noise.requires_grad = False

# define hyper-parameters
w_n_X = w_X_true.shape[0]
w_n_C = len(w_ls_of_ls_train_C[0])
w_n_total = w_n_X * w_n_C
w_index_dim = 1
w_latent_dim = 2
w_n_inducing_C = 30
w_n_inducing_X = 30
w_pca = False
learn_inducing_locations_X= True # True
learn_inducing_locations_C = True

Y_train = w_sample_train_data

# specify model
w_my_model = LVMOGP_SVI(w_n_X, w_n_C, w_index_dim, w_latent_dim, w_n_inducing_C, w_n_inducing_X, Y_train.reshape(w_n_X, -1), pca=w_pca, learn_inducing_locations_latent=learn_inducing_locations_X, learn_inducing_locations_input=learn_inducing_locations_C)

# Likelihood & training objective
w_likelihood = GaussianLikelihood()
w_mll = VariationalELBO(w_likelihood, w_my_model, num_data=w_n_total)

import json
with open(f'{synth_data_path}/dictionary.json', 'r') as file:
    true_hyperparams = json.load(file)
true_hyperparams['likelihood_noise'] = 0.05

# only_train_variational_params_fix_others(true_hyperparams=true_hyperparams, my_model=w_my_model, my_likelihood=w_likelihood)

# optimizer and scheduler
w_optimizer = torch.optim.Adam([
    {'params': w_my_model.parameters()},
    {'params': w_likelihood.parameters()}
], lr=0.1)

w_scheduler = StepLR(w_optimizer, step_size=20, gamma=0.95)  # every 50 iterations, learning rate multiple 0.95

# Initialize inducing points in C space
w_my_model.variational_strategy.inducing_points_input.data = Tensor(np.linspace(-10, 10, w_n_inducing_C).reshape(-1, 1))
# Another initialization: random initialization
# i.e. torch.rand(w_n_inducing_C).reshape(-1,1) * 20 - 10

# Initialize inducing points in latent space
# w_my_model.variational_strategy.inducing_points_X.data = 3 * torch.randn(w_n_inducing_X, w_latent_dim)

# Initialize likelihood noise as true value, 0.05
w_likelihood.raw_noise.data = Tensor([-2.973])
# w_likelihood.raw_noise.requires_grad = False

# start training!
w_loss_list = []
n_iterations = 1000 # 5000 # 10000
iterator = trange(n_iterations, leave=True)
batch_size_X = 50 # mini-batch for latents
batch_size_C = 20 # mini-batch for inputs, one can set w_n_C_train
num_X_MC = 1 # the number of MC samples used to approximate E_{q(X)}
w_model_max_grad_norm = 1
w_likeli_max_grad_norm = 0.1

'''
for name, params in w_my_model.named_parameters():
    print(name)
for name, params in w_likelihood.named_parameters():
    print(name)
'''

w_my_model.train()
w_likelihood.train()
start_time = time.time()
for i in iterator: 
    batch_index_X, batch_index_C = sample_index_X_and_C_from_list(w_ls_of_ls_train_C, batch_size_X=batch_size_X, batch_size_C=batch_size_C)
    # core code is here 
    w_optimizer.zero_grad()

    loss_value = 0.0
    for _ in range(num_X_MC):
        sample_batch_X = w_my_model.sample_latent_variable(batch_index_X)
        sample_batch_C = w_C_total[batch_index_C]
        output_batch = w_my_model(sample_batch_X, sample_batch_C) # q(f)
        batch_index_Y = inhomogeneous_index_of_batch_Y(batch_index_X, batch_index_C, w_n_X, w_n_C_total)
        # print('batch_index_Y', len(batch_index_Y))
        loss = -w_mll(output_batch, w_sample_total_data[batch_index_Y]).sum()
        loss_value += loss.item()
        loss.backward()

    loss_value /= num_X_MC
    
    w_loss_list.append(loss_value)
    iterator.set_description('Loss: ' + str(float(np.round(loss_value, 3))) + ", iter no: " + str(i))
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(w_my_model.parameters(), w_model_max_grad_norm)
    torch.nn.utils.clip_grad_norm_(w_likelihood.parameters(), w_likeli_max_grad_norm)

    w_optimizer.step()
    w_scheduler.step()

end_time = time.time()
print('Total Training Time:',  end_time - start_time)

torch.save(w_my_model.state_dict(), '/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/experi_results/model_weight.pth')
torch.save(w_likelihood.state_dict(), '/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/experi_results/likelihood_weight.pth')  