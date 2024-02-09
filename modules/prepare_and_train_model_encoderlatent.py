################################################################################################################################################################################
##  This file is for modelling with latent variables as output of neural network.


################################################################################################################################################################################
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import torch
from models_.lvmogp_preparation import BayesianGPLVM_
from models_.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from models_.kronecker_variational_strategy import KroneckerVariationalStrategy
from gpytorch.priors import NormalPrior
from models_.latent_variables import NNEncoderLatentVariable
from models_.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, MaternKernel
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
import numpy as np
from linear_operator.operators import KroneckerProductLinearOperator
import yaml
from modules.training_module import train_the_model
from modules.prepare_data import *
import random
from util_functions import *
# ------------------------------------------------------------------------------------------------------------------

class LVMOGP_SVI(BayesianGPLVM_):

    def __init__(self, n_outputs, 
                 n_input, 
                 input_dim, 
                 latent_dim,
                 n_inducing_input, 
                 n_inducing_latent, 
                 learn_inducing_locations_latent=True, 
                 learn_inducing_locations_input=True, 
                 latent_kernel_type='Scale_RBF', 
                 input_kernel_type='Scale_RBF'):

        # NOTE:
        # PAY ATTENTION: n_input shoud be number of training data input. NOT total input.
        self.n_outputs = n_outputs
        self.n_input = n_input
        self.inducing_inputs_latent = torch.randn(n_inducing_latent, latent_dim)
        self.inducing_inputs_input = torch.randn(n_inducing_input, input_dim)
        
        q_u = CholeskyKroneckerVariationalDistribution(n_inducing_input, n_inducing_latent)

        q_f = KroneckerVariationalStrategy(self, self.inducing_inputs_latent, self.inducing_inputs_input, q_u, 
                                           learn_inducing_locations_latent=learn_inducing_locations_latent, 
                                           learn_inducing_locations_input=learn_inducing_locations_input)

        # Define prior for latent
        latent_prior_mean = torch.zeros(n_outputs, latent_dim)  # shape: N x Q
        prior_latent = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean))

        # LatentVariable (c)
        latent_variables = NNEncoderLatentVariable(
                    n=n_outputs, 
                    data_dim=n_input, 
                    latent_dim=latent_dim, 
                    prior_x=prior_latent, 
                    latent_info_dim=2, 
                    layers=[4, 8, 4]
        )
        super().__init__(latent_variables, q_f)
        self.mean_module = ZeroMean()

        # Kernel (acting on latent dimensions)
        # Scale_RBF is the default choice, as prediction via integration of latent variable is possible.
        if latent_kernel_type == 'Scale_RBF':
            self.covar_module_latent = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

        #### ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
        # Kernel (acting on index dimensions)
        self.covar_module_input = helper_specify_kernel_by_name(input_kernel_type, input_dim)

    def _get_batch_idx(self, batch_size, sample_latent = True):
        if sample_latent == True:
            valid_indices = np.arange(self.n_outputs)
        else:
            valid_indices = np.arange(self.n_input)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return batch_indices

if __name__ == "__main__":

    #### Load hyperparameters from .yaml file

    root_config = '/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/configs/'
    with open(f'{root_config}/spatiotemp_lvmogp_encoderlatent_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    #### specify random seed
        
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    #### Specify the dataset

    if config['dataset_type'] == 'synthetic_regression':
        data_inputs, data_Y_squeezed, idx_ls_of_ls, *arg = prepare_synthetic_regression_data(config)
    
    elif config['dataset_type'] == 'spatio_temporal_data':
        # assert config['latent_dim'] == 2
        data_inputs, data_Y_squeezed, idx_ls_of_ls, _, lon_lat_tensor, *arg = prepare_spatio_temp_data(config)
    
    #### Define model and likelihood

    my_model = LVMOGP_SVI(
        n_outputs = config['n_outputs'],
        n_input = config['n_input_train'], # NOTE PAY ATTENTION, not total n_inputs.
        input_dim = config['input_dim'],
        latent_dim = config['latent_dim'],
        n_inducing_input = config['n_inducing_input'],
        n_inducing_latent = config['n_inducing_latent'],
        learn_inducing_locations_latent = config['learn_inducing_locations_latent'],
        learn_inducing_locations_input = config['learn_inducing_locations_input'],
        latent_kernel_type = config['latent_kernel_type'],
        input_kernel_type = config['input_kernel_type']
    )

    my_likelihood = GaussianLikelihood()

    #### Model Initialization (after instantiation) ... 

    my_model, my_likelihood = helper_init_model_and_likeli(my_model, my_likelihood, config)

    #### Training the model ... 

    total_time = train_the_model(
        data_Y_squeezed = data_Y_squeezed,
        data_inputs = data_inputs,
        idx_ls_of_ls = idx_ls_of_ls,
        my_model = my_model,
        my_likelihood = my_likelihood,
        config = config,
        latent_info = lon_lat_tensor
    )

    print('total_time is: ', total_time)