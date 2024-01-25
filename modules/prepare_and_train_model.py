# 
# (1). Prepare the data (2). Specify the model by hyper-parameters (3). Apply the model on the data. (4). Save model and write down related training information.
import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import torch
from models_.lvmogp_preparation import BayesianGPLVM_
from models_.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from models_.kronecker_variational_strategy import KroneckerVariationalStrategy
from gpytorch.priors import NormalPrior
from models_.latent_variables import VariationalLatentVariable
from models_.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
import numpy as np
from linear_operator.operators import KroneckerProductLinearOperator
import yaml
from modules.training_module import train_the_model
from modules.prepare_data import *
# ------------------------------------------------------------------------------------------------------------------

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class LVMOGP_SVI(BayesianGPLVM_):

    def __init__(self, n_outputs, n_input, input_dim, latent_dim, n_inducing_input, n_inducing_latent, data_Y=None, pca=False, learn_inducing_locations_latent=True, learn_inducing_locations_input=True, latent_kernel_type='Scale_RBF', input_kernel_type='Scale_RBF'):

        self.n_outputs = n_outputs
        self.n_input = n_input
        self.inducing_inputs_latent = torch.randn(n_inducing_latent, latent_dim)
        self.inducing_inputs_input = torch.randn(n_inducing_input, input_dim)
        
        q_u = CholeskyKroneckerVariationalDistribution(n_inducing_input, n_inducing_latent)

        q_f = KroneckerVariationalStrategy(self, self.inducing_inputs_latent, self.inducing_inputs_input, q_u, learn_inducing_locations_latent=learn_inducing_locations_latent, learn_inducing_locations_input=learn_inducing_locations_input)

        # Define prior for latent
        latent_prior_mean = torch.zeros(n_outputs, latent_dim)  # shape: N x Q
        prior_latent = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
            assert data_Y.shape[0] == self.n_outputs
            assert data_Y.shape[1] == self.n_input
            latent_init = _init_pca(data_Y, latent_dim) # Initialise X to PCA 
        # TODO: how about training a GPLVM_SVI independent model for initialization ...
        else:
            latent_init = torch.nn.Parameter(torch.randn(n_outputs, latent_dim))
        
        # LatentVariable (c)
        latent_variables = VariationalLatentVariable(n_outputs, n_input, latent_dim, latent_init, prior_latent)

        super().__init__(latent_variables, q_f)

        self.mean_module = ZeroMean()

        # Kernel (acting on latent dimensions)
        if latent_kernel_type == 'Scale_RBF':
            self.covar_module_latent = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

        # Kernel (acting on index dimensions)
        if input_kernel_type == 'Scale_RBF':
            self.covar_module_input = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
        
    def _get_batch_idx(self, batch_size, sample_latent = True):
        if sample_latent == True:
            valid_indices = np.arange(self.n_outputs)
        else:
            valid_indices = np.arange(self.n_input)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        # return np.sort(batch_indices)
        return batch_indices
    
if __name__ == "__main__":

    #### Load hyperparameters from .yaml file
    with open('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/configs/train_lvmogp_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # specify random seed
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    #### Specify the dataset

    if config['dataset_type'] == 'synthetic_regression':
        data_inputs, data_Y_squeezed, idx_ls_of_ls, *arg = prepare_synthetic_regression_data(config)
    
    #### Define model and likelihood

    my_model = LVMOGP_SVI(
        n_outputs = config['n_outputs'],
        n_input = config['n_input_train'],
        input_dim = config['input_dim'],
        latent_dim = config['latent_dim'],
        n_inducing_input = config['n_inducing_input'],
        n_inducing_latent = config['n_inducing_latent'],
        pca = config['pca'],
        learn_inducing_locations_latent = config['learn_inducing_locations_latent'],
        learn_inducing_locations_input = config['learn_inducing_locations_input'],
        latent_kernel_type = config['latent_kernel_type'],
        input_kernel_type = config['input_kernel_type']
    )

    my_likelihood = GaussianLikelihood()

    #### Model Initialization ... 

    my_model.variational_strategy.inducing_points_input.data = Tensor(np.linspace(config['init_inducing_input_LB'], config['init_inducing_input_UB'], config['n_inducing_input']).reshape(-1, 1)) 
    my_likelihood.raw_noise.data = Tensor([config['init_likelihood_raw_noise']])

    #### Training the model

    total_time = train_the_model(
        data_Y_squeezed = data_Y_squeezed,
        data_inputs = data_inputs,
        idx_ls_of_ls = idx_ls_of_ls,
        my_model = my_model,
        my_likelihood = my_likelihood,
        config = config
    )

    print('total_time is: ', total_time)




