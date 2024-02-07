import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
import torch
import yaml
import numpy as np
from modules.prepare_data import *
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from models_.gaussian_likelihood import GaussianLikelihood
from modules.training_IGP import train_multi_IGP
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, MaternKernel

class Variational_GP(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='Scale_RBF', learn_inducing_locations=True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations)
        super(Variational_GP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == 'Scale_RBF':
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel_type == 'Scale_Periodic_times_Scale_RBF':
            self.covar_module = ScaleKernel(PeriodicKernel()) * ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class Multi_Variational_IGP:
    '''each output corresponds to an independent model'''
    def __init__(self, num_models, inducing_points, init_likelihood_raw_noise, kernel_type='Scale_RBF', learn_inducing_locations='True'):
        self.models = [Variational_GP(inducing_points, kernel_type, learn_inducing_locations) for _ in range(num_models)]
        self.likelihoods = [GaussianLikelihood() for _ in range(num_models)]

        for likelihood in self.likelihoods:
            likelihood.raw_noise.data = Tensor([init_likelihood_raw_noise])

    def get_model(self, model_number):
        if 0 <= model_number <= len(self.models) - 1:
            return self.models[model_number]
        else:
            raise ValueError(f"Model number must be between 1 and {len(self.models)}")
    
    def get_likelihood(self, likelihood_number):
        if 0 <= likelihood_number <= len(self.likelihoods) - 1:
            return self.likelihoods[likelihood_number]
        else:
            raise ValueError(f"Likelihood number must be between 1 and {len(self.likelihoods)}")


if __name__ == "__main__":
    #### Load hyperparameters from .yaml file
    with open('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/configs/spatiotemp_IGP_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # specify random seed
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    #### Specify the dataset
    if config['dataset_type'] == 'synthetic_regression':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, train_sample_idx_ls, test_sample_idx_ls = prepare_synthetic_regression_data(config)
    elif config['dataset_type'] == 'spatio_temporal_data':
        data_inputs, data_Y_squeezed, ls_of_ls_train_input, ls_of_ls_test_input, _, train_sample_idx_ls, test_sample_idx_ls = prepare_spatio_temp_data(config)
    
    #### Define model and likelihood
    #### Initialization is also done
    MultiIGP = Multi_Variational_IGP(
        num_models = config['n_outputs'],
        inducing_points = Tensor(np.linspace(config['init_inducing_input_LB'], config['init_inducing_input_UB'], config['n_inducing_input']).reshape(-1, 1)), 
        init_likelihood_raw_noise = config['init_likelihood_raw_noise'],
        kernel_type = config['kernel_type'],
        learn_inducing_locations = config["learn_inducing_locations"]
    )

    for i in range(config['n_outputs']):
        MultiIGP.get_model(i).covar_module.kernels[1].base_kernel.raw_lengthscale.data = torch.tensor([[config['2thKernel_raw_lengthscale_init']]])

    total_time = train_multi_IGP(
        data_inputs = data_inputs, 
        data_Y_squeezed = data_Y_squeezed,
        ls_of_ls_train_input = ls_of_ls_train_input, 
        ls_of_ls_test_input = ls_of_ls_test_input,
        train_sample_idx_ls = train_sample_idx_ls,
        test_sample_idx_ls = test_sample_idx_ls,
        my_models_and_likelihoods = MultiIGP,
        config = config
    )

    # print('total_time is: ', total_time)
    
    