import torch
from models_.lvmogp_preparation import BayesianGPLVM_
from models_.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from models_.kronecker_variational_strategy import KroneckerVariationalStrategy
from gpytorch.priors import NormalPrior
# from gpytorch.models.gplvm.latent_variable import *
from models_.latent_variables import VariationalLatentVariable
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
import numpy as np
from linear_operator.operators import KroneckerProductLinearOperator



def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class LVMOGP_SVI(BayesianGPLVM_):

    def __init__(self, n_latent, n_input, input_dim, latent_dim, n_inducing_input, n_inducing_latent, data_Y, pca=False, learn_inducing_locations_latent=True, learn_inducing_locations_input=True):

        self.n_latent = n_latent
        self.n_input = n_input
        self.inducing_inputs_latent = torch.randn(n_inducing_latent, latent_dim)
        self.inducing_inputs_input = torch.randn(n_inducing_input, input_dim)
        
        q_u = CholeskyKroneckerVariationalDistribution(n_inducing_input, n_inducing_latent)

        q_f = KroneckerVariationalStrategy(self, self.inducing_inputs_latent, self.inducing_inputs_input, q_u, learn_inducing_locations_latent=learn_inducing_locations_latent, learn_inducing_locations_input=learn_inducing_locations_input)

        # Define prior for latent
        latent_prior_mean = torch.zeros(n_latent, latent_dim)  # shape: N x Q
        prior_latent = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
            assert data_Y.shape[0] == self.n_latent
            assert data_Y.shape[1] == self.n_input
            latent_init = _init_pca(data_Y, latent_dim) # Initialise X to PCA 
        # TODO: how about training a GPLVM_SVI independent model for initialization ...
        else:
            latent_init = torch.nn.Parameter(torch.randn(n_latent, latent_dim))
        
        # LatentVariable (c)
        latent_variables = VariationalLatentVariable(n_latent, n_input, latent_dim, latent_init, prior_latent)

        super().__init__(latent_variables, q_f)

        self.mean_module = ZeroMean()

        # Kernel (acting on latent dimensions)
        self.covar_module_latent = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        # Kernel (acting on index dimensions)
        self.covar_module_input = ScaleKernel(RBFKernel(ard_num_dims=input_dim))

    def _get_batch_idx(self, batch_size, sample_latent = True):
        if sample_latent == True:
            valid_indices = np.arange(self.n_latent)
        else:
            valid_indices = np.arange(self.n_input)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        # return np.sort(batch_indices)
        return batch_indices
