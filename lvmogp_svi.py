import torch
from lvmogp_preparation import BayesianGPLVM_
from cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from kronecker_variational_strategy import KroneckerVariationalStrategy
from gpytorch.priors import NormalPrior
# from gpytorch.models.gplvm.latent_variable import *
from latent_variables import VariationalLatentVariable
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

    def __init__(self, n_X, n_C, index_dim, latent_dim, n_inducing_C, n_inducing_X, data_Y, pca=False):
        self.n_X = n_X
        self.n_C = n_C
        self.inducing_inputs_X = torch.randn(n_inducing_X, latent_dim)
        self.inducing_inputs_C = torch.randn(n_inducing_C, index_dim)
        
        q_u = CholeskyKroneckerVariationalDistribution(n_inducing_C, n_inducing_X)

        q_f = KroneckerVariationalStrategy(self, self.inducing_inputs_X, self.inducing_inputs_C, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n_X, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
            assert data_Y.shape[0] == self.n_X
            assert data_Y.shape[1] == self.n_C
            X_init = _init_pca(data_Y, latent_dim) # Initialise X to PCA 
        # TODO: how about training a GPLVM_SVI independent model for initialization ...
        else:
            X_init = torch.nn.Parameter(torch.randn(n_X, latent_dim))
        
        # LatentVariable (c)
        X = VariationalLatentVariable(n_X, n_C, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        self.mean_module_X = ZeroMean()

        # Kernel (acting on latent dimensions)
        self.covar_module_X = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        # Kernel (acting on index dimensions)
        self.covar_module_C = ScaleKernel(RBFKernel(ard_num_dims=index_dim))
    
    def forward(self, X, C, jitter_val=1e-4):
        n_total = int(X.shape[-2] * C.shape[-2])
        # This implementation ONLY works for ZeroMean()
        mean_x = self.mean_module_X(Tensor([i for i in range(n_total)])) 
        covar_x = KroneckerProductLinearOperator(self.covar_module_X(X), self.covar_module_C(C)).to_dense() 
        # for numerical stability
        covar_x += torch.eye(covar_x.size(0)) * jitter_val
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def _get_batch_idx(self, batch_size, sample_X = True):
        if sample_X == True:
            valid_indices = np.arange(self.n_X)
        else:
            valid_indices = np.arange(self.n_C)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        # return np.sort(batch_indices)
        return batch_indices
