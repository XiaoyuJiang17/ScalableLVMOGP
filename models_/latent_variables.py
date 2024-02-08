import torch
import numpy as np
from gpytorch.module import Module
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm


class LatentVariable(Module):
    """
    This super class is used to describe the type of inference
    used for the latent variable :math:`\\mathbf X` in GPLVM models.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim

    def forward(self, x):
        raise NotImplementedError


class PointLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models to recover a MLE estimate of
    the latent variable :math:`\\mathbf X`.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    """

    def __init__(self, n, latent_dim, X_init):
        super().__init__(n, latent_dim)
        self.register_parameter("X", X_init)

    def forward(self):
        return self.X


class MAPLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models to recover a MAP estimate of
    the latent variable :math:`\\mathbf X`, based on some supplied prior.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    :param ~gpytorch.priors.Prior prior_x: prior for :math:`\\mathbf X`
    """

    def __init__(self, n, latent_dim, X_init, prior_x):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.register_parameter("X", X_init)
        self.register_prior("prior_x", prior_x, "X")

    def forward(self):
        return self.X


class VariationalLatentVariable(LatentVariable):
    """
    This class is used for GPLVM models to recover a variational approximation of
    the latent variable :math:`\\mathbf X`. The variational approximation will be
    an isotropic Gaussian distribution.

    n: number of latent (outputs)
    data_dim: number of inputs
    """

    def __init__(self, n, data_dim, latent_dim, X_init, prior_x, trainable_latent_dim=None):
        super().__init__(n, latent_dim)

        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        if trainable_latent_dim == None:
            self.q_mu = torch.nn.Parameter(X_init)
        else:
            NotImplementedError

        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))
        # self.q_log_sigma = -5 * torch.ones(n, latent_dim)
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")
    
    def forward(self, batch_idx=None):

        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_log_sigma_batch = self.q_log_sigma[batch_idx, ...]

        q_x = torch.distributions.Normal(q_mu_batch, q_log_sigma_batch.exp())

        p_mu_batch = self.prior_x.loc[batch_idx, ...]
        p_var_batch = self.prior_x.variance[batch_idx, ...]

        p_x = torch.distributions.Normal(p_mu_batch, p_var_batch)
        x_kl = kl_gaussian_loss_term(q_x, p_x, len(batch_idx), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        
        return q_x.rsample()
    '''
    def forward(self):
        from gpytorch.mlls import KLGaussianAddedLossTerm

        # Variational distribution over the latent variable q(x)
        # q_x = torch.distributions.Normal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma)) # TODO: Is softplus function a correct choice?
        q_x = torch.distributions.Normal(self.q_mu, torch.exp(self.q_log_sigma))
        x_kl = KLGaussianAddedLossTerm(q_x, self.prior_x, self.n, self.data_dim) # devide by self.n * self.data_dim
        self.update_added_loss_term("x_kl", x_kl)  # Update the KL term
        return q_x.rsample()
    '''
class VariationalCatLatentVariable(LatentVariable):
    """
    Advanced VariationalLatentVariable.
    Supports flexible latent variables with only parts of them trainable, i.e Cat of two parts.
    """

    def __init__(self, n, data_dim, latent_dim, X_init, prior_x, trainable_latent_dim=None):
        super().__init__(n, latent_dim)

        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        if trainable_latent_dim == None:
            NotImplementedError
        else:
            self.register_parameter('q_mu_1', torch.nn.Parameter(X_init[:, :trainable_latent_dim]))
            self.register_buffer('q_mu_2', X_init[:, trainable_latent_dim:])

        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))
        # self.q_log_sigma = -5 * torch.ones(n, latent_dim)
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")
    
    @property
    def q_mu(self):
        return torch.cat((self.q_mu_1, self.q_mu_2), dim=1)

    # Save as VariationalLatentVariable implementation
    def forward(self, batch_idx=None):

        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_log_sigma_batch = self.q_log_sigma[batch_idx, ...]

        q_x = torch.distributions.Normal(q_mu_batch, q_log_sigma_batch.exp())

        p_mu_batch = self.prior_x.loc[batch_idx, ...]
        p_var_batch = self.prior_x.variance[batch_idx, ...]

        p_x = torch.distributions.Normal(p_mu_batch, p_var_batch)
        x_kl = kl_gaussian_loss_term(q_x, p_x, len(batch_idx), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        
        return q_x.rsample()
    
class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_x, p_x, n, data_dim):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        
        # G 
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0) # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return (kl_per_point/self.data_dim)