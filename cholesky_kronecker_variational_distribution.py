from gpytorch.module import Module
import torch
from abc import ABC
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator, KroneckerProductLinearOperator

class CholeskyKroneckerVariationalDistribution(Module, ABC):

    def __init__(
        self,
        n_inducing_C: int,
        n_inducing_X: int,
        mean_init_std: float = 1e-3
    ):
        super().__init__()
        self.n_inducing_C = n_inducing_C
        self.n_inducing_X = n_inducing_X
        self.mean_init_std = mean_init_std

        mean_init = torch.zeros(n_inducing_C*n_inducing_X)
        covar_init_X = torch.eye(n_inducing_X, n_inducing_X)
        covar_init_C = torch.eye(n_inducing_C, n_inducing_C)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar_X", parameter=torch.nn.Parameter(covar_init_X))
        self.register_parameter(name="chol_variational_covar_C", parameter=torch.nn.Parameter(covar_init_C))

        _ = self.forward() # get self.variational_covar in this step.

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def shape(self) -> torch.Size:
        r"""
        rtype: torch.Size
        There are n_inducing_C * n_inducing_X inducing points.
        """
        return torch.Size([(self.n_inducing_C)*(self.n_inducing_X)])

    def forward(self) -> MultivariateNormal:
        chol_variational_covar_X = self.chol_variational_covar_X
        chol_variational_covar_C = self.chol_variational_covar_C
        assert chol_variational_covar_X.dtype == chol_variational_covar_C.dtype
        assert chol_variational_covar_X.device == chol_variational_covar_C.device 
        dtype = chol_variational_covar_X.dtype
        device = chol_variational_covar_X.device

        # First make the cholesky factor is upper triangular
        lower_mask_X = torch.ones(self.chol_variational_covar_X.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_X = TriangularLinearOperator(chol_variational_covar_X.mul(lower_mask_X))
        
        lower_mask_C = torch.ones(self.chol_variational_covar_C.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_C = TriangularLinearOperator(chol_variational_covar_C.mul(lower_mask_C))

        # Now construct the actual matrix
        variational_covar_X = CholLinearOperator(chol_variational_covar_X)
        variational_covar_C = CholLinearOperator(chol_variational_covar_C)
        self.variational_covar = KroneckerProductLinearOperator(variational_covar_X, variational_covar_C)

        # Some Test Unit
        '''
        assert torch.allclose(chol_variational_covar_X.to_dense(), self.chol_variational_covar_X.mul(lower_mask_X)) == True
        assert torch.allclose(chol_variational_covar_C.to_dense(), self.chol_variational_covar_C.mul(lower_mask_C)) == True
        assert torch.allclose(variational_covar_X.to_dense(), chol_variational_covar_X.to_dense() @ chol_variational_covar_X.to_dense().T) == True
        assert torch.allclose(torch.kron(variational_covar_X.to_dense(), variational_covar_C.to_dense()), self.variational_covar.to_dense())
        '''

        return MultivariateNormal(self.variational_mean, self.variational_covar)

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        raise NotImplementedError("This function is not implemented yet.")
