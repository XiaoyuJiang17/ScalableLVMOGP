from gpytorch.module import Module
from abc import ABC
from gpytorch.models import ApproximateGP
from torch import Tensor
from models_.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistribution
from typing import Optional
import torch
from gpytorch.utils.errors import CachingError
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.distributions import MultivariateNormal, Distribution
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from linear_operator.operators import (
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator import to_dense
from gpytorch import settings

class KroneckerVariationalStrategy(Module, ABC):

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points_X: Tensor,
        inducing_points_C: Tensor,
        variational_distribution: CholeskyKroneckerVariationalDistribution,
        learn_inducing_locations_X: bool = True,
        learn_inducing_locations_C: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__()
        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points X and C
        inducing_points_X = inducing_points_X.clone()
        inducing_points_C = inducing_points_C.clone()

        if inducing_points_X.dim() == 1:
            inducing_points_X = inducing_points_X.unsqueeze(-1)
        if inducing_points_C.dim() == 1:
            inducing_points_C = inducing_points_C.unsqueeze(-1)

        if learn_inducing_locations_X:
            self.register_parameter(name="inducing_points_X", parameter=torch.nn.Parameter(inducing_points_X))
        else:
            self.register_buffer("inducing_points_X", inducing_points_X)

        if learn_inducing_locations_C:
            self.register_parameter(name="inducing_points_C", parameter=torch.nn.Parameter(inducing_points_C))
        else:
            self.register_buffer("inducing_points_C", inducing_points_C)
        
        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @cached(name="cholesky_factor_X", ignore_args=True)
    def _cholesky_factor_X(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)
    
    @cached(name="cholesky_factor_C", ignore_args=True)
    def _cholesky_factor_C(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
        return self._variational_distribution()

    @property
    def jitter_val(self) -> float:
        if self._jitter_val is None:
            return settings.variational_cholesky_jitter.value(dtype=self.inducing_points_X.dtype)
        return self._jitter_val

    @jitter_val.setter
    def jitter_val(self, jitter_val: float):
        self._jitter_val = jitter_val
    
    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        """
        # NOTE: due to whitening, prior is N(0,I) not N(0, K_ZZ).
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence
        
    def forward(
        self,
        x: Tensor,
        c: Tensor,
        inducing_points_X: Tensor,
        inducing_points_C: Tensor,
        inducing_values: Tensor, 
        variational_inducing_covar: Optional[LinearOperator] = None, # after kron_product X (@) C
        **kwargs,
    ) -> MultivariateNormal:
        # Ensure x, c has the same length, i.e. a (x[i],c[i]) pair jointly determines a prediction value / target value.
        assert x.shape[-2] == c.shape[-2]
        mini_batch_size = x.shape[-2]

        test_mean = self.model.mean_module_X(Tensor([i for i in range(mini_batch_size)]))

        # NOTE: following two tensors might contains repeting elements! That's a problem when computing cov matrix!
        full_inputs_X = torch.cat([x, inducing_points_X], dim=-2)
        full_inputs_C = torch.cat([c, inducing_points_C], dim=-2)

        full_X_covar = self.model.covar_module_X(full_inputs_X)
        full_C_covar = self.model.covar_module_C(full_inputs_C)

        # Covariance terms
        induc_X_covar = full_X_covar[mini_batch_size:, mini_batch_size:]
        induc_C_covar = full_C_covar[mini_batch_size:, mini_batch_size:]
        data_data_covar = full_X_covar[:mini_batch_size, :mini_batch_size] * full_C_covar[:mini_batch_size, :mini_batch_size] # elementwise product

        induc_X_data_X_covar = full_X_covar[mini_batch_size:, :mini_batch_size] # (n_induc_X, mini_batch_size)
        induc_C_data_C_covar = full_C_covar[mini_batch_size:, :mini_batch_size] # (n_induc_C, mini_batch_size)
        n_induc_X, n_induc_C = inducing_points_X.shape[-2], inducing_points_C.shape[-2]
        # Some Test Unit
        assert induc_X_data_X_covar.shape[-2] == n_induc_X
        assert induc_C_data_C_covar.shape[-2] == n_induc_C
        assert induc_X_data_X_covar.shape[-1] == mini_batch_size
        assert induc_C_data_C_covar.shape[-1] == mini_batch_size
        # broadcasting
        induc_data_covar = induc_X_data_X_covar.to_dense().unsqueeze(1) * induc_C_data_C_covar.to_dense().unsqueeze(0)
        induc_data_covar = induc_data_covar.reshape((n_induc_X*n_induc_C), mini_batch_size)
        
        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX

        L_X_inv = self._cholesky_factor_X(induc_X_covar).solve(torch.eye(induc_X_covar.size(-1), device=induc_X_covar.device, dtype=induc_X_covar.dtype))
        L_C_inv = self._cholesky_factor_C(induc_C_covar).solve(torch.eye(induc_C_covar.size(-1), device=induc_C_covar.device, dtype=induc_C_covar.dtype))
        L_inv = KroneckerProductLinearOperator(L_X_inv, L_C_inv).to_dense()
        
        if L_inv.shape[0] != induc_data_covar.shape[0]:
            print('nasty shape incompatibilies error happens!')
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L_X_inv = self._cholesky_factor_X(induc_X_covar).solve(torch.eye(induc_X_covar.size(-1), device=induc_X_covar.device, dtype=induc_X_covar.dtype))
            L_C_inv = self._cholesky_factor_C(induc_C_covar).solve(torch.eye(induc_C_covar.size(-1), device=induc_C_covar.device, dtype=induc_C_covar.dtype))
            L_inv = KroneckerProductLinearOperator(L_X_inv, L_C_inv).to_dense()
        interp_term = (L_inv @ induc_data_covar.to(L_inv.dtype))
        '''
        # L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs_X.dtype)
        '''
        # Compute the mean of q(f)
        predictive_mean = (interp_term.transpose(-1, -2) @ (inducing_values.to(interp_term.dtype).unsqueeze(-1)).squeeze(-1)) + test_mean

        # Compute the covariance of q(f)
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1).to(interp_term.dtype)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term.to(interp_term.dtype) @ interp_term),
            )
        
        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
    
    def __call__(self, x: Tensor, c: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        if prior:
            return self.model.forward(x, c, **kwargs)

        if self.training:
            self._clear_cache()
        
        inducing_points_X = self.inducing_points_X
        inducing_points_C = self.inducing_points_C

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal): 
            return super().__call__(
                x,
                c,
                inducing_points_X,
                inducing_points_C,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution (NOT IMPLEMENTED YET)."
            )

        