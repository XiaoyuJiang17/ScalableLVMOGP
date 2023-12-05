from gpytorch.distributions import base_distributions, MultivariateNormal
from torch import Tensor
import torch


class Multi_Output_Multi_Class_AR_Likelihood():

    def __init__(self, list_classes):
        self.multiplier = Tensor(list_classes) - 1

    def expected_log_prob(self, input: MultivariateNormal, ref: Tensor):
        '''
        Arg:
            ref: of shape (num_outputs, num_class_per_output+1, num_input_samples)
        '''
        mean_tensor = input.loc.reshape(ref.shape)
        var_tensor = input.variance.reshape(ref.shape)

        psy = torch.exp(var_tensor[:,-1,:]/2 - mean_tensor[:,-1,:])
        gamma = torch.exp(var_tensor[:,:-1,:]/2 + mean_tensor[:,-1,:])

        batch_sum_gamma = gamma.sum(dim=1)
        psy = psy.reshape(batch_sum_gamma.shape) # (num_outputs, num_input_samples)
        assert psy.shape[0] == ref.shape[0]
        assert psy.shape[-1] == ref.shape[-1]
        P = batch_sum_gamma * psy
        intermediate_term = ( P.transpose(-1,-2) * self.multiplier / (ref.shape[1]-1) ).transpose(-1,-2) + 1
        # print('have a look at minimum term inside log', intermediate_term.min())
        intermediate_term = torch.log(intermediate_term.data)

        result = - intermediate_term.sum()

        return result

