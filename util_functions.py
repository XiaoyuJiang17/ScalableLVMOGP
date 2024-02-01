import os
import numpy as np
import pandas as pd
import json
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.settings import _linalg_dtype_cholesky
from linear_operator.operators import KroneckerProductLinearOperator, TriangularLinearOperator, LinearOperator, CholLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator import to_dense
from linear_operator.operators import MatmulLinearOperator
from torch import Tensor
from torch.distributions import MultivariateNormal
import csv
import random
import math
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score

################################################   Convention Introduction  ################################################
# X (latent) , C (input) , Y (target) are all vectors. len(Y) = len(X) * len(C). For i th element x_i in X and j th element c_j in C, the corresponding target in Y 
# is at  (i - 1) * len(C) + j position. In cases that Y representing 2d images (for instance 28*28 image), C is of form [[1,1], [1,2], ... , [1,28], [2,1], [2,2], ... , [2,28], ... , [28,1], [28,2], ..., [28,28]].
# Y is of form [Picture1(1,1), Picture1(1,2), ..., Picture1(28,1), Picture1(28,28), Picture2(1,1), ..., Picture2(28,28), ..., PictureN(1,1), ..., PictureN(28,28)].


def inhomogeneous_index_of_batch_Y(batch_index_X, batch_index_C, n_X, n_C):
    """
    Inhomogeneously get set of indices for Y given X and C indicies. 
    Args:
        batch_index_X and batch_index_C jointly determine the position of the corresponding element in Y.
        n_X: number of elements in X (not in use)
        n_C: number of elements in C
    Return:
        List of indices of elements in Y, which of length len(batch_index_X), which also equal to len(batch_index_C).
    """
    assert len(batch_index_X) == len(batch_index_C)
    batch_index_Y = []
    for (index_X, index_C) in zip(batch_index_X, batch_index_C):
        batch_index_Y.append(index_X * n_C + index_C)
    
    return batch_index_Y

def remove_forbidden_pairs(batch_index_X, batch_index_C, forbidden_pairs):
    '''
    NOTE: elements in batch_index_X and batch_index_C are paired, this function removes pairs shown in forbidden_pairs.
    Args:
        batch_index_X: list of X indices in current mini-batch
        batch_index_C: list of C incices in current mini-batch
        forbidden_pairs: list of (X,C) pairs forbidden in training.
    Return:
        _batch_index_X, _batch_index_C: new list of X/C indices after removing forbidden pairs.
    '''
    _batch_index_X = []
    _batch_index_C = []
    for index_X, index_C in zip(batch_index_X, batch_index_C):
        if (index_X, index_C) not in forbidden_pairs:
            _batch_index_X.append(index_X)
            _batch_index_C.append(index_C)

    return _batch_index_X, _batch_index_C


################################################    Randomly Generating Forbidden Pairs    ################################################


def random_gene_forbidden_pairs(n_X, n_C, X_forbidden_rate=0.3, C_forbidden_rate=0.3):
    """
    Randomly generate forbidden pairs given forbidden ratios.
    Args:
        n_X: total number of latents X.
        n_C: total number of indices C.
        X_forbidden_rate: forbidden ratio for latents.
        C_forbidden_rate: forbidden ratio for indices.
    Return:  
        List of forbidden pairs, which are masked during training.
    NOTE: X_indices and C_indices might have duplicates, but final forbidden_pairs won't.
    """
    valid_X_indices = np.arange(n_X)
    valid_C_indices = np.arange(n_C)
    X_indices = np.random.choice(valid_X_indices, size=int(X_forbidden_rate*n_X), replace=True)
    C_indices = np.random.choice(valid_C_indices, size=int(C_forbidden_rate*n_C), replace=True)
    forbidden_pairs = set() # Remove duplicates
    for x_index, c_index in zip(X_indices, C_indices):
        forbidden_pairs.add((x_index, c_index))

    return list(forbidden_pairs)

def proper_gene_forbidden_pairs(n_X, n_C, num_forbidden=20):
    """
    Generate forbidden pairs. For each X (latent), same number of C are forbidden, but
    different X may leads to different set of C to forbidden.
    Args:
        n_X: the number of latents.
        n_C: the number of indices (for each latent X).
        num_forbidden: number of C forbidden for each latent X.
    Return:
        List of forbidden pairs, which are masked during training.
    """
    valid_X_indices = np.arange(n_X)
    valid_C_indices = np.arange(n_C)
    forbidden_pairs = []
    for x_index in valid_X_indices:
        C_indices = np.random.choice(valid_C_indices, size=int(num_forbidden), replace=False)
        for c_index in C_indices:
            forbidden_pairs.append((x_index, c_index))

    assert len(forbidden_pairs) == n_X * num_forbidden

    return forbidden_pairs


def tidily_gene_forbidden_pairs(n_X, n_C, num_forbidden=20, C_indices:Tensor=None):
    """
    Generate forbidden pairs. For each X (latent), same number of C are forbidden, and different X 
    have the SAME set of C to forbidden.
    Args:
        n_X: the number of latents.
        n_C: the number of indices (for each latent X).
        num_forbidden: number of C forbidden for each latent X.
        C_indices: optional, if given C_indices no need to be generated again.
    Return:
        List of forbidden pairs, which is masked during training.
    """
    valid_X_indices = np.arange(n_X)
    if C_indices == None:
        C_indices = np.random.choice(np.arange(n_C), size=int(num_forbidden), replace=False)
    else:
        num_forbidden = C_indices.shape[0]
    forbidden_pairs = []
    for x_index in valid_X_indices:
        for c_index in C_indices:
            forbidden_pairs.append((x_index, c_index))
 
    assert len(forbidden_pairs) ==  n_X * num_forbidden

    return forbidden_pairs

################################################    Plotting functions   ################################################

def plot_2dlatent_with_label(latents_2d, labels, figsize=(10, 8)):
    """
    Plot 2 selected dims of latents in 2D plane (colored with labels).
    Args:
        latents_2d: tensor of shape (n_data, 2)
        labels: tensor of shape (n_data)
    Return:
        None
    """
    unique_labels = torch.unique(labels)
    plt.figure(figsize=figsize)

    for label in unique_labels:
        mask = labels == label
        subset = latents_2d[mask]
        plt.scatter(subset[:, 0], subset[:, 1], label=str(label.item()))

    plt.legend()
    plt.show()

def plot_loss_and_savefig(loss_list:list, rootpath_to_save='/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/experi_results', figsize=(10, 5)):
    """
    Given list of training losses, plot them in fig and save that fig to path.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    # Save the figure to the designated path
    save_path = 'training_loss_curve.png'
    plt.savefig(f'{rootpath_to_save}/{save_path}')

def  plot_traindata_testdata_fittedgp(train_X: Tensor, train_Y: Tensor, test_X: Tensor, test_Y: Tensor, gp_X: Tensor, gp_pred_mean: Tensor, gp_pred_std: Tensor, inducing_points_X: Tensor, n_inducing_C:int=15, picture_save_path:str=''):
    '''
    This is a 1 dim plot: train (corss) and test (dot) data, fitted gp all in the same figure.
    The shadowed area is mean +/- 1.96 gp_pred_std.
    Args:
        train_X: training input locations.
        train_Y: target values for corresponding train_X.
        test_X: testing input locations.
        test_Y: target values for corresponding test_X.
        gp_pred_mean: mean of predictive funtion.
        gp_pred_std: std of predictive function. 

    Return:    
    NOTE: gp_pred_std**2 is formed by two parts, variance from q(f_test) and likelihood variance.
    '''
    # Get numpy 
    train_X_np = train_X.numpy().squeeze()
    train_Y_np = train_Y.numpy().squeeze()
    test_X_np = test_X.numpy().squeeze()
    test_Y_np = test_Y.numpy().squeeze()
    gp_pred_mean_np = gp_pred_mean.numpy().squeeze()
    gp_pred_std_np = gp_pred_std.numpy().squeeze()
    gp_X = gp_X.numpy().squeeze()
    inducing_points_X = inducing_points_X.numpy().squeeze()

    # Plot training data as crosses
    plt.scatter(train_X_np, train_Y_np, c='r', marker='x', label='Training Data')

    # Plot test data as dots
    plt.scatter(test_X_np, test_Y_np, c='b', marker='o', label='Test Data', alpha=0.2)

    # Plot inducing points on x axis
    plt.scatter(inducing_points_X, [plt.gca().get_ylim()[0] - 1] * n_inducing_C, color='black', marker='^', label='Inducing Locations')


    # Plot GP predictions as a line
    plt.plot(gp_X, gp_pred_mean_np, 'k', lw=1, zorder=9)
    plt.fill_between(gp_X, gp_pred_mean_np - 1.96 * gp_pred_std_np, gp_pred_mean_np + 1.96 * gp_pred_std_np, alpha=0.2, color='k')

    plt.legend()
    plt.title("Train/Test Data and Fitted GP")
    plt.tight_layout()
    plt.savefig(picture_save_path)
    plt.show()

    return None

def plot_true_and_fitted_latent(X_true:Tensor, X_fitted_mean:Tensor, X_fitted_var:Tensor):
    from matplotlib.patches import Ellipse
    '''
    Plot true latents (X_true) in red cross and fitted latent distribution (discribed by X_fitted_mean, X_fitted_var) as blue ellipses.
    NOTE: all three inputs have same size (20, 2) 
    '''

    assert X_true.shape == X_fitted_mean.shape == X_fitted_var.shape == (20, 2), "All inputs must have the size (20, 2)"
    
    plt.figure(figsize=(8, 6))
    
    # Plot true latents
    plt.scatter(X_true[:, 0], X_true[:, 1], c='red', marker='x', label='True Latents')
    
    # Plot fitted latent distributions
    ceofficient = 1
    for mean, var in zip(X_fitted_mean, X_fitted_var):
        # Convert variance to standard deviation and use it as the width and height of the ellipse
        std_dev = np.sqrt(var)
        
        # Draw an ellipse for each point
        
        ellipse = Ellipse(xy=mean, width=ceofficient*std_dev[0], height=ceofficient*std_dev[1], edgecolor='blue', facecolor='none', label='Fitted Latent Distribution')
        plt.gca().add_patch(ellipse)
    
    # Only use one label for fitted latent distributions in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('True and Fitted Latent Variables')
    plt.grid(True)
    plt.show()

################################################    Sampling of Latents and Inputs during Training    ################################################
# 


def random_sample_index_X_and_C(my_model, batch_size, forbidden_pairs=None):
    '''
    Randomly sample (valid) indices of X and C, then remove all forbidden pairs (if necessary).
    No control of how many indices for each latent X has in one sampling.
    NOTE: no duplicates in batch_index_X and batch_index_C!
    Args:
        my_model: instance of LVMOGP_SVI, the function _get_batch_idx is needed.
        batch_size: number of X(C) indices, duplicates are not possible as in _get_batch_idx, replace=False.
        forbidden_pairs (Optional): X C pairs forbidden to appear duing sampling. 
    Return:
        batch_index_X, batch_index_C: of same length (batch_size), two lists jointly determine positions in Y.
    '''
    batch_index_X = my_model._get_batch_idx(batch_size, sample_X = True)
    batch_index_C = my_model._get_batch_idx(batch_size, sample_X = False)

    if forbidden_pairs != None:
        # missing data: some (x,c) pairs are forbidden during sampling 
        batch_index_X, batch_index_C = remove_forbidden_pairs(batch_index_X, batch_index_C, forbidden_pairs)
    return batch_index_X, batch_index_C

def proper_sample_index_X_and_C(my_model, batch_size_X, batch_size_C, forbidden_pairs=None):
    '''
    Sample same number of indices of C (but possibly different sets) for every latent X.
    Args:
        my_model: instance of LVMOGP_SVI, the function _get_batch_idx is needed.
        batch_size_X: number of X indices.
        batch_size_C: number of C indices for every latent X.
        forbidden_pairs (Optional): X C pairs forbidden to appear duing sampling. 
    Return:
        _batch_index_X, _batch_index_C: of same length

    NOTE: teturned lists do NOT have guaranteed length, which is due to seperation of 2 steps:
            sampling and removing.
    '''
    _batch_index_X = []
    _batch_index_C = []
    batch_index_X = my_model._get_batch_idx(batch_size_X, sample_X = True)
    for index_X in batch_index_X:
        batch_index_C = my_model._get_batch_idx(batch_size_C, sample_X = False)
        for index_C in batch_index_C:
            _batch_index_X.append(index_X)
            _batch_index_C.append(index_C)
    
    if forbidden_pairs != None:
        # missing data: some (x,c) pairs are forbidden during sampling 
        _batch_index_X, _batch_index_C = remove_forbidden_pairs(_batch_index_X, _batch_index_C, forbidden_pairs)

    return _batch_index_X, _batch_index_C

def proper_sample_index_X_and_C_(my_model, batch_size_X, batch_size_C, forbidden_pairs_csv=None):
    '''
    Sample X and C indices from valid range, i.e. all indices defined by my_model but exclude pairs appeared in forbidden_pairs_csv. 
    NOTE: different from the function 'proper_sample_index_X_and_C'; in '_proper_sample_index_X_and_C'
        two returned lists have equal guaranteed length.
    '''
    first_elements, second_elements = extract_tuple_elements(forbidden_pairs_csv) # equal length, lists of X and C indices
    first_elements, second_elements = np.array(first_elements), np.array(second_elements)
    _batch_index_X = []
    _batch_index_C = []
    batch_index_X = my_model._get_batch_idx(batch_size_X, sample_X = True)
    for index_X in batch_index_X:
        forbidden_C = second_elements[first_elements == index_X]
        _index_C_list = [c for c in list(range(my_model.n_C)) if c not in forbidden_C]
        index_C_list = random.sample(_index_C_list, batch_size_C) # too large batch_size for C may lead to error
        assert len(index_C_list) == batch_size_C 
        for index_C in index_C_list:
            _batch_index_X.append(index_X)
            _batch_index_C.append(index_C)

    assert len(_batch_index_X) == len(_batch_index_C) == batch_size_X * batch_size_C
    return _batch_index_X, _batch_index_C

def tidily_sample_index_X_and_C(my_model, batch_size_X, batch_size_C, forbidden_pairs = None):
    '''
    Tidily sample the same set of indices for each latent X.
    Args:
        my_model: instance of LVMOGP_SVI, the function _get_batch_idx is needed.
        batch_size_X: number of X indices.
        batch_size_C: number of C indices.
        forbidden_pairs (Optional): X C pairs forbidden to appear duing sampling. 
    Return:
        _batch_index_X, _batch_index_C: np.array of same length (batch_size_X * batch_size_C), two lists jointly determine positions in Y.
    '''
    batch_all = []
    batch_index_X = my_model._get_batch_idx(batch_size_X, sample_X = True)
    batch_index_C = my_model._get_batch_idx(batch_size_C, sample_X = False)
    for index_X in batch_index_X:
        for index_C in batch_index_C:
            batch_all.append([index_X, index_C])

    batch_all = np.array(batch_all)
    _batch_index_X = batch_all[:, 0]
    _batch_index_C = batch_all[:, 1]

    if forbidden_pairs != None:
        # missing data: some (x,c) pairs are forbidden during sampling 
        _batch_index_X, _batch_index_C = remove_forbidden_pairs(_batch_index_X, _batch_index_C, forbidden_pairs)

    return _batch_index_X, _batch_index_C

################################################  Regression Synthetic Data Expri  ################################################
#                               

# Below is some code about generating syhthetic datasets.
def tidily_sythetic_data_from_MOGP(n_C:int=700, n_X:int=20, latent_dim:int=2, noise_scale:int=0.05, X_:Tensor=None, C_:Tensor=None, kernel_parameters: dict=None, random_seed=0):

    '''
    Generate sythetic data without missing data. 
    See 'Synthetic Data' section of Zhenwen's paper.

    Zhenwen Dai et. al. 2017 NIPS Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes.

    Args:
        n_C: number of inputs for each output.
        n_X: mumber of latent variables (output functions).
        latent_dim: the dimensionality of hidden variable.
        noise_scale: noise variance of gaussian likelihood.
        X_: optional, if not None, X does not need to be generated again.
        C_: optional, if not None, C does not need to be generated again.

    Return:
        X: generated hidden variables.
        C: generated input locations.
        sample: (sampled) function values at corresponding X and C.
    '''
    # Set random seeds for reproducibility
    np.random.seed(random_seed)  # Set seed for NumPy
    torch.manual_seed(random_seed)  # Set seed for PyTorch

    index_dim = 1

    if kernel_parameters == None:
        default_kernel_parameters = {'X_raw_outputscale': torch.tensor(0.0), 'X_raw_lengthscale': torch.tensor([0.1 for _ in range(latent_dim)]),
                                     'C_raw_outputscale': torch.tensor(0.9), 'C_raw_lengthscale': torch.tensor([0.1 for _ in range(index_dim)])}
    else:
        default_kernel_parameters = kernel_parameters

    if C_ == None:
        C = Tensor(np.linspace(-10, 10, n_C)) # inputs in our cases, 1 point every distance 0.5 
    else:
        C = C_
        
    if X_ == None:
        X = Tensor(np.random.multivariate_normal([0 for _ in range(latent_dim)], np.eye(latent_dim), (n_X,))) # 20 outputs, sampled from Normal(0, I)
    else:
        X = X_

    covar_module_X = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
    covar_module_C = ScaleKernel(RBFKernel(ard_num_dims=index_dim))
    # TODO: Try another implementation ... 
    covar_module_X.raw_outputscale.data = default_kernel_parameters['X_raw_outputscale'].to(covar_module_X.device)
    covar_module_X.base_kernel.raw_lengthscale.data = default_kernel_parameters['X_raw_lengthscale'].to(covar_module_X.device) 
    covar_module_C.raw_outputscale.data = default_kernel_parameters['C_raw_outputscale'].to(covar_module_C.device)
    covar_module_C.base_kernel.raw_lengthscale.data = default_kernel_parameters['C_raw_lengthscale'].to(covar_module_C.device)

    covar_X = covar_module_X(X)
    print('covar_X has shape', covar_X.shape)
    covar_C = covar_module_C(C)
    print('covar_C has shape', covar_C.shape)
    # K + sigma^2 * I 
    covar_final = KroneckerProductLinearOperator(covar_X, covar_C).add_jitter(noise_scale).to_dense().detach()
    print('covar_final has shape', covar_final.shape)
    mean_final = Tensor([0. for _ in range(n_C * n_X)])

    dist = MultivariateNormal(mean_final, covar_final)
    sample = dist.sample()
    print('Dataset Generated!')
    return X, C, sample, default_kernel_parameters

# TODO: non-tidily generating data from MOGP.
def tidily_sythetic_data_from_MOGP_smartly(n_C:int=700, n_X:int=20, latent_dim:int=2, noise_scale:int=0.05, X_:Tensor=None, C_:Tensor=None, kernel_parameters: dict=None, random_seed=10):
    '''
    follows same methodology as above for generating synthetic dataset ... 
    but the implementation is more smart ... (no inverting whole big covariance matrix is required)
    '''
    # Set random seeds for reproducibility
    np.random.seed(random_seed)  # Set seed for NumPy
    torch.manual_seed(random_seed)  # Set seed for PyTorch

    index_dim = 1

    if kernel_parameters == None:
        default_kernel_parameters = {'X_raw_outputscale': torch.tensor(0.0), 'X_raw_lengthscale': torch.tensor([0.1 for _ in range(latent_dim)]),
                                     'C_raw_outputscale': torch.tensor(0.9), 'C_raw_lengthscale': torch.tensor([0.1 for _ in range(index_dim)])}
    else:
        default_kernel_parameters = kernel_parameters

    if C_ == None:
        C = Tensor(np.linspace(-10, 10, n_C)) # inputs in our cases, 1 point every distance 0.5 
    else:
        C = C_
        
    if X_ == None:
        X = Tensor(np.random.multivariate_normal([0 for _ in range(latent_dim)], 1 * np.eye(latent_dim), (n_X,))) # 20 outputs, sampled from Normal(0, I)
    else:
        X = X_

    covar_module_X = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
    covar_module_C = ScaleKernel(RBFKernel(ard_num_dims=index_dim))
    # TODO: Try another implementation ... 
    covar_module_X.raw_outputscale.data = default_kernel_parameters['X_raw_outputscale'].to(covar_module_X.device)
    covar_module_X.base_kernel.raw_lengthscale.data = default_kernel_parameters['X_raw_lengthscale'].to(covar_module_X.device) 
    covar_module_C.raw_outputscale.data = default_kernel_parameters['C_raw_outputscale'].to(covar_module_C.device)
    covar_module_C.base_kernel.raw_lengthscale.data = default_kernel_parameters['C_raw_lengthscale'].to(covar_module_C.device)

    covar_X = covar_module_X(X)
    # print('covar_X has shape', covar_X.shape)
    covar_C = covar_module_C(C)
    # print('covar_C has shape', covar_C.shape)
    total_dim = int(covar_X.shape[0] * covar_C.shape[0])
    Chol_X = TriangularLinearOperator(psd_safe_cholesky(to_dense(covar_X)))
    Chol_C = TriangularLinearOperator(psd_safe_cholesky(to_dense(covar_C)))

    ### Approach 0:
    # K + sigma^2 * I 
    # covar_final = KroneckerProductLinearOperator(covar_X, covar_C).add_jitter(noise_scale).to_dense().detach()
    # print('covar_final has shape', covar_final.shape)
    # mean_final = Tensor([0. for _ in range(n_C * n_X)])
    # sample = MultivariateNormal(mean_final, covar_final)

    ### Approach 1:
    # NOTE using mixed Kronecker matrix-vector product property
    # NOTE: something WRONG here! Samples looks have high variance.
    # standard_noise_matrix = torch.randn(n_C, n_X)
    # sample_matrix = (MatmulLinearOperator(Chol_C, standard_noise_matrix).to_dense().detach() @ Chol_X.to_dense().detach().transpose(-1, -2))
    # sample_ = sample_matrix.reshape(-1)
    
    ### Approach 2: 
    # Chol_whole_2 = KroneckerProductLinearOperator(Chol_X, Chol_C).to_dense().detach()
    # print('Chol_whole_2', Chol_whole_2)
    # sample_ = Chol_whole_2 @ torch.randn(total_dim)

    ### Approach 3:
    # Iteratively using approach 2, so that only small RAM is necessary
    it_stop = False
    it_latent_length = 20
    it_start = 0
    it_end = it_start + it_latent_length
    standard_noise = torch.randn(total_dim)
    sample_ = torch.zeros(n_X, n_C)

    while it_stop == False:
        it_Chol_X = Chol_X[it_start:it_end, :] # of shape (it_end-it_start, n_X)
        it_Chol_whole = KroneckerProductLinearOperator(it_Chol_X, Chol_C).to_dense().detach() # of shape (it_end-it_start)*n_C, n_X*n_C
        sample_[it_start:it_end , :] = ( it_Chol_whole @  standard_noise).reshape(int(it_end-it_start), n_C)

        if it_end < n_X:
            it_start += it_latent_length
            it_end = min(it_end+it_latent_length, n_X)
        else:
            it_stop = True

    sample_ = sample_.reshape(-1)

    torch.manual_seed(98 + random_seed) # make sure this is a different random sample.
    extra_noise = torch.randn(total_dim) * math.sqrt(noise_scale)

    sample = sample_ + extra_noise
    return X, C, sample, default_kernel_parameters

# For Synthetic dataset experiment
def sample_index_X_and_C_from_list(C_index_list, batch_size_X, batch_size_C):
    '''
    Given C_index_list (list of list) containing available C indices for each latent X.
        * len(C_list) == the number of all latents.
        * C_list[i] refers to the list of available indices for (i+1) th latent X.
    Args:
        batch_size_X: the number of elements sampled from all latents.
        batch_size_C: the number of elements sampled from all available C.
    Return:
        X_list: selected_X_indices
        C_list: selected_C_indices
    '''
    assert batch_size_X <= len(C_index_list)
    assert batch_size_C <= len(C_index_list[0]) # here assume C_index_list[0] is a list
    X_indices = random.sample(range(len(C_index_list)), batch_size_X)
    X_list = [x for x in X_indices for _ in range(batch_size_C)]
    C_list = []
    for i in X_indices:
        C_indices = random.sample(C_index_list[i], batch_size_C)
        for j in C_indices:
            C_list.append(j) # C_index_list[i] is a list
    assert len(X_list) == len(C_list) == batch_size_X*batch_size_C

    return X_list, C_list


def extract_tuple_elements(file_path):
    """
    Given a .csv file path containing forbidden pairs in a form '(29, 187)', extract first and second elements
    in this .csv to two lists.
    Return:
        first_elements: indices for latent X.
        second_elements: indicies for input C.
    """

    first_elements = []
    second_elements = []

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for element in row:  # Each row may contain multiple tuple elements
                # Strip the parentheses and split the tuple into two elements
                tuple_elements = element.strip("()").split(', ')
                if len(tuple_elements) == 2:
                    first, second = tuple_elements
                    first_elements.append(int(first))
                    second_elements.append(int(second))
    assert len(first_elements) == len(second_elements)

    return first_elements, second_elements


def gene_2dimage_inputs(image_size=28, shift=14):
    """
    Given the size of the image (such as 28*28), generate corresponding input for each pixel value.  
    for pixel at i th row and j th column, the input is [i, j].
    Args:
        image_size: the size of the image
    Return:
        2: index_dim.
        C: Tensor of shape (image_size**2, 2)
    """
    C = np.empty((image_size, image_size, 2), dtype=int)
    for i in range(image_size):
        for j in range(image_size):
            C[i, j] = [i-shift, j-shift]

    C = Tensor(C.reshape(-1,2))
    return 2, C

def gradient_clip(model:torch.nn.Module, approach:str='Global Norm Clipping', clip_value:int = 10, clip_factor:int = None):

    if approach == 'Global Norm Clipping':
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    elif approach == 'Value Clipping':
        for params in model.parameters():
            if params.grad is not None:
                # Clip the gradients to be within [-clip_value, clip_value]
                params.grad.data.clamp_(-clip_value, clip_value)

    elif approach == 'Adaptive Gradient Clipping':
        eps = 1e-3
        assert clip_factor != None
        for params in model.parameters():
            if params.grad is not None:
                # Compute the L2 norm of the gradient and the parameter
                grad_norm = params.grad.data.norm(2)
                param_norm = params.data.norm(2)
                
                # Compute the maximum allowed norm based on the parameter norm
                max_norm = param_norm * clip_factor
                
                # Compute the clip value (minimum between 1 and the allowed ratio)
                clip_value = max_norm / (grad_norm + eps)
                
                # Clip the gradient
                params.grad.data.mul_(clip_value.clamp(max=1.0))

    elif approach == '':
        # This is the case of NO gradient clipping. 
        pass

def mnist_preprocess(mnist_data:Tensor, method=None):
    '''
    Given original mnist_data and a method, return pre-processed data.
    '''
    if method == None:
        processed_mnist_data = mnist_data
        param_dict = None

    elif method == 'normalization':
        _mean = mnist_data.mean()
        _std = mnist_data.std()
        param_dict = {'mean': _mean, 'std': _std}
        processed_mnist_data = (mnist_data - _mean) / _std

    elif method == 'devide_by_255':
        # NOTE original data has range 0 - 255.
        processed_mnist_data = mnist_data / 255
        param_dict = None

    return processed_mnist_data, param_dict

def revserse_mnist_preprocess(processed_mnist_data, method, param_dict=None):
    '''
    Reverse function of mnist_preprocess. 
    '''
    if method == 'normalization':
        assert param_dict != None
        mnist_data = processed_mnist_data * param_dict['std'] + param_dict['mean']
    elif method == None:
        mnist_data = processed_mnist_data
    elif method == 'devide_by_255':
        mnist_data = processed_mnist_data * 255
        
    return mnist_data

def sample_from_multivariantgaussian(mean_tensor:Tensor, log_sigma_tensor:Tensor, monte_carlo_samples:int=10):
    '''
    Sample samples from several mean-field multivariable gaussian distribution (specified by mean and log-sigma).

    return:
        samples: (monte_carlo_samples, n_variables, n_dim) tensor. 
    '''
    assert len(mean_tensor.shape) == len(log_sigma_tensor.shape) == 2
    assert mean_tensor.shape == log_sigma_tensor.shape
    n_variables = mean_tensor.shape[0]
    n_dim = mean_tensor.shape[1]
    sigma_tensor = torch.exp(log_sigma_tensor)

    # storing samples ... 
    samples = torch.zeros(n_variables, monte_carlo_samples, n_dim)

    for i in range(n_variables):
        mu = mean_tensor[i]
        cov = torch.diag(sigma_tensor[i]**2)
        dist = MultivariateNormal(mu, cov)
        samples[i] = dist.sample((monte_carlo_samples,))

    samples = samples.transpose(0, 1)
    return samples

"""
from models_.gaussian_likelihood import GaussianLikelihood
def mc_pred_helper(model, likelihood: GaussianLikelihood, sample_X_tensor:Tensor, C_total, batch_index_X, batch_index_C):
    '''
    Helper function to get q(y_*) (prediction on evaluation set) when monte carlo integration of latent variable X is applied.
    Args:
        model: LVMOGP_SVI.
        likelihood: GaussianLikelihood.
        sample_X_tensor: the tensor return by function 'sample_from_multivariantgaussian'.
        C_total: 
        batch_index_X: pick indices of X (which outputs are considered?)
        batch_index_C: pick indices of C (which inputs are considered?)
    Return:
        list_grid_output_batches, contains several gaussian distributions.
    '''
    list_grid_output_batches = []
    
    for i in range(sample_X_tensor.shape[0]):
        sample_X = sample_X_tensor[i]
        sample_batch_X = sample_X[batch_index_X]
        sample_batch_C = C_total[batch_index_C]

        grid_output_batch = model(sample_batch_X, sample_batch_C) # q(f)
        # passing through likelihood.
        grid_output_batch = likelihood(grid_output_batch) # q(y) Gaussian Distributed
        list_grid_output_batches.append(grid_output_batch) 
    
    # TODO: this is wrong ... the average distribution is no longer gaussian .... 
    ave_mu, ave_cov = torch.zeros_like(grid_output_batch.loc), torch.zeros_like(grid_output_batch.loc)
    for output in list_grid_output_batches:
        ave_mu += output.loc
        ave_cov += output.stddev ** 2
    ave_mu  /= sample_X_tensor.shape[0]
    ave_cov /= (sample_X_tensor.shape[0] ** 2)

    average_grid_output_batches = MultivariateNormal(ave_mu, torch.diag(ave_cov))

    return list_grid_output_batches, average_grid_output_batches
"""
def helper_model_diagonsis(my_model, mode='all'):
    '''
    This is a helper function to help print out all parameter values inside the model -- for analyzing model...
    '''
    for name, param in my_model.named_parameters():
        print(name, param.size())
        print(param)

def neg_log_likelihood(Target:Tensor, GaussianMean:Tensor, GaussianVar:Tensor):
    '''
    Evaluate negative log likelihood on given i.i.d. targets, where likelihood function is 
    gaussian with mean GaussianMean variance GaussianVar.

    Return:
        nll: scalar
    '''
    assert Target.shape == GaussianMean.shape == GaussianVar.shape
    nll = 0.5 * torch.mean(torch.log(2 * torch.pi * GaussianVar) + (Target - GaussianMean)**2 / GaussianVar)
    return nll


def store_data_from_synth_reg(store_path:str, latents:Tensor, inputs:Tensor, target_data:Tensor, kernel_parameters:dict):
    '''
    create a sub-folder under given store_path to contain all files.
    '''
    n_inputs = inputs.shape[0]
    n_latents = latents.shape[0]
    assert int(n_inputs * n_latents) == target_data.shape[0]
    # create sub-folder
    new_folder_path = f'{store_path}/ninputs_{n_inputs}_nlatents_{n_latents}'
    os.makedirs(new_folder_path, exist_ok=True)
    # pump in tensors
    pd.DataFrame(latents.numpy()).to_csv(f'{new_folder_path}/latents.csv', index=False)
    pd.DataFrame(inputs.numpy()).to_csv(f'{new_folder_path}/inputs.csv', index=False)
    pd.DataFrame(target_data.numpy()).to_csv(f'{new_folder_path}/target_data.csv', index=False)

    # due to kernel_parameters contains tensors which are not ok to work with
    # we need to transform tensors to lists ....
    new_kernel_params = {}
    for key, value in kernel_parameters.items():
        new_kernel_params[key] = value.tolist()
    with open(f'{new_folder_path}/dictionary.json', 'w') as json_file:
        json.dump(new_kernel_params, json_file)

def collect_model_gradients(my_model):
    '''
    my_model contains several parts of parameters:
        variational_strategy: (vs)
            inducing_points_X
            inducing_points_C
            _variational_distribution.variational_mean
            _variational_distribution.chol_variational_covar_X
            _variational_distribution.chol_variational_covar_C
        X:
            q_mu
            q_log_sigma
        covar_module_X: (cov_X)
            raw_outputscale
            base_kernel.raw_lengthscale
        covar_module_C: (cov_C)
            raw_outputscale
            base_kernel.raw_lengthscale

    '''
    dict_grads, dict_values = {}, {}

    dict_grads['vs_mean'] = my_model.variational_strategy._variational_distribution.variational_mean.grad.detach().abs().mean()
    dict_grads['vs_chol_X'] = my_model.variational_strategy._variational_distribution.chol_variational_covar_X.grad.detach().abs().mean()
    dict_grads['vs_chol_C'] = my_model.variational_strategy._variational_distribution.chol_variational_covar_C.grad.detach().abs().mean()
    dict_grads['X_q_mu'] = my_model.X.q_mu.grad.detach().abs().mean()
    dict_grads['X_q_log_sigma'] = my_model.X.q_log_sigma.grad.detach().abs().mean()
    dict_grads['cov_X_raw_outputscale'] = my_model.covar_module_X.raw_outputscale.grad.detach().abs().mean()
    dict_grads['cov_X_raw_lengthscale'] = my_model.covar_module_X.base_kernel.raw_lengthscale.grad.detach().abs().mean()
    dict_grads['cov_C_raw_outputscale'] = my_model.covar_module_C.raw_outputscale.grad.detach().abs().mean()
    dict_grads['cov_C_raw_lengthscale'] = my_model.covar_module_C.base_kernel.raw_lengthscale.grad.detach().abs().mean()
    
    dict_values['vs_mean'] = my_model.variational_strategy._variational_distribution.variational_mean.detach().abs().mean()
    dict_values['vs_chol_X'] = my_model.variational_strategy._variational_distribution.chol_variational_covar_X.detach().abs().mean()
    dict_values['vs_chol_C'] = my_model.variational_strategy._variational_distribution.chol_variational_covar_C.detach().abs().mean()
    dict_values['X_q_mu'] = my_model.X.q_mu.detach().abs().mean()
    dict_values['X_q_log_sigma'] = my_model.X.q_log_sigma.detach().abs().mean()
    dict_values['cov_X_raw_outputscale'] = my_model.covar_module_X.raw_outputscale.detach().abs().mean()
    dict_values['cov_X_raw_lengthscale'] = my_model.covar_module_X.base_kernel.raw_lengthscale.detach().abs().mean()
    dict_values['cov_C_raw_outputscale'] = my_model.covar_module_C.raw_outputscale.detach().abs().mean()
    dict_values['cov_C_raw_lengthscale'] = my_model.covar_module_C.base_kernel.raw_lengthscale.detach().abs().mean()
    
    return dict_grads, dict_values

def compute_signal_to_noise_ratio(time_series:Tensor):
    '''
    Compute SNR for all time series.
    Arg:
        time_series: of shape (num_time_series, num_timeframes)
    Return:
        a list of SNR
    '''
    snr_list = []
    for i in range(time_series.shape[0]):
        # Calculate the mean (signal)
        signal_mean = time_series[i].mean()
        
        # Calculate the standard deviation (noise)
        noise_std = time_series[i].std()

        # Compute SNR. If noise_std is zero, handle the division by zero case.
        if noise_std <= 1e-4:
            snr = float('inf')  # Infinite SNR if there is no noise
        else:
            snr = signal_mean / noise_std

        # Convert to dB (20*log10(snr)) and append to list
        # We use 20*log10 for amplitude ratio. For power ratio, it would be 10*log10.
        snr_db = float(20 * np.log10(abs(snr)))
        snr_list.append(snr_db)

    # Sort the SNR list in descending order and get the sorted indices
    sorted_indices = np.argsort(snr_list)[::-1]

    return snr_list, sorted_indices.tolist()

################################################   Classification Utility Functions  ################################################

def clf_sample_f_index_everyoutput(my_model, clf_list:List, labels:Tensor, num_class_per_output=5, num_input_samples:int=100, re_index_latent_idxs=True):
    '''
    After feeding (all_outputs, all_classes, all_inputs, index_dim) of inputs, we will get f of shape (all_outputs, all_classes, all_inputs).
    This function subsamples indices of f.
    All outputs are preserved, only classes and inputs are subsampled.
    Args:
        my_model: an instance of LVMOGP_SVI, _get_batch_idx function is in use.
        clf_list: list of n_classes. for example, [20, 13, 17] means 3 outputs with 20, 13, 17 classes respectively.
        labels: of shape (n_inputs, n_outputs). labels[a][b] extracts the classification label for a+1 th input at b+1 th output. 
        num_class_per_output: how many classes we want during subsampling.
            TODO: different output has different num of classes
        num_input_samples: how many data samples we want duing subsampling.

    Return:
        batch_index_latent: of shape (num_outputs, num_class_per_output+1, num_input_samples)
        batch_index_inputs: of shape (num_outputs, num_class_per_output+1, num_input_samples)
    NOTE: 
        1. Same set of inputs for every output.
        2. Same number of classes are downsampled for every output, seems unresonable if # total classes vary a lot across outputs.
        3. The final index on the second dim of batch_index_inputs is true label of the corresponding (input, output) pair which is useful in the future.
    '''

    num_outputs = len(clf_list)
    input_samples = Tensor(my_model._get_batch_idx(num_input_samples, sample_X = False)).to(int)

    final_inputs_idxs = input_samples.unsqueeze(0).unsqueeze(0)
    final_inputs_idxs = final_inputs_idxs.expand(num_outputs, (num_class_per_output+1), num_input_samples)

    final_latent_idxs = torch.zeros(num_outputs, (num_class_per_output+1), num_input_samples)

    for i in range(num_input_samples):
        for j in range(num_outputs):
            curr_true_label_idx = int(labels[input_samples[i], j]) # classification label for i+1 th input at j+1 th output ; labels[final_inputs_idxs[j,0,i]][j]
            num_class_curr_output = clf_list[j]
            available_range = list(np.arange(num_class_curr_output)[np.arange(num_class_curr_output) != curr_true_label_idx]) 
            assert len(available_range) == num_class_curr_output - 1
            curr_class_idx_list = random.sample(available_range, num_class_per_output)
            curr_class_idx_list.append(curr_true_label_idx) # of length num_class_per_output + 1
            assert len(curr_class_idx_list) == num_class_per_output + 1
            
            final_latent_idxs[j,:,i] = Tensor(curr_class_idx_list)
    
    assert final_inputs_idxs.shape == final_latent_idxs.shape

    if not re_index_latent_idxs:
        return final_latent_idxs.to(int), final_inputs_idxs
    
    # Transform idx properly to better match slicing functionality from my_model.sample_latent_variable()
    # as 
    else:
        counter = 0
        for j in range(num_outputs):
            final_latent_idxs[j,...] += counter
            counter += clf_list[j]
        return final_latent_idxs.to(int), final_inputs_idxs

def Softmax_function(f_mean:Tensor, f_var:Tensor, num_MC_samples:int=50):
    '''
    Single output softmax funciton, i.e., given latent parameter values, we get probabilities for each class.
    The reparametrization trick is in use for Monte Carlo estimation ... 
    The methodology is from paper:
        <Scalable Gaussian Process Classification With Additive Noise for Non-Gaussian Likelihoods> 2022, Liu et al.
    
    Args:
        f_mean: of shape (n_test_samples, n_classes)
        f_var: of shape (n_test_samples, n_classes)
    
    Return:
        results_prob_mean: of shape (n_test_samples, n_classes)
        results_prob_var: of shape (n_test_samples, n_classes)
        results_decisions: of shape (n_test_samples)
        results_decisions_var: of shape (n_test_samples)
    '''
    n_test_samples, n_classes = f_mean.shape[0], f_mean.shape[1]
    # reparametrization trick for MC estimation!
    sample_f = f_mean.unsqueeze(-1).expand(-1, -1, num_MC_samples) + f_var.sqrt().unsqueeze(-1).expand(-1, -1, num_MC_samples) + torch.randn(n_test_samples, n_classes, num_MC_samples)
    assert sample_f.shape == torch.Size([n_test_samples, n_classes, num_MC_samples])
    exp_sample_f_term = sample_f.exp()
    exp_sample_f_sum_term = exp_sample_f_term.sum(1).unsqueeze(1).expand(-1, n_classes, -1) # sum over n_classes, then expand to proper size (for future use)
    softmax_ratios = exp_sample_f_term / exp_sample_f_sum_term

    results_prob_mean = softmax_ratios.mean(-1)
    results_prob_var = softmax_ratios.var(-1)
    results_decisions = results_prob_mean.argmax(1)
    results_decisions_var = results_prob_var.sum(1)

    return results_prob_mean, results_prob_var, results_decisions, results_decisions_var  
    
def MOMOC_predict(my_model, X_test:Tensor, clf_list:List, test_mini_batch:int=201, num_MC_samples:int=50, mode='All_Outputs'):
    '''
    MultiOutput MultiClass prediction given:
    Args:
        my_model: trained model.
        X_test: test input locations. of shape (n_test_samples, num_features).
        clf_list: list of # of classes (for every output). 
        mode: whether all output predictions of all test inputs are needed.
    Return:
        all_outputs_prob_mean, all_outputs_prob_var: List of tensors. length of List is n_outputs, shape of each tensor is (num_test_inputs, num_classes), num_classes might varies for different output.
        all_outputs_decisions, all_outputs_decisions_var: tensor of shape (n_test_samples, num_outputs)
    '''
    my_model.eval()
    n_outputs = len(clf_list)
    n_test_samples = X_test.shape[0]
    X_q_mu = my_model.X.q_mu.detach()
    n_latent = X_q_mu.shape[0]

    if mode == 'All_Outputs':

        # NOTE we would like two equal length 1d tensor for extracting elements in X_q_mu and X_test and feed them to my_model.
        # the length is n_latent * n_test_samples

        # ------------------------------------------------------------------------------------------------------------------------------
        # * we will get prediction results input by input. i.e. the first batch of outputs (of length n_latent) are for first test input, followed by
        # second input, third input and so on. 

        # * for first n_latent of prediction outputs, they are ordered task by task. i.e. first batch of them (of length clf_list[0]) are for first task,
        # followed by second task (of first input) and so on.

        # * by doing this, we may have very very long tensor which might not available for feeding into the model entirely at once. the solution here is
        # applying mini-batching ...
        # ------------------------------------------------------------------------------------------------------------------------------
        
        all_latent_index = Tensor(np.arange(n_latent)).repeat(n_test_samples)
        all_input_index = Tensor([i for i in range(n_test_samples) for _ in range(n_latent)])

        assert all_input_index.shape == all_latent_index.shape
        assert (all_latent_index[:n_latent]).var() != 0.0
        assert (all_input_index[:n_latent]).var() == 0.0 # all same elements

        test_mini_batch = test_mini_batch
        pred_results_mean = torch.zeros(int(n_latent * n_test_samples))
        pred_results_var = torch.zeros(int(n_latent * n_test_samples))
        test_continue = True
        start_idx = 0
        end_idx = test_mini_batch
        while test_continue:
            batch_latent = X_q_mu[all_latent_index[start_idx:end_idx].to(int)] # TODO: only mean are taken into consideration ...
            batch_inputs = X_test[all_input_index[start_idx:end_idx].to(int)]
            batch_output = my_model(batch_latent, batch_inputs) # q(f): batch prediction
            pred_results_mean[start_idx:end_idx] = batch_output.loc.detach()
            pred_results_var[start_idx:end_idx] = batch_output.variance.detach()
            # pred_results_mean.append(batch_output.loc.detach().tolist()) # This will leads to list of list, which is not desirable...
            # pred_results_var.append(batch_output.variance.detach().tolist())

            if end_idx < n_latent * n_test_samples:
                start_idx += test_mini_batch
                end_idx += test_mini_batch
                end_idx = min(end_idx, int(n_latent * n_test_samples))
            else:
                test_continue = False

        assert len(pred_results_mean) == len(pred_results_var) == int(n_latent * n_test_samples)
        
        pred_results_mean_tensor = Tensor(pred_results_mean).reshape(n_test_samples, n_latent)
        pred_results_var_tensor  = Tensor(pred_results_var).reshape(n_test_samples, n_latent)
        
        # NOTE: n_latent is the number of all latents for all outputs.
        # ------------------------------------------------------------------------------------------------------------------------------
        # We need to chunk pred_results_mean_tensor and pred_results_var_tensor into n_outputs tensors, each of them can be fed into previously defined
        # Softmax_function for getting predictions.
        # ------------------------------------------------------------------------------------------------------------------------------
        split_mean_tensors = torch.split(pred_results_mean_tensor, clf_list, dim=-1) # tuple of tensors
        split_var_tensors = torch.split(pred_results_var_tensor, clf_list, dim=-1)
        assert n_outputs == len(split_mean_tensors) == len(split_var_tensors)

        all_outputs_prob_mean = []
        all_outputs_prob_var = []
        all_outputs_decisions = torch.zeros(n_test_samples, n_outputs)
        all_outputs_decisions_var = torch.zeros(n_test_samples, n_outputs)


        for i in range(n_outputs):
            _prob_mean, _prob_var, _decisions, _decisions_var = Softmax_function(f_mean=split_mean_tensors[i], f_var=split_var_tensors[i], num_MC_samples=num_MC_samples)
            all_outputs_prob_mean.append(_prob_mean) # List of tensors
            all_outputs_prob_var.append(_prob_var)   # List of tensors
            all_outputs_decisions[:,i] = _decisions
            all_outputs_decisions_var[:,i] = _decisions_var

    return all_outputs_prob_mean, all_outputs_prob_var, all_outputs_decisions, all_outputs_decisions_var

def MOMC_classification_eval(predictions:Tensor, labels:Tensor) -> Dict[str, List[float]]:
    '''
    This function evaluate classification performance for every output given predictions tensor and labels tensor.
    
    Evaluation metrices are: precisions weighted, recall weighted and F1 weighted.

    Args:
        predictions: Tensor of shape (num_samples, num_outputs)
        labels: Tensor of shape (num_samples, num_outputs)
    
    Return:
        eval_results: dictionary, keys: Precision_Weighted, Recall_Weighted, F1_weighted.
                    for each key, the dict contains a list of performance for all outputs.
    '''
    eval_results = {'Precision_Weighted': [], 'Recall_Weighted': [], 'F1_Weighted': []}  

    # Iterate through each output
    for output in range(labels.size(1)):
        # Extract predictions and labels for the current output
        preds = predictions[:, output]
        lbls = labels[:, output]

        # Calculate weighted metrics for the current output
        precision = precision_score(lbls, preds, average='weighted')
        recall = recall_score(lbls, preds, average='weighted')
        f1 = f1_score(lbls, preds, average='weighted')

        # Store the results
        eval_results['Precision_Weighted'].append(precision)
        eval_results['Recall_Weighted'].append(recall)
        eval_results['F1_Weighted'].append(f1)

    return eval_results











################################################   Classification Synthetic Data Experi  ################################################



################################################   CMU Motion Capture Dataset  ################################################


def process_motion_capture_data(file_path, num_times=309):
    # Dictionary with the number of dimensions for each body part
    # return: np.ndarray; of shape (62, #time_points)
    # i.e. 62 outputs, each output has #time_points input locations.
    dimensions_dict = {
        'root': 6, 'lowerback': 3, 'upperback': 3, 'thorax': 3, 
        'lowerneck': 3, 'upperneck': 3, 'head': 3, 'rclavicle': 2, 
        'rhumerus': 3, 'rradius': 1, 'rwrist': 1, 'rhand': 2, 
        'rfingers': 1, 'rthumb': 2, 'lclavicle': 2, 'lhumerus': 3, 
        'lradius': 1, 'lwrist': 1, 'lhand': 2, 'lfingers': 1, 
        'lthumb': 2, 'rfemur': 3, 'rtibia': 1, 'rfoot': 2, 
        'rtoes': 1, 'lfemur': 3, 'ltibia': 1, 'lfoot': 2, 
        'ltoes': 1
    }

    # Calculate the total number of dimensions
    total_dimensions = sum(dimensions_dict.values())

    # Initialize the numpy array
    data_array = np.zeros((total_dimensions, num_times))

    # Open the file and process each line
    with open(file_path, 'r') as file:
        # Skip the first three lines (header)
        for _ in range(3):
            next(file)

        current_timestamp = 0
        current_dimension = 0
        for line in file:
            if line.strip().isdigit():
                # New timestamp line found
                current_timestamp = int(line.strip()) - 1
                current_dimension = 0  # Reset dimension counter for each timestamp
            else:
                # Process body part data
                parts = line.strip().split()
                body_part = parts[0]
                num_dims = dimensions_dict[body_part]
                values = np.array([float(v) for v in parts[1:1+num_dims]])
                data_array[current_dimension:current_dimension+num_dims, current_timestamp] = values
                current_dimension += num_dims

    return data_array


################################################   Prediction via Integration of Uncertain Latents  ################################################

def prepare_common_background_info(my_model, config):
    '''Prepare all values of a dict called common_background_information, which being used in integration_prediction_func'''
    
    def _cholesky_factor(induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)
    
    K_uu_latent = my_model.covar_module_latent(my_model.variational_strategy.inducing_points_latent.data).to_dense().data.to(torch.float64)
    # K_uu_latent_inv = torch.linalg.solve(K_uu_latent, torch.eye(K_uu_latent.size(-1)).to(torch.float64))
    K_uu_input = my_model.covar_module_input(my_model.variational_strategy.inducing_points_input.data).to_dense().data.to(torch.float64)
    # K_uu_input_inv = torch.linalg.solve(K_uu_input, torch.eye(K_uu_input.size(-1)).to(torch.float64))

    K_uu = KroneckerProductLinearOperator(K_uu_latent, K_uu_input).to_dense().data
    # chol_K_uu_inv_t = _cholesky_factor_latent(KroneckerProductLinearOperator(K_uu_latent_inv, K_uu_input_inv)).to_dense().data.t()
    chol_K_uu_inv_t = KroneckerProductLinearOperator(
            torch.linalg.solve( _cholesky_factor(K_uu_latent).to_dense().data, torch.eye(K_uu_latent.size(-1)).to(torch.float64)),
            torch.linalg.solve( _cholesky_factor(K_uu_input).to_dense().data, torch.eye(K_uu_input.size(-1)).to(torch.float64)),
        ).to_dense().data.t()
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    chol_covar_latent_u = my_model.variational_strategy._variational_distribution.chol_variational_covar_latent.data.to(torch.float64)
    covar_latent_u = CholLinearOperator(chol_covar_latent_u).to_dense()
    chol_covar_input_u = my_model.variational_strategy._variational_distribution.chol_variational_covar_input.data.to(torch.float64)
    covar_input_u = CholLinearOperator(chol_covar_input_u).to_dense()

    covar_u = KroneckerProductLinearOperator(covar_latent_u, covar_input_u).to_dense().data

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    common_background_information = {
                        'K_uu': K_uu.data,
                        'chol_K_uu_inv_t': chol_K_uu_inv_t.data, 
                        'm_u': my_model.variational_strategy._variational_distribution.variational_mean.data,
                        'Sigma_u': covar_u.data,
                        'A': chol_K_uu_inv_t @ (covar_u - torch.eye(covar_u.shape[0])) @ chol_K_uu_inv_t.t(),
                        'var_H': my_model.covar_module_latent.outputscale.data,
                        'var_X': my_model.covar_module_input.outputscale.data,
                        'W': my_model.covar_module_latent.base_kernel.lengthscale.data.reshape(-1)**2
                        }
    '''
    chol_K_uu_inv_t: inverse of K_uu matrix, of shape (M_H * M_X, M_H * M_X)
    m_u: mean of the variational distribution
    Sigma_u: covariance matrix of the variational distribution
    A: chol_K_uu_inv_t (Sigma_u - K_uu) chol_K_uu_inv_t.T
    var_H: 
    var_X: 
    W: vector; containing all lengthscales in the RAD kernel
    c: constant
    '''
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    c = (2 * torch.pi)**(config['latent_dim'] / 2) * common_background_information['var_H'] * common_background_information['W'].sqrt().prod()
    common_background_information['constant_c'] = c

    return common_background_information

def integration_prediction_func(test_input, output_index, my_model, common_background_information, config):

    input_K_f_u = my_model.covar_module_input(test_input, my_model.variational_strategy.inducing_points_input.data).to_dense().data
    input_K_u_f_K_f_u = input_K_f_u.t() @ input_K_f_u

    data_specific_background_information = {
        'm_plus': my_model.X.q_mu.data[output_index],
        'Sigma_plus': 1.0 * my_model.X.q_log_sigma.exp().square().data[output_index],
        'input_K_f_u': input_K_f_u, 
        'input_K_u_f_K_f_u': input_K_u_f_K_f_u,
        'expectation_K_uu': None
    }
    
    # helper functions -----------------------------------------------------------------------------------------------------------------------
    def multivariate_gaussian_pdf(x, mu, cov):
        '''cov is a vector, representing all elements in the diagonal matrix'''
        k = mu.size(0)
        cov_det = cov.prod()
        cov_inv = torch.diag(1.0 / cov)
        norm_factor = torch.sqrt((2 * torch.pi) ** k * cov_det)

        x_mu = x - mu
        result = torch.exp(-0.5 * x_mu @ cov_inv @ x_mu.t()) / norm_factor
        return result.item()

    def G(h:Tensor, common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):

        mu = data_specific_background_information['m_plus']
        cov_diag = data_specific_background_information['Sigma_plus'] + common_background_information['W']
        result = multivariate_gaussian_pdf(h, mu, cov_diag)
        return common_background_information['constant_c'] * result

    def R(h_1:Tensor, h_2:Tensor, common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        mu_1 = h_2
        cov_diag_1 = 2 * common_background_information['W']
        mu_2 = (h_1 + h_2) / 2
        cov_diag_2 = 0.5 * common_background_information['W'] + data_specific_background_information['Sigma_plus']
        result1 = multivariate_gaussian_pdf(h_1, mu_1, cov_diag_1)
        result2 = multivariate_gaussian_pdf(data_specific_background_information['m_plus'], mu_2, cov_diag_2)
        return (common_background_information['constant_c'] ** 2 ) * result1 * result2
    
    def expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        result_ = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_f_u'].reshape(1, -1), data_specific_background_information['input_K_f_u'].reshape(1, -1)).to_dense().data 
        result_ = result_ @ common_background_information['chol_K_uu_inv_t'].to(result_.dtype) @ common_background_information['m_u'].to(result_.dtype)
        return result_
        
    def expectation_lambda_square(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        result_ = common_background_information['m_u']
        _result = result_ @ common_background_information['chol_K_uu_inv_t'].t().to(result_.dtype)
        interm_term = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_u_f_K_f_u'], data_specific_background_information['input_K_u_f_K_f_u']).to_dense().data
        result_ = _result @ interm_term.to(result_.dtype) @ _result.t()
        # result_ = result_ @ common_background_information['chol_K_uu_inv_t'].to(result_.dtype) @ common_background_information['m_u']

        if data_specific_background_information['expectation_K_uu'] == None:
            data_specific_background_information['expectation_K_uu'] = interm_term
        return result_
        
    def expectation_gamma(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        result_ = common_background_information['var_H'] * common_background_information['var_X']

        if data_specific_background_information['expectation_K_uu'] == None:
            data_specific_background_information['expectation_K_uu'] = KroneckerProductLinearOperator(data_specific_background_information['expectation_latent_K_u_f_K_f_u'], \
                                                                                                    data_specific_background_information['input_K_u_f_K_f_u']).to_dense().data

        return result_ + (common_background_information['A'] * data_specific_background_information['expectation_K_uu']).sum()
    
    def integration_predictive_mean(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        return expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information)


    def integration_predictive_var(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information):
        return expectation_lambda_square(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information) \
            + expectation_gamma(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information) \
            - expectation_lambda(common_background_information=common_background_information, data_specific_background_information=data_specific_background_information)**2
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    expectation_latent_K_f_u = Tensor([G(my_model.variational_strategy.inducing_points_latent.data[i]).item() for i in range(config['n_inducing_latent'])])
    expectation_latent_K_u_f_K_f_u = Tensor([R(my_model.variational_strategy.inducing_points_latent.data[i], my_model.variational_strategy.inducing_points_latent.data[j]).item() \
                                            for j in range(config['n_inducing_latent']) for i in range(config['n_inducing_latent'])]).reshape(config['n_inducing_latent'], config['n_inducing_latent'])

    data_specific_background_information['expectation_latent_K_f_u'] = expectation_latent_K_f_u
    data_specific_background_information['expectation_latent_K_u_f_K_f_u'] = expectation_latent_K_u_f_K_f_u

    return integration_predictive_mean(data_specific_background_information=data_specific_background_information), \
           integration_predictive_var(data_specific_background_information=data_specific_background_information)



################################################   Post Train Model Analysis  ################################################

## Inference:

def pred4all_outputs_inputs(my_model, my_likelihood, data_inputs, config, common_background_information=None, approach='mean', not4visual=True, n_data4visual=0):
    '''
    Perform inference on all inputs and outputs pairs, two possible approaches: mean and integration. 
    '''
    my_model.eval()
    my_likelihood.eval()

    if not4visual:
        all_index_latent = np.array([[i]*config['n_input'] for i in range(config['n_outputs'])]).reshape(-1).tolist() 
        all_index_input = [i for i in range(config['n_input'])] * config['n_outputs'] 
        len_outputs = len(all_index_latent)

    else:
        assert n_data4visual > 0
        all_index_latent = np.array([[i]*n_data4visual for i in range(config['n_outputs'])]).reshape(-1).tolist() 
        all_index_input = [i for i in range(n_data4visual)] * config['n_outputs'] 
        len_outputs = len(all_index_latent)

    # used to store all predictions
    all_pred_mean = torch.zeros(len_outputs)
    all_pred_var = torch.zeros(len_outputs)

    if approach == 'mean':
        # access the latent variables for all outputs
        all_mean_outputs = my_model.X.q_mu.data

        test_mini_batch_size = 1000
        test_continue = True
        test_start_idx = 0
        test_end_idx = test_mini_batch_size

        # iteratively inference
        while test_continue:
            batch_latent = all_mean_outputs[all_index_latent[test_start_idx:test_end_idx]]
            batch_input = data_inputs[all_index_input[test_start_idx:test_end_idx]]
            batch_output = my_likelihood(my_model(batch_latent, batch_input))
            # store predictions (mean and var) for current batch
            all_pred_mean[test_start_idx:test_end_idx] = batch_output.loc.detach().data
            all_pred_var[test_start_idx:test_end_idx] = batch_output.variance.detach().data

            if test_end_idx < len_outputs:
                test_start_idx += test_mini_batch_size
                test_end_idx += test_mini_batch_size
                test_end_idx = min(test_end_idx, len_outputs)
            else:
                test_continue = False

    elif approach == 'integration':
        # iteratively inference
        for idx in trange(len_outputs, leave=True):
            curr_latent_index = all_index_latent[idx]
            curr_input = data_inputs[all_index_input[idx]].reshape(-1)
            curr_pred_mean, curr_pred_var = integration_prediction_func(test_input=curr_input,
                                                                        output_index=curr_latent_index,
                                                                        my_model=my_model,
                                                                        common_background_information=common_background_information,
                                                                        config=config)
            all_pred_mean[idx] = curr_pred_mean
            all_pred_var[idx] = curr_pred_var + my_likelihood.noise.data
            
    return all_pred_mean, all_pred_var

## Evaluation on single picked output

def evaluate_on_single_output(
        function_index,
        data_inputs,
        data_Y_squeezed,
        ls_of_ls_train_input,
        ls_of_ls_test_input,
        train_sample_idx_ls,
        test_sample_idx_ls,
        all_pred_mean,
        all_pred_var,
        n_data4visual,
        all_pred_mean4visual,
        all_pred_var4visual
    ):
    # Pick the index of the funtion to show
    # function_index = 982

    performance_dirct = {}
    train_input = data_inputs[ls_of_ls_train_input[function_index]]
    train_start = 0
    for i in range(function_index):
        train_start += len(ls_of_ls_train_input[i]) # don't assume every output has the same length of inputs
    train_end = train_start + len(ls_of_ls_train_input[function_index])
    train_target = data_Y_squeezed[train_sample_idx_ls][train_start:train_end]
    train_predict = all_pred_mean[train_sample_idx_ls][train_start:train_end]
    train_rmse_ = (train_target - train_predict).square().mean().sqrt()
    train_nll_ = neg_log_likelihood(train_target, all_pred_mean[train_sample_idx_ls][train_start:train_end], all_pred_var[train_sample_idx_ls][train_start:train_end])
    performance_dirct['train_rmse'] = train_rmse_
    performance_dirct['train_nll'] = train_nll_

    test_input = data_inputs[ls_of_ls_test_input[function_index]]
    test_start = 0
    for j in range(function_index):
        test_start += len(ls_of_ls_test_input[i])
    test_end = test_start + len(ls_of_ls_test_input[function_index])
    test_target = data_Y_squeezed[test_sample_idx_ls][test_start:test_end]
    test_predict = all_pred_mean[test_sample_idx_ls][test_start:test_end]
    test_rmse_ = (test_predict - test_target).square().mean().sqrt()
    test_nll_ = neg_log_likelihood(test_target, all_pred_mean[test_sample_idx_ls][test_start:test_end], all_pred_var[test_sample_idx_ls][test_start:test_end])
    performance_dirct['test_rmse'] = test_rmse_
    performance_dirct['test_nll'] = test_nll_

    gp4visual_start = n_data4visual * function_index
    gp4visual_end = n_data4visual * (function_index + 1)
    gp_pred_mean = all_pred_mean4visual[gp4visual_start:gp4visual_end]
    gp_pred_std = all_pred_var4visual.sqrt()[gp4visual_start:gp4visual_end]

    return train_input, train_target, test_input, test_target, gp_pred_mean, gp_pred_std, performance_dirct