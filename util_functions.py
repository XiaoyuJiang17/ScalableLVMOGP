import numpy as np
import torch
import matplotlib.pyplot as plt
from gpytorch.kernels import ScaleKernel, RBFKernel
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Tensor
from torch.distributions import MultivariateNormal
import csv
import random

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

def  plot_traindata_testdata_fittedgp(train_X: Tensor, train_Y: Tensor, test_X: Tensor, test_Y: Tensor, gp_X: Tensor, gp_pred_mean: Tensor, gp_pred_std: Tensor, inducing_points_X: Tensor, n_inducing_C:int=15):
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
    plt.scatter(inducing_points_X, [plt.gca().get_ylim()[0]] * n_inducing_C, color='black', marker='^', label='Inducing Locations')


    # Plot GP predictions as a line
    plt.plot(gp_X, gp_pred_mean_np, 'k', lw=1, zorder=9)
    plt.fill_between(gp_X, gp_pred_mean_np - 1.96 * gp_pred_std_np, gp_pred_mean_np + 1.96 * gp_pred_std_np, alpha=0.2, color='k')

    plt.legend()
    plt.title("Train/Test Data and Fitted GP")
    plt.tight_layout()

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

################################################  Synthetic Data Expri  ################################################
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
        default_kernel_parameters = {'X_raw_outputscale': torch.tensor(0.0), 'X_raw_lengthscale': torch.tensor([[0.1 for _ in range(latent_dim)]]),
                                     'C_raw_outputscale': torch.tensor(0.9), 'C_raw_lengthscale': torch.tensor([[0.1 for _ in range(index_dim)]])}

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

    covar_module_X.raw_outputscale.data = default_kernel_parameters['X_raw_outputscale'].to(covar_module_X.device)
    covar_module_X.base_kernel.raw_lengthscale.data = default_kernel_parameters['X_raw_lengthscale'].to(covar_module_X.device) 
    covar_module_C.raw_outputscale.data = default_kernel_parameters['C_raw_outputscale'].to(covar_module_C.device)
    covar_module_C.base_kernel.raw_lengthscale.data = default_kernel_parameters['C_raw_lengthscale'].to(covar_module_C.device)

    covar_X = covar_module_X(X)
    covar_C = covar_module_C(C)
    # K + sigma^2 * I 
    covar_final = KroneckerProductLinearOperator(covar_X, covar_C).add_jitter(noise_scale).to_dense().detach()
    mean_final = Tensor([0. for _ in range(n_C * n_X)])

    dist = MultivariateNormal(mean_final, covar_final)
    sample = dist.sample()

    return X, C, sample, default_kernel_parameters

# TODO: non-tidily generating data from MOGP.

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
