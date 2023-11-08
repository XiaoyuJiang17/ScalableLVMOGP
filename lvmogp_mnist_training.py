import yaml
import mnist
import torch
from torch import Tensor
from lvmogp_svi import LVMOGP_SVI
from gaussian_likelihood import GaussianLikelihood
from variational_elbo import VariationalELBO
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
from util_functions import *

# Load hyperparameters from .yaml file
with open('config_mnist_expri.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Assign hyperparameters from yaml file
root_path = config['root_path']
preprocessing_method = config['preprocessing_method']
num_train_data = config['num_train_data']
latent_dim = config['latent_dim']
n_inducing_C = config['n_inducing_C']
n_inducing_X = config['n_inducing_X']
pca = config['pca']
num_forbidden = config['num_forbidden']
learning_rate = config['learning_rate']
schduler_step_size = config['schduler_step_size']
schduler_gamma = config['schduler_gamma']
model_path = config['model_path']
likelihood_path = config['likelihood_path']
n_train_iterations = config['n_train_iterations']
train_batch_size_X = config['train_batch_size_X']
train_batch_size_C = config['train_batch_size_C']
gradient_clip_approach = config['gradient_clip_approach']

# Other parameters...
mnist_training_data = Tensor(mnist.train_images()[:num_train_data])
mnist_training_data, param_dict = mnist_preprocess(mnist_training_data, method=preprocessing_method)
print('Preprocessing params stored in dict:', param_dict)
n_X = mnist_training_data.shape[0]
n_C = int(mnist_training_data.shape[-2] * mnist_training_data.shape[-1]) # also data_dim
n_total = n_X * n_C
index_dim, C = gene_2dimage_inputs(mnist_training_data.shape[-2])
Y_train = mnist_training_data.reshape(-1)
forbidden_pairs = proper_gene_forbidden_pairs(n_X=n_X, n_C=n_C, num_forbidden=num_forbidden)
forbidden_pairs_file_path = f'{root_path}/forbidden_pairs.csv'
with open(forbidden_pairs_file_path, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(forbidden_pairs)

# Specify model, likelihood and training objective.
my_model = LVMOGP_SVI(n_X, n_C, index_dim, latent_dim, n_inducing_C, n_inducing_X, Y_train.reshape(n_X, -1), pca=pca)
likelihood = GaussianLikelihood() # how many outputs
mll = VariationalELBO(likelihood, my_model, num_data=n_total)

# Optimizer and Scheduler
optimizer = torch.optim.Adam([ # TODO: tune the choice of optimizer: SGD...
    {'params': my_model.parameters()},
    {'params': likelihood.parameters()}
], lr=learning_rate)
scheduler = StepLR(optimizer, step_size=schduler_step_size, gamma=schduler_gamma)

# Initialize inducing points in C space
selected_indices = torch.randperm(C.shape[0])[:n_inducing_C]
assert C[selected_indices].shape == torch.Size([n_inducing_C, index_dim])
my_model.variational_strategy.inducing_points_C.data = C[selected_indices]

# Training!
loss_list = []
iterator = trange(n_train_iterations, leave=True)

my_model.train()
for i in iterator: 
    batch_index_X, batch_index_C = proper_sample_index_X_and_C_(my_model, train_batch_size_X, train_batch_size_C, forbidden_pairs_file_path)
    # batch_index_X, batch_index_C = proper_sample_index_X_and_C(my_model, train_batch_size_X, train_batch_size_C, forbidden_pairs)
    # core code is here 
    optimizer.zero_grad()
    sample_X = my_model.sample_latent_variable()  # a full sample returns latent x across all n_X TODO: more efficient?
    sample_batch_X = sample_X[batch_index_X]
    sample_batch_C = C[batch_index_C]
    output_batch = my_model(sample_batch_X, sample_batch_C) # q(f)
    batch_index_Y = inhomogeneous_index_of_batch_Y(batch_index_X, batch_index_C, n_X, n_C)
    loss = -mll(output_batch, Y_train[batch_index_Y]).sum()
    loss_list.append(loss.item())
    iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
    loss.backward()

    # Gradient Clipping. Try Many Different Approaches.
    gradient_clip(my_model, approach=gradient_clip_approach, clip_value=10)
    gradient_clip(likelihood, clip_value=1)

    optimizer.step()
    scheduler.step()

# Save model, likelihood. Save and plot training losses.
loss_file_path = f'{root_path}/loss.csv'
with open(loss_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(loss_list)
plot_loss_and_savefig(loss_list, root_path)

torch.save(my_model.state_dict(), f'{model_path}/model_weights2.pth')
torch.save(likelihood.state_dict(), f'{likelihood_path}/likelihood_weights2.pth')