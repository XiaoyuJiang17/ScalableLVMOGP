
25/11/2023

lvmogp_synth_data_expri.ipynb contains lvmogp-svi model of 2 versions: with and without missing values. 
ind_gp_synth_data_expri.ipynb contains independent GP models (one gp for each dimension), also of two versions with and without missing values.

26/11/2023

util_functions.py add random_seed implementation to ensure reproductivity.

28/11/2023

Two files are renamed: ind_gp_synth_data_expri_w.o._runall.ipynb and ind_gp_synth_data_expri_w.o.ipynb
Two files are created: ind_gp_synth_data_expri_w_runall.ipynb and ind_gp_synth_data_expri_w.ipynb

30/11/2023

re-organize files.

20/12/2023

h_mnist_pixel_as_output is different from h-mnist.
h_mnist_pixel_as_output: every pixel modelled as an output. (data plz see healing_mnist_data_v2)
h-mnist: every picture modelled as an output. (data plz see healing_mnist_data)

29/12/2023

kronecker_variational_strategy.py is updated, no inverse of kronecker product is implemented, which is replaced by kronecker product of two inverses.

kronecker_variational_strategy_.py is the original implementation.

03/01/2024

In h_mnist_pixel_as_output folder,
lvmogp_all_dims.ipynb are code to model all dims of mnist image,
lvmogp_selected_dims.ipynb are code to only model selected dims. 

7/2/2024
The saved model end with 2 is the current best model in spatio-temporal dataset. We keep that for future use.
