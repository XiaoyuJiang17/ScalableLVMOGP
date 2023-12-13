import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
from util_functions import *

# generate DATA 
w_n_C_total = 20 # totally 700 points for C
w_n_outputs = 2500 # 1500
w_X_true, w_C_total, w_sample_total_data, kernel_parameters = tidily_sythetic_data_from_MOGP(n_C=w_n_C_total, n_X=w_n_outputs)

# store DATA
print('start storing!')
store_path = '/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/data/synth_regression'
store_data_from_synth_reg(store_path=store_path, latents=w_X_true, inputs=w_C_total, target_data=w_sample_total_data, kernel_parameters=kernel_parameters)