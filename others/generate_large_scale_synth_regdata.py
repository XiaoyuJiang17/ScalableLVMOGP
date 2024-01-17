import sys
sys.path.append('/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/')
from util_functions import *

# generate DATA 
w_n_C_total = 50
w_n_outputs = 5000 # 1500

# You can choose which approach to use:
# w_X_true, w_C_total, w_sample_total_data, kernel_parameters = tidily_sythetic_data_from_MOGP(n_C=w_n_C_total, n_X=w_n_outputs)
w_X_true, w_C_total, w_sample_total_data, kernel_parameters = tidily_sythetic_data_from_MOGP_smartly(n_C=w_n_C_total, n_X=w_n_outputs)
# store DATA
print('start storing!')
store_path = '/Users/jiangxiaoyu/Desktop/All Projects/GPLVM_project_code/data/synth_regression/smartly_generated'
store_data_from_synth_reg(store_path=store_path, latents=w_X_true, inputs=w_C_total, target_data=w_sample_total_data, kernel_parameters=kernel_parameters)