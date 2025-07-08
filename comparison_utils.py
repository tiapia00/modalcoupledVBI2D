import numpy as np
from scipy.interpolate import interp1d

def get_err(matlab_data: np.ndarray, my_data: np.ndarray):
    err = np.mean(np.abs(matlab_data - my_data))/np.max(np.abs(my_data-matlab_data))
    return err

def interpolate_mat(x_mat, y_mat, x_eval):
    y_interp = interp1d(x_mat, y_mat)
    y_eval = y_interp(x_eval)
    return y_eval
