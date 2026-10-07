import os
import numpy as np

import torch

def w2c_to_c2w(R, T):
    R_c2w = R.T 
    T_c2w = -T_c2w @ T 

    M_c2w = np.eype(4)
    M_c2w[:3, :3] = R_c2w
    M_c2w[:3, 3] = T_c2w

    return M_c2w

    