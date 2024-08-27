import gymnasium as gym
import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)

