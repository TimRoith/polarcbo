import numpy as np

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import test_functions as tf
import utils as ut

sys.path.pop(0)

#%%
def get_conf_2d_Ackley():
    conf = ut.config(T = 3000, tau=0.01, num_particles =30)
    
    cs = np.array([[-1,0],[2,1],[-1,-2]])
    alphas = np.array([1,3,7])
    conf.V = tf.Ackley_multimodal(alpha=alphas,z=cs)
    
    conf.minima=-cs
    
    conf.x_max = 3
    conf.x_min = -3
    conf.random_seed = 309
    conf.d = 2
    conf.beta = 30.
    conf.sigma = 0.6
    conf.kappa = 10.5
    conf.heavy_correction = True
    conf.noise = ut.normal_noise(tau=conf.tau)
    conf.beta_schedule = ut.beta_eff
    conf.eta = 0.0
    conf.factor = 1.01
    
    conf.noise = ut.normal_noise(tau=conf.tau)
    conf.sigma = 0.6
    conf.kappa = 1000.0
    conf.heavy_correction = True
    conf.factor = 1.00
    
    return conf

def get_conf_20d_Ackley():
    conf = ut.config(T = 3000, tau=0.01, num_particles =30)
    conf.sigma = 8.0
    conf.d = 20
    
    z = np.ones((2,conf.d))
    z[1,:] *= -1
    
    z *=2
    
    
    alphas = np.array([1,3])
    conf.V = tf.Ackley_multimodal(alpha=alphas,z=z)
    
    conf.minima=-z
    
    conf.x_max = 3
    conf.x_min = -3
    conf.random_seed = 309
    
    conf.beta = 30.
    conf.kappa = 0.5
    conf.heavy_correction = False
    conf.noise = ut.comp_noise(tau=conf.tau)
    conf.beta_schedule = ut.beta_eff
    conf.eta = 0.0
    conf.factor = 1.01
    
    return conf
