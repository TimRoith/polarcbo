r"""
Scheduler
==========

This module implements the :math:`\beta`-schedulers employed in the conensuse schemes.

"""

import numpy as np
from scipy.special import logsumexp

class beta_exponential():
    def __init__(self, opt, beta=1.0, beta_max = 100000., r=1.0):
        self.opt = opt
        self.beta = beta
        self.beta_max = beta_max
        self.r = r
    
    def __call__(self):
        return self.opt.beta
    
    def update(self):
        beta = self.opt.beta
        self.opt.beta = min(self.r * beta, self.beta_max)
    
    


# class for beta_eff scheduler
class beta_eff():
    def __init__(self, opt, eta=1.0, beta_max=1e5, factor=1.05):
        self.opt = opt
        self.eta = eta
        self.beta_max = beta_max
        self.J_eff = 1.0
        self.factor=factor
    
    def __call__(self):
        return self.opt.beta
    
    def update(self,):
        beta = self.opt.beta
        
        term1 = logsumexp(-beta*self.opt.V(self.opt.x))
        term2 = logsumexp(-2*beta*self.opt.V(self.opt.x))
        self.J_eff = np.exp(2*term1-term2)
        
        #w = np.exp(-beta * self.opt.V(self.opt.x))
        #self.J_eff = np.sum(w)**2/max(np.linalg.norm(w)**2,1e-7)
        
        if self.J_eff >= self.eta * self.opt.num_particles:
            self.opt.beta = min(beta*self.factor, self.beta_max)
        else:
            pass
            #self.opt.beta /= self.factor