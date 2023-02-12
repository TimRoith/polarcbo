import numpy as np
from scipy.special import logsumexp

from .pdyn import ParticleDynamic

#%% CBO
class CBO(ParticleDynamic):
    def __init__(self,x, V, noise,\
                 beta = 1.0, noise_decay=0.0, diff_exp=1.0,\
                 tau=0.1, sigma=1.0, lamda=1.0, M=None,\
                 overshoot_correction=False, heavi_correction=False):
        
        super(CBO, self).__init__(x, V, beta = beta)
        
        # additional parameters
        self.noise_decay = noise_decay
        self.tau = tau
        self.beta = beta
        self.diff_exp = diff_exp
        self.noise = noise
        self.sigma = sigma
        self.lamda = lamda
        self.overshoot_correction = overshoot_correction
        self.heavi_correction = heavi_correction

        self.M = M
        if self.M is None:
            self.M = self.num_particles

        self.q = self.num_particles// self.M
        
        # compute mean for init particles
        self.m_beta = self.compute_mean()
        self.update_diff = float('inf')
        self.m_diff = self.x - self.m_beta
        
    
    def step(self,time=0.0):
        ind = np.random.permutation(self.num_particles)
        
        for i in range(self.q):
            loc_ind = ind[i*self.M:(i+1)*self.M]
            self.m_beta = self.compute_mean(loc_ind)
            
            if self.heavi_correction:
                heavi_step = np.where(self.energy - self.V(self.m_beta)>0, 1,0)
            else:
                heavi_step = np.ones(self.energy.shape)
            
            x_old = self.x.copy()
            self.m_diff = self.x - self.m_beta
            
            #
            # V_beta = self.V(self.m_beta)[:, np.newaxis]
            # V_min_beta = np.min(V_beta)
            # e = 0.001*np.abs(V_beta - V_min_beta)/V_min_beta
            
            if self.overshoot_correction:
                y = self.m_beta[loc_ind,:] + self.m_diff[loc_ind,:] * np.exp(-self.lamda*self.tau)
                self.x[loc_ind,:] = y + self.sigma * self.noise(y - self.m_beta[loc_ind,:])
            else:
                self.x[loc_ind,:] = self.x[loc_ind,:] -\
                                    self.lamda * self.tau * heavi_step[:,np.newaxis] * self.m_diff[loc_ind,:] +\
                                    self.sigma * self.noise(self.m_diff[loc_ind,:])

            self.update_diff = np.linalg.norm(self.x - x_old)
        
        
    def compute_mean(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)
        
        m_beta = np.zeros(self.x.shape)
        # update energy
        self.energy = self.V(self.x)[ind]
        # V_min = np.min(self.energy)
        
        # for j in ind:
        weights = - self.beta * self.energy
        coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
        m_beta[ind,:] = np.sum(self.x[ind,:]*coeffs,axis=0)
        
        return m_beta