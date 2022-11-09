import numpy as np
from scipy.special import logsumexp

from .particledynamic import ParticleDynamic
from polarcbo import functional

#%% Multi-mean CBO    
class CCBO(ParticleDynamic):
    def __init__(self,x, V, noise, num_means=1,\
                 beta = 1.0, noise_decay=0.0, diff_exp=1.0,kernel=functional.Gaussian_kernel(),\
                 tau=0.1, sigma=1.0, lamda=1.0, repulsion_scale=0.0, M=None,\
                 overshoot_correction=False, heavi_correction=False):
        
        super(CCBO, self).__init__(x, V, beta = beta)
        
        # additional parameters
        self.repulsion_scale = repulsion_scale
        self.noise_decay = noise_decay
        self.kernel=kernel
        self.tau = tau
        self.beta = beta
        self.diff_exp = diff_exp
        self.noise = noise
        self.sigma = sigma
        self.lamda = lamda
        self.overshoot_correction = overshoot_correction
        self.heavi_correction = heavi_correction

        
        if M is None:
            self.M = self.num_particles
        else:
            self.M = min(M, self.num_particles)

        self.q = self.num_particles// self.M
        
        # compute mean for init particles
        self.num_means = num_means
        self.means = np.zeros((num_means,x.shape[1]))
        
        p = np.random.uniform(0, 1, (self.num_means, self.num_particles))
        p = p/np.sum(p, axis=0)
        
        self.logp = np.log(p)
        
        self.compute_mean()
        
        self.m_beta = p.T @ self.means
        
        
        
        self.set_logp()
        self.update_diff = float('inf')
        
    
    def step(self,time=0.0):
        ind = np.random.permutation(self.num_particles)
        
        for i in range(self.q):
            loc_ind = ind[i*self.M:(i+1)*self.M]
            
            self.set_logp(ind=loc_ind)
            self.compute_mean(ind=loc_ind)
            if self.heavi_correction:
                heavi_step = np.where(self.energy - self.V(self.m_beta)>0, 1,0)
            else:
                heavi_step = np.ones(self.energy.shape)
            
            x_old = self.x.copy()
            
            p = np.exp(self.logp)
            self.m_beta = p.T @ self.means
            
            self.m_diff = self.x - self.m_beta

            self.x[loc_ind,:] = self.x[loc_ind,:] -\
                                self.lamda * self.tau * heavi_step[loc_ind, np.newaxis] * self.m_diff[loc_ind,:] +\
                                self.sigma * self.noise(self.m_diff[loc_ind,:])
                                    
            self.update_diff = np.linalg.norm(self.x - x_old)
        
    
    def set_logp(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)
            
        update_ind = np.zeros(self.logp.shape[1], dtype=bool)
        update_ind[ind] = True
        
        logp_max = np.max(self.logp[:, ind], axis = 0)
        for i in range(self.num_means):
            state = self.logp[i, ind]                         
            alpha = self.repulsion_scale
            
            if np.sum(state<=-1e50) >= len(ind):
                state = np.random.uniform(0.,1., size=(len(ind),))
                update_ind[i] = False
            else:            
                self.logp[i, ind] = alpha * (state - logp_max)\
                    - self.kernel.neg_log(self.means[i,:], self.x[ind, :]) 
                    

            
            self.logp[i, np.where(self.logp[i, :]<-1e100)] = -np.inf
            
            
        self.logp[:, update_ind] = self.logp[:, update_ind] - logsumexp(self.logp[:, update_ind], axis=0)[np.newaxis,:]
        #self.logp[np.where(np.isnan(self.logp))] = -np.inf
        
    def compute_mean(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)

        # update energy
        self.energy = self.V(self.x)
        for j in range(self.num_means):  
            weights = self.logp[j, ind] - self.beta * self.energy[ind]
            
            if not (logsumexp(weights) <= -1e100):
                coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
                self.means[j,:] = np.sum(self.x[ind,:] * coeffs, axis=0)