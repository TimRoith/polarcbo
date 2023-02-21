import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .pdyn import ParticleDynamic


#%% EM CBO
class EMCBO(ParticleDynamic):
    def __init__(self, x, V, noise,\
                 beta=1.0, tau=0.01, lamda=1.0,sigma=0.1,\
                 num_components=1.0, lower_thresh=0.0):
        
        super(EMCBO, self).__init__(x, V, beta=beta)
        self.num_components = num_components
        self.GM = GaussianMixture(n_components=num_components, warm_start=True)
        self.lower_thresh = lower_thresh
        self.beta = beta
        self.M = self.num_particles
        self.q = 1
        self.tau = tau
        self.lamda=lamda
        self.noise = noise
        self.sigma = sigma
        
        self.set_logp()
        self.m_beta = self.compute_mean()
        p = np.exp(self.logp)
        self.m_x = np.sum(p[:,np.newaxis,:] * self.m_beta[np.newaxis,:,:],axis=2)
        self.m_diff = self.x - self.m_x
        
        
    def set_logp(self,):
        self.GM.fit(self.x)
        logphi = np.zeros(shape=(self.x.shape[0], self.num_components))
        
        for l in range(self.num_components):
            nn,mm = self.GM.covariances_[l,:,:].shape
            logphi[:,l] = multivariate_normal.logpdf(self.x, mean=self.GM.means_[l,:], \
                                                cov=self.GM.covariances_[l,:,:]+self.lower_thresh*np.eye(nn,mm))

        self.logp = logphi - logsumexp(logphi, axis=1)[:,np.newaxis]
        

    def compute_mean(self,):
        self.energy = self.V(self.x)
        m_beta = np.zeros((self.x.shape[1], self.num_components))
        
        for l in range(self.num_components):
            denom = logsumexp(-self.beta* self.energy + self.logp[:,l])
            coeff_sum = np.expand_dims(np.exp(self.logp[:, l] - self.beta*self.energy - denom),axis=1)
            m_beta[:,l] = np.sum(self.x*coeff_sum,axis=0)
            #
        return m_beta
    

    def step(self,time=0.0):
        ind = np.random.permutation(self.num_particles)
        
        for i in range(self.q):
            loc_ind = ind[i*self.M:(i+1)*self.M]
            
            self.set_logp()
            self.m_beta = self.compute_mean()
            
            
            x_old = self.x.copy()
            
            p = np.exp(self.logp)
            self.m_x = np.sum(p[:,np.newaxis,:] * self.m_beta[np.newaxis,:,:],axis=2)
            
            self.m_diff = self.x - self.m_x

            self.x[loc_ind,:] = self.x[loc_ind,:] -\
                                self.lamda * self.tau * self.m_diff[loc_ind,:] +\
                                self.sigma * self.noise(self.m_diff[loc_ind,:])
                                    
            self.update_diff = np.linalg.norm(self.x - x_old)
        