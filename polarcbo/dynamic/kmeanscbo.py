import numpy as np
from scipy.special import logsumexp
import sklearn.cluster as skc

from .optimizer import Optimizer

#KMeans CBO
class KMeansCBO(Optimizer):
    def __init__(self,x, V, noise,\
                 beta = 1.0, noise_decay=0.0, diff_exp=1.0, kappa = 1.0,\
                 tau=0.1, sigma=1.0, lamda=1.0, n_clusters=1):
        
        super(KMeansCBO, self).__init__(x, V, beta = beta)
        
        # additional parameters
        self.noise_decay = noise_decay
        self.kappa = kappa
        self.tau = tau
        self.beta = beta
        self.diff_exp = diff_exp
        self.noise = noise
        self.sigma = sigma
        self.lamda = lamda
        self.n_clusters = n_clusters
        self.KMeans = skc.KMeans(n_clusters=n_clusters)
        self.M = self.num_particles
        self.q = 1
        
        
        self.set_logp()
        self.m_beta = self.compute_mean()
        
        p = np.exp(self.logp)
        self.m_x = np.sum(p[:,np.newaxis,:] * self.m_beta[np.newaxis,:,:],axis=2)
        self.m_diff = self.x - self.m_x
        
        
        self.update_diff = float('inf')
    
    def set_logp(self):
        if hasattr(self.KMeans,'cluster_centers_'):
            self.KMeans = skc.KMeans(n_clusters=self.n_clusters, init=self.KMeans.cluster_centers_, n_init=1)
        self.KMeans.fit(self.x)
        
        
        
        res = -float('inf') * np.ones((self.x.shape[0], self.KMeans.n_clusters))
        for l in range(self.KMeans.n_clusters):
            res[self.KMeans.labels_==l, l] = 0.0
        self.logp = res
        

    def compute_mean(self,):  
        
        V_min = np.min(self.V(self.x))
        energy = self.V(self.x)
        m_beta = np.zeros((self.x.shape[1], self.n_clusters))
        
        for l in range(self.n_clusters):
            
            denom = logsumexp(-self.beta*(energy) + self.logp[:,l])
            coeff_sum = np.expand_dims(np.exp(self.logp[:, l] -self.beta*(energy) - denom),axis=1)
            
            #h = np.expand_dims(p[:,l] * np.exp(-beta*(V(x))), axis=1)
            
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
            
            m_diff = self.x - self.m_x

            self.x[loc_ind,:] = self.x[loc_ind,:] -\
                                self.lamda * self.tau * m_diff[loc_ind,:] +\
                                self.sigma * self.noise(m_diff[loc_ind,:])
                                    
            self.update_diff = np.linalg.norm(self.x - x_old)