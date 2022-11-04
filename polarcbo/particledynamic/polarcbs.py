import numpy as np
from scipy.special import logsumexp
import scipy.linalg as scpl

from .particledynamic import ParticleDynamic
from kernelcbo import kernels

class PolarCBS(ParticleDynamic):
    def __init__(self,x, V,\
                 beta = 1.0, tau=0.1, mode = 'sampling', kernel=kernels.Gaussian_kernel()):
        
        super(PolarCBS, self).__init__(x, V, beta = beta)
        
        # additional parameters
        self.tau = tau
        self.alpha = np.exp(-self.tau)
        self.beta = beta
                
        if mode == 'sampling':
            self.lamda = 1/(1 + self.beta)
        elif mode == 'optimization':
            self.lamda = 1
        else:
            raise NotImplementedError("Invalid mode")
        
        self.kernel = kernel
        
                        
        # compute mean for init particles
        self.m_beta = np.zeros(self.x.shape)
        
            
    def step(self,time=0.0):
        
        self.compute_mean()
        self.m_diff = self.x - self.m_beta
        
        self.x = self.m_beta + self.alpha * self.m_diff + self.covariance_noise()
               
    
    def covariance_noise(self):
        self.update_covariance()
        z = np.random.normal(0, 1, size=self.x.shape) # num, d
        noise = np.zeros(self.x.shape)
        for j in range(self.x.shape[0]):
            noise[j,:] = self.C_sqrt[j,::]@(z[j,:])
        return (np.sqrt(1/self.lamda * (1-self.alpha**2)))*noise
    
    
    def update_covariance(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)             
    
        S = np.zeros((self.x.shape[0], self.x.shape[1], self.x.shape[1]))
        
        for j in ind:   
            
            neg_log_kernel = self.kernel.neg_log(self.x[j,:], self.x[ind,:])                        
            weights = - neg_log_kernel - self.beta * self.energy
            coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
            
            diff = self.x[ind,:] - self.m_beta[j,:]           
            D = diff[:,:,np.newaxis]@diff[:,np.newaxis,:]
            
            S[j,::] = np.sum(coeffs[::,np.newaxis] * D, axis=0)

           
        
        self.C = S 
        self.C_sqrt = np.zeros(self.C.shape)
        for j in ind:
            self.C_sqrt[j,::] = np.real(scpl.sqrtm(self.C[j,::]))              


    def compute_mean(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)
        
        m_beta = np.zeros(self.x.shape)
        
        # update energy
        self.energy = self.V(self.x)[ind]
        
        for j in ind:
            neg_log_kernel = self.kernel.neg_log(self.x[j,:], self.x[ind,:])
                        
            weights = - neg_log_kernel - self.beta * self.energy

            coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
            m_beta[j,:] = np.sum(self.x[ind,:]*coeffs,axis=0)
            #
        self.m_beta = m_beta