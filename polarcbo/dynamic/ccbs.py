import numpy as np
from scipy.special import logsumexp
import scipy.linalg as scpl
import warnings

from .pdyn import ParticleDynamic
from polarcbo import functional

#%% Multi-mean CBS (deprecated)
            
class CCBS(ParticleDynamic):
    r"""Cluster CBO class

    This class implements the cluster CBO algorithm as described in [1]_. The algorithm is based on the
    consensus-based dynamics (CBO) algorithm [2]_.

    Parameters
    ----------
    x : array_like
        The initial positions of the particles. The shape of the array should be (num_particles, num_dimensions).
    V : obejective
        The objective function :math:`V(x)` of the system.
    beta : float, optional
        The heat parameter :math:`\beta` of the system. The default is 1.0.
    tau : float, optional
        The time constant :math:`\tau` of the noise model. The default is 0.1.
    mode : str, optional
        The mode of the algorithm. The default is ``sampling``.
    kernel : object, optional
        The kernel function :math:`K(x_i, x_j)` that is used to compute the mean :math:`\mathsf{m}(x_i)`. The default is ``Gaussian_kernel()``.
    num_means : int, optional
        The number of means :math:`\mathsf{m}(x_i)` that are used to compute the polarized mean :math:`\mathsf{m}_\beta(x_i)`. The default is 1.
    repulsion_scale : float, optional
        The repulsion scale :math:`\lambda` of the repulsion term. The default is 0.0.

    References
    ----------
    .. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Polarized consensus-based
              dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.
    .. [2] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
           Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.

    See Also
    --------
    CBO : Consensus-based dynamics
    PolarCBS : Polarized CBS
    CCBO : Cluster CBO
    
    """

    def __init__(self,x, V,\
                 beta = 1.0, tau=0.1, mode = 'sampling', kernel=functional.Gaussian_kernel(),\
                 num_means=1, repulsion_scale=0.):
        
        super(CCBS, self).__init__(x, V, beta = beta)
        
        warnings.warn('Deprecated algorithm! Consider using PolarCBS instead.')

        
        # additional parameters
        self.tau = tau
        self.alpha = np.exp(-self.tau)
        self.beta = beta
        
        self.num_means = num_means
        
        
        if mode == 'sampling':
            self.lamda = 1/(1 + self.beta)
        elif mode == 'optimization':
            self.lamda = 1
        else:
            raise NotImplementedError("Invalid mode")
        
        self.kernel = kernel
        
        self.repulsion_scale = repulsion_scale
                        
        # compute mean for init particles
        self.num_means = num_means
        self.means = np.zeros((num_means,x.shape[1]))
                
        p = np.random.uniform(0, 1, (self.num_means, self.num_particles))
        p = p/np.sum(p, axis=0)
        self.logp = np.log(p)
        
        self.compute_mean()
        
        self.m_beta = p.T @ self.means
        
        self.set_logp()
        
            
    def step(self,time=0.0):
        
        self.set_logp()
        self.compute_mean()
        
        p = np.exp(self.logp)
        p = np.round(p)
        self.m_beta = p.T @ self.means
        
        self.m_diff = self.x - self.m_beta
        
        self.x = self.m_beta + self.alpha * self.m_diff + self.covariance_noise()
    
           
    
    def set_logp(self,):
        logp_max = np.max(self.logp, axis = 0)
        for i in range(self.num_means):
            state = self.logp[i,:]                         
            alpha = self.repulsion_scale
            
            self.logp[i,:] = alpha * (state - logp_max) \
                - self.kernel.neg_log(self.means[i,:], self.x)                  
            
            
        self.logp = self.logp - logsumexp(self.logp, axis=0)[np.newaxis,:]
        
        
    def compute_mean(self):
        # update energy
        self.energy = self.V(self.x)
        for j in range(self.num_means):
            
            weights = self.logp[j,:] - self.beta * self.energy
            coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
            self.means[j,:] = np.sum(self.x*coeffs,axis=0)
        
            
    def update_covariance(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)
                
        S = np.zeros((self.x.shape[0], self.x.shape[1], self.x.shape[1]))
        
        for i in ind:                               
            pi = self.logp[:,i] 
            j_max = np.argmax(pi)   #likeliest cluster center            
            indices = np.where(self.logp[j_max,:] == np.max(self.logp, axis=0))[0]   #points with the same center
            
            neg_log_kernel = 0*self.kernel.neg_log(self.x[i,:], self.x[indices,:])                                    
            weights = - neg_log_kernel - self.beta * self.energy[indices]        # do the kernel CBS weighting for those points
            
            coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
            
            diff = self.x[indices,:] -  self.means[j_max,:]     #self.m_beta[i,:]
            D = diff[:,:,np.newaxis]@diff[:,np.newaxis,:]
            S[i,::] = np.sum(coeffs[::,np.newaxis] * D, axis=0)
            
        ###
        ###############    
        self.C = S 
        self.C_sqrt = np.zeros(self.C.shape)
        for j in ind:
            self.C_sqrt[j,::] = np.real(scpl.sqrtm(self.C[j,::]))              
    
    def covariance_noise(self):
        self.update_covariance()
        z = np.random.normal(0, 1, size=self.x.shape) # num, d
        noise = np.zeros(self.x.shape)
        for j in range(self.x.shape[0]):
            noise[j,:] = self.C_sqrt[j,::]@(z[j,:])
        return (np.sqrt(1/self.lamda * (1-self.alpha**2)))*noise
    
    # def covariance_noise(self):
    #     return  np.sqrt(self.tau) * \
    #         np.random.normal(0, 1, size=self.m_diff.shape) * \
    #             np.linalg.norm(self.m_diff, axis=1,keepdims=True)
