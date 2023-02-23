import numpy as np
from scipy.special import logsumexp

from .pdyn import ParticleDynamic
from polarcbo import functional

#%% Multi-mean CBO    
class CCBO(ParticleDynamic):
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
    noise : float, optional
        The noise parameter :math:`\sigma` of the system. The default is 1.0.
    num_means : int, optional
        The number of means :math:`\mathsf{m}_i` that are used to compute the mean :math:`\mathsf{m}(x_i)`. The default is 1.
    kernel : object, optional   
        The kernel function :math:`K(x_i, x_j)` that is used to compute the mean :math:`\mathsf{m}(x_i)`. The default is ``Gaussian_kernel()``.
    tau : float, optional
        The time constant :math:`\tau` of the noise model. The default is 0.1.
    sigma : float, optional
        The noise parameter :math:`\sigma` of the system. The default is 1.0.
    lamda : float, optional
        The default is 1.0.
    repulsion_scale : float, optional
        The repulsion scale :math:`\alpha` of the system. The default is 0.0.
    M : int, optional
        The number of particles that are used to compute the mean :math:`\mathsf{m}(x_i)`. The default is ``num_particles``.
    overshoot_correction : bool, optional
        If ``True``, the overshoot correction is applied. The default is ``False``.
    heavi_correction : bool, optional
        If ``True``, the Heaviside correction is applied. The default is ``False``.
        
    References
    ----------
    .. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Cluster consensus-based
              dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.
    .. [2] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
           Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.

    See Also
    --------
    CBO : Consensus-based dynamics class
    CCBS : Cluster consensus-based sampling class
    """


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