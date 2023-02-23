import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .pdyn import ParticleDynamic


#%% EM CBO
class EMCBO(ParticleDynamic):
    r"""EMCBO class

    This class implements the EM CBO algorithm. The algorithm is described in detail in [1]_. The algorithm
    is a variant of the CBO algorithm that uses a Gaussian mixture model to approximate the distribution
    of the particles. The algorithm is initialized with a set of particles and a set of weights. The weights
    are used to initialize the Gaussian mixture model. The algorithm then proceeds to update the particles
    according to the following rule:

    .. math::

        x_{k+1} = x_k - \lambda \tau \nabla \log p(x_k) + \sigma \mathcal{N}(0,I)

    where :math:`\lambda` is the learning rate, :math:`\tau` is the time step, :math:`\sigma` is the noise
    parameter and :math:`\mathcal{N}(0,I)` is a zero-mean isotropic Gaussian distribution.

    Parameters
    ----------
    x : array_like
        The initial positions of the particles. For a system of :math:`J` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    V : obejective
        The objective function :math:`V(x)` of the system.
    beta : float, optional
        The heat parameter :math:`\beta` of the system. The default is 1.0.
    tau : float, optional
        The time constant :math:`\tau` of the noise model. The default is 0.01.
    lamda : float, optional
        The learning rate :math:`\lambda` of the algorithm. The default is 1.0.
    sigma : float, optional
        The noise parameter :math:`\sigma` of the algorithm. The default is 0.1.
    num_components : int, optional
        The number of components of the Gaussian mixture model. The default is 1.
    lower_thresh : float, optional
        The lower threshold of the covariance matrix of the Gaussian mixture model. The default is 0.0.

    References
    ----------
    .. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Cluster consensus-based
              dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.

    """

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
        """Set the log probability of the particles
        

        Returns
        -------
        None.
        """
        self.GM.fit(self.x)
        logphi = np.zeros(shape=(self.x.shape[0], self.num_components))
        
        for l in range(self.num_components):
            nn,mm = self.GM.covariances_[l,:,:].shape
            logphi[:,l] = multivariate_normal.logpdf(self.x, mean=self.GM.means_[l,:], \
                                                cov=self.GM.covariances_[l,:,:]+self.lower_thresh*np.eye(nn,mm))

        self.logp = logphi - logsumexp(logphi, axis=1)[:,np.newaxis]
        

    def compute_mean(self,):
        """Compute the mean of the particles

        Returns
        -------
        m_beta : array_like
            The mean of the particles.

        """

        self.energy = self.V(self.x)
        m_beta = np.zeros((self.x.shape[1], self.num_components))
        
        for l in range(self.num_components):
            denom = logsumexp(-self.beta* self.energy + self.logp[:,l])
            coeff_sum = np.expand_dims(np.exp(self.logp[:, l] - self.beta*self.energy - denom),axis=1)
            m_beta[:,l] = np.sum(self.x*coeff_sum,axis=0)
            #
        return m_beta
    

    def step(self,time=0.0):
        """Perform a single step of the algorithm

        Parameters
        ----------
        time : float, optional
            The current time of the algorithm. The default is 0.0.

        Returns
        -------
        None.

        """
        
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
        