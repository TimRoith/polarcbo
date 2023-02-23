import numpy as np
from scipy.special import logsumexp
import sklearn.cluster as skc

from .pdyn import ParticleDynamic

#KMeans CBO
class KMeansCBO(ParticleDynamic):
    r"""KMeansCBO class

    This class implements the KMeansCBO algorithm. It is a particle-based algorithm that uses the KMeans algorithm
    to cluster the particles and then uses the cluster centers as the mean of the particles in the cluster. The
    algorithm is initialized with a set of particles and a set of weights. The weights are used to initialize the
    KMeans algorithm. The algorithm then proceeds to update the particles according to the following rule:

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
        The default is 1.0.
    sigma : float, optional
        The default is 1.0.
    n_clusters : int, optional
        The number of clusters to form as well as the number of centroids to generate. The default is 1.

    See Also
    --------
    CBO : Consensus-based dynamics
    
    """


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