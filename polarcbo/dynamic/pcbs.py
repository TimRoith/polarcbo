import numpy as np
from scipy.special import logsumexp
import scipy.linalg as scpl

from .pdyn import ParticleDynamic
from polarcbo import functional

class PolarCBS(ParticleDynamic):
    r"""Polarized CBS class

    This class implements the polarized CBS algorithm as described in [1]_.

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
    
    References
    ----------
    .. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Polarized consensus-based
              dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.

    """

    def __init__(self,x, V,\
                 beta = 1.0, tau=0.1, mode = 'sampling', kernel=functional.Gaussian_kernel()):
        
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
        r"""Perform one step of the algorithm

        Parameters
        ----------
        time : float, optional
            The current time of the algorithm. The default is 0.0.

        Returns
        -------
        None.

        """
        
        self.compute_mean()
        self.m_diff = self.x - self.m_beta
        
        self.x = self.m_beta + self.alpha * self.m_diff + self.covariance_noise()
               
    
    def covariance_noise(self):
        r"""Compute the covariance noise

        Returns
        -------
        noise : array_like
            The covariance noise.

        """

        self.update_covariance()
        z = np.random.normal(0, 1, size=self.x.shape) # num, d
        noise = np.zeros(self.x.shape)
        for j in range(self.x.shape[0]):
            noise[j,:] = self.C_sqrt[j,::]@(z[j,:])
        return (np.sqrt(1/self.lamda * (1-self.alpha**2)))*noise
    
    
    def update_covariance(self, ind=None):
        r"""Update the covariance matrix :math:`\mathsf{C}(x_i)` of the noise model

        Parameters
        ----------
        ind : array_like, optional
            The indices of the particles for which the covariance matrix is updated. The default is None.

        Returns
        -------
        None.

        """

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
        r"""Compute the mean :math:`\mathsf{m}(x_i)` of the noise model

        Parameters
        ----------
        ind : array_like, optional
            The indices of the particles for which the mean is computed. The default is None.

        Returns
        -------
        None.

        """
        
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