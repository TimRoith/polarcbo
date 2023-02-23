import numpy as np
from scipy.special import logsumexp

from .pdyn import ParticleDynamic

#%% CBO
class CBO(ParticleDynamic):
    r"""CBO class

    This class implements the CBO algorithm as described in [1]_. The algorithm
    is a particle dynamic algorithm that is used to minimize the objective function :math:`V(x)`.

    Parameters
    ----------
    x : array_like, shape (J, d)
        The initial positions of the particles. For a system of :math:`J` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    V : obejective
        The objective function :math:`V(x)` of the system.
    beta : float, optional
        The heat parameter :math:`\beta` of the system. The default is 1.0.
    noise : noise_model, optional
        The noise model that is used to compute the noise vector. The default is ``normal_noise(tau=0.1)``.
    noise_decay : float, optional
        The decay parameter :math:`\lambda` of the noise model. The default is 0.0.
    diff_exp : float, optional
        The exponent :math:`\alpha` of the difference vector :math:`x_i - \mathsf{m}(x_i)`. The default is 1.0.
    tau : float, optional
        The parameter :math:`\tau` of the noise model. The default is 0.1.
    sigma : float, optional
        The parameter :math:`\sigma` of the noise model. The default is 1.0.
    lamda : float, optional
        The decay parameter :math:`\lambda` of the noise model. The default is 1.0.
    M : int, optional
        The number of particles that are used to compute the batch mean :math:`\mathsf{m}(x_i)` as described in [2]_.
        The default is ``None``. If ``None``, then :math:`M = J`.
    overshoot_correction : bool, optional
        If ``True``, then the overshoot correction is used as described in [3]_. The default is ``False``.
    heavi_correction : bool, optional
        If ``True``, then the Heaviside correction is used as described in [4]_. The default is ``False``.
    
    References
    ----------
    .. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
        Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.
    .. [2] Carrillo, J. A., Jin, S., Li, L., & Zhu, Y. (2021). A consensus-based global optimization method for high dimensional machine learning problems. 
        ESAIM: Control, Optimisation and Calculus of Variations, 27, S5.
    .. [3] M. A. S. Amin, M. A. S. Amin, and M. A. S. Amin, “Consensus-based optimization: A survey,”
        IEEE Transactions on Cybernetics, vol. 48, no. 1, pp. 1–13, 2017.
    .. [4] ref missing
    .. [5] ref missing

    """

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
        r"""Performs one step of the CBO algorithm.

        Parameters
        ----------
        
        time : float, optional
            The current time of the algorithm. The default is 0.0.
        
        """

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
        r"""Computes the mean of the particles.

        Parameters
        ----------
        ind : array_like, optional
            The indices of the particles that are used to compute the mean. The default is ``None``. If ``None``, then all particles are used.
        
        Returns
        -------
        m_beta : numpy.ndarray
            The mean of the particles.

        """

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