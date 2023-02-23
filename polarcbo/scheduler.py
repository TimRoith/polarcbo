r"""
Scheduler
==========

This module implements the :math:`\beta`-schedulers employed in the conensuse schemes.

"""

import numpy as np
from scipy.special import logsumexp

class scheduler_base(object):
    r"""scheduler_base class
    
    This class implements the base scheduler class. It is used to implement the :math:`\beta`-schedulers
    employed in the consensus schemes.
    
    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\beta`-parameter should be updated

    beta : float, optional
        The initial value of the :math:`\beta`-parameter. The default is 1.0.

    beta_max : float, optional
        The maximum value of the :math:`\beta`-parameter. The default is 100000.0.

    """

    def __init__(self, opt, beta=1.0, beta_max = 100000.):
        self.opt = opt
        self.beta = beta
        self.beta_max = beta_max

    def update(self):
        pass

    def __call__(self):
        r"""Call method for classes that inherit from ``scheduler_base``
        
        Returns
        -------
        
        beta : float
            The current value of the :math:`\beta`-parameter in opt.
        """

        return self.opt.beta

class beta_exponential(scheduler_base):
    r"""Exponential scheduler class

    This class implements an exponential scheduler for the :math:`\beta`-parameter. The :math:`\beta`-parameter
    is updated according to the rule

    .. math::

        \beta_{k+1} = \beta_k \cdot r

    where :math:`r` is a parameter of the class.

    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\beta`-parameter should be updated   
    beta : float, optional
        The initial value of the :math:`\beta`-parameter. The default is 1.0.
    beta_max : float, optional
        The maximum value of the :math:`\beta`-parameter. The default is 100000.0.
    r : float, optional
        The parameter :math:`r` of the scheduler. The default is 1.0.

    """

    def __init__(self, opt, beta = 1.0, beta_max = 100000., r = 1.0):
        super(beta_exponential, self).__init__(opt, beta = beta, beta_max = beta_max)

        self.r = r
    
    def update(self):
        r"""Update the :math:`\beta`-parameter in opt according to the exponential scheduler."""

        beta = self.opt.beta
        self.opt.beta = min(self.r * beta, self.beta_max)
    
    


# class for beta_eff scheduler
class beta_eff(scheduler_base):
    r"""beta_eff scheduler class
    
    This class implements a scheduler for the :math:`\beta`-parameter based on the effective number of particles.
    The :math:`\beta`-parameter is updated according to the rule
    
    .. math::
        
        \beta_{k+1} = \begin{cases}
        \beta_k \cdot r & \text{if } J_{eff} \geq \eta \cdot J \\ 
        \beta_k / r & \text{otherwise}
        \end{cases} 
        
    where :math:`r`, :math:`\eta` are parameters and :math:`J` is the number of particles. The effictive number of
    particles is defined as

    .. math::

        J_{eff} = \frac{1}{\sum_{i=1}^J w_i^2}
    
    where :math:`w_i` are the weights of the particles. This was, e.g., employed in [1]_.


    
    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\beta`-parameter should be updated
    eta : float, optional
        The parameter :math:`\eta` of the scheduler. The default is 1.0.
    beta_max : float, optional
        The maximum value of the :math:`\beta`-parameter. The default is 100000.0.
    factor : float, optional
        The parameter :math:`r` of the scheduler. The default is 1.05. 

    References
    ----------
    .. [1] Carrillo, J. A., Hoffmann, F., Stuart, A. M., & Vaes, U. (2022). Consensus‐based sampling. Studies in Applied Mathematics, 148(3), 1069-1140. 


    """
    def __init__(self, opt, eta=1.0, beta_max=1e5, factor=1.05):
        super(beta_eff, self).__init__(opt, beta_max = beta_max)

        self.eta = eta
        self.beta_max = beta_max
        self.J_eff = 1.0
        self.factor=factor
    
    def update(self,):
        beta = self.opt.beta
        
        term1 = logsumexp(-beta*self.opt.V(self.opt.x))
        term2 = logsumexp(-2*beta*self.opt.V(self.opt.x))
        self.J_eff = np.exp(2*term1-term2)
        
        #w = np.exp(-beta * self.opt.V(self.opt.x))
        #self.J_eff = np.sum(w)**2/max(np.linalg.norm(w)**2,1e-7)
        
        if self.J_eff >= self.eta * self.opt.num_particles:
            self.opt.beta = min(beta*self.factor, self.beta_max)
        else:
            pass
            #self.opt.beta /= self.factor