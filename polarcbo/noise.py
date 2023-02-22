"""
Noise
======

This module implements some noise models commonly used in consensus 
based methods. 
  
"""

import numpy as np
from abc import ABC, abstractmethod

#%%
class noise_model(ABC):
    """Abstract noise model
    """

    @abstractmethod
    def __call__(self, m_diff):
        """Abstract call method for noise_model classes

        Parameters
        ----------
        m_diff : array_like, shape (J, d) 
            For a system of :math:`J` particles, the i-th row of this array ``m_diff[i,:]`` 
            represents the vector :math:`x_i - \mathsf{m}(x_i)` where :math:`x\in\R^d` denotes 
            the position of the i-th particle and :math:`\mathsf{m}(x_i)` its weighted mean.

        Returns
        -------
        n : array_like, shape (J,d)
            The random vector that is added to each particle in the dynamic.
        """

class normal_noise(noise_model):
    """
    Utility function.

    Does some good stuff.
    """
    def __init__(self, tau = 0.1):
        self.tau = tau

    def __call__(self, m_diff):
        z = np.sqrt(self.tau) * np.random.normal(0, 1, size=m_diff.shape)
        return z * np.linalg.norm(m_diff, axis=1,keepdims=True)
    
class comp_noise(noise_model):   
    """
    Utility function.

    Does some good stuff.
    """    
    def __init__(self, tau = 0.1):
        self.tau = tau

    def __call__(self, m_diff):
        """call
        """
        z = np.random.normal(0, np.sqrt(self.tau), size=m_diff.shape) * m_diff
        return z