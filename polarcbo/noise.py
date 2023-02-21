"""
Noise
======

This module implements some noise models commonly used in conensus 
based methods. They are similarly implemented as classes, where the 
``__call__`` function has the signature 

"""

import numpy as np

class normal_noise():
    def __init__(self, tau = 0.1):
        self.tau = tau
        
    def __call__(self, m_diff):
        z = np.sqrt(self.tau) * np.random.normal(0, 1, size=m_diff.shape)
        return z * np.linalg.norm(m_diff, axis=1,keepdims=True)
    
class comp_noise:
    def __init__(self, tau=0.1):
        self.tau = tau
        
    def __call__(self, m_diff):
        z = np.random.normal(0, np.sqrt(self.tau), size=m_diff.shape) * m_diff
        return z