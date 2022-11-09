import numpy as np

__all__ = ['Gaussian_kernel', 'Laplace_kernel', 'Constant_kernel',\
           'InverseQuadratic_kernel', 'Taz_kernel']

class Gaussian_kernel:
    """Gaussian Kernel
    ========
    This class implements a Gaussian kernel, that can be used for PolarCBO.
    ----------
    Arguments:
        kappa (float, optional): The communication radius of the kernel. 
            Using kappa=np.inf yields a constant kernel. Default: 1.0.
    """
    def __init__(self, kappa = 1.0):
        self.kappa = kappa
    
    def __call__(self, x,y):
        """Evaluates the Gaussian Kernel
        
        Arguments:
            x (np.array)
            y (np.array)
        """
        
        dists = np.linalg.norm(x-y, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-np.true_divide(1, 2*self.kappa**2) *dists**2)
    
    def neg_log(self, x,y):
        """Evaluates the negative lograithm of the Gaussian Kernel
        
        Arguments:
            x (np.array)
            y (np.array)
        """
        
        dists = np.linalg.norm(x-y, axis=1,ord=2)
        return np.true_divide(1, 2*self.kappa**2) * dists**2       

class Laplace_kernel:
    def __init__(self, kappa = 1.0):
        self.kappa = kappa
    
    def __call__(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-np.true_divide(1, self.kappa) * dists)
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1,ord=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(1, self.kappa) * dists
        
class Constant_kernel:
    def __init__(self, kappa = 1.0):
        self.kappa = kappa
    
    def __call__(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        dists = dists / self.kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-dists**np.infty)
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        dists = dists / self.kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            return dists**np.infty
    
class InverseQuadratic_kernel:
    def __init__(self, kappa = 1.0):
        self.kappa = kappa
    
    def __call__(self, x,y):
        dists = np.true_divide(1, self.kappa) * np.linalg.norm(x-y, axis=1,ord=2)
        return 1/(1+dists**2)
    
    def neg_log(self, x,y):
        dists = np.true_divide(1, self.kappa) * np.linalg.norm(x-y, axis=1,ord=2)
        return -np.log(1/(1+dists**2))
    
class Taz_kernel:
    def __init__(self, kappa = 1.0):
        self.kappa = kappa
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=1,ord=2)
        dists =  np.true_divide(1, self.kappa) *  dists/np.max(dists)
        return  dists**2

