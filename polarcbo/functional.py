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


class LogBarrier():
    def __call__(theta):
        return np.sum(np.log(1/(1-theta)) + np.log(1/(1+theta)),axis=1)
    
    def grad(self, theta):
        return 1/(1-theta) - 1/(1+theta)
    
    def grad_conj(self, y):
        return -1/y + 1/y * np.sqrt(1+ y**2)
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(((1/(1-theta))**2 + (1/(1+theta))**2),axis=1)*np.eye(m)
    
class L2():
    def __call__(theta):
        return 0.5*np.sum(theta**2)
    
    def grad(self, theta):
        return theta
    
    def grad_conj(self, y):
        return y
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(np.ones(theta.shape),axis=1)*np.eye(m)
    
class weighted_L2():
    def __init__(self, A):
        self.A = A
    
    def __call__(self,theta):
        return 0.5*theta.T@self.A@theta
    
    def grad(self, theta):
        return np.reshape(0.5*(self.A + self.A.T)@theta[:,:,np.newaxis],theta.shape)
    
    def grad_conj(self, y):
        raise Warning('Check implementation')
        return np.linalg.solve(0.5*(self.A + self.A.T),y.T).T
    
    def hessian(self, theta):
        return np.expand_dims(np.ones(theta.shape),axis=1) * 0.5*(self.A.T+self.A)
    
class NonsmoothBarrier():
    def __call__(self, theta):
        return np.sum(np.abs(theta)/(1-np.abs(theta)))
    
    def grad(self, theta):        
        return np.sign(theta)/(1 + np.abs(theta)**2 - 2*np.abs(theta))
    
    def grad_conj(self, y):        
        return np.sign(y) * np.maximum(1-np.sqrt(1/np.abs(y)), 0)
    
    def hessian(self, theta):
        n,m = theta.shape
        
        tmp = (-2*np.abs(theta) + 2)\
            /(-4*np.abs(theta)-4*np.abs(theta)**3+np.abs(theta)**4 + 6*np.abs(theta)**2 + 1)    
        res = np.expand_dims(tmp,axis=1)*np.eye(m)
        
        return res
    
class ElasticNet():
    
    def __init__(self, delta=1.0, lamda=1.0):
        self.delta = delta
        self.lamda = lamda
    
    def __call__(self,theta):
        return (1/(2*self.delta))*np.sum(theta**2) + self.lamda*np.sum(np.abs(theta))
    
    def grad(self, theta):
        return (1/(self.delta))*theta + self.lamda*np.sign(theta)
    
    def grad_conj(self, y):
        return self.delta*np.sign(y) * np.maximum((np.abs(y) - self.lamda),0)
    
    def hessian(self, theta):
        n,m = theta.shape
        I = 1/self.delta * np.expand_dims(np.ones(theta.shape),axis=1)*np.eye(m)
        
        # idx,_ = np.where(np.abs(theta)<1e-12)
        # rho = np.zeros(theta.shape)
        # rho[idx] = 0e1
        
        # J = np.expand_dims(rho,axis=1)*np.eye(m)
        return I #+ J