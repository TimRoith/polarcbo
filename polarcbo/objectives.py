import numpy as np
#%% Objective function
def Rosenbrock(x, a=1.0, b=100.0):
    return (a-x[:,0])**2 + b*(x[:,1]-x[:,0]**2)**2

def three_hump_camel(x):
    return 2*x[:,0]**2 - 1.05 * x[:,0]**4 + (1/6) * x[:,0]**6 + x[:,0]*x[:,1] + x[:,1]**2

def McCormick(x):
    return np.sin(x[:,0] + x[:,1]) + (x[:,0] - x[:,1])**2 - 1.5 * x[:,0] + 2.5*x[:,1]+1


class Himmelblau():
    def __init__(self, factor=1.0):
        self.factor = factor
        
    def __call__(self, x):
        x = self.factor*x
        return (x[...,0]**2 + x[...,1] - 11)**2 + (x[...,0] + x[...,1]**2 - 7)**2

class Rastrigin():
    def __init__(self, b=0., c=0.):
        self.b = b
        self.c = c
        
    def __call__(self, x):
        return (1/x.shape[1]) * np.sum((x - self.b)**2 - \
                10*np.cos(2*np.pi*(x - self.b)) + 10, axis=-1) + self.c
            
            
class Rastrigin_multimodal():
    def __init__(self, alpha = [1.], z = np.array([[0]])):
        self.V = Rastrigin()
        self.alpha = alpha
        self.z = z
        self.minima = z
        self.num_terms = len(alpha)
        
    def __call__(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y            

class Ackley():
    def __init__(self, a=20., b=0.2, c=2*np.pi):
        self.a=a
        self.b=b
        self.c=c
    
    def __call__(self, x):
        d = x.shape[-1]
        
        arg1 = -self.b * np.sqrt(1/d) * np.linalg.norm(x,axis=-1)
        arg2 = (1/d) * np.sum(np.cos(self.c * x), axis=-1)
        
        return -self.a * np.exp(arg1) - np.exp(arg2) + self.a + np.e

class Ackley_multimodal():
    def __init__(self, alpha = [1.], z = np.array([[0]])):
        self.V = Ackley()
        self.alpha = alpha
        self.z = z
        self.minima = z
        self.num_terms = len(alpha)
        
    def __call__(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y
        
class test2d():
    def __init__(self):
        return
    
    def __call__(self, x):
        return np.cos(x.T[0])+np.sin(x.T[1])



        
        
class accelerated_sinus():
    def __init__(self, a=1.0):
        self.a = a

    def __call__(self, x):
        return np.sin((self.a * x)/(1+x*x)).squeeze() + 1
    
class nd_sinus():
    def __init__(self, a=1.0):
        self.a = a

    def __call__(self, x):
        
        x = 0.3*x
        z = 1/x.shape[-1] * np.linalg.norm(x,axis=-1)**2
        
        
        res = (np.sin(z) + 1) * (x[...,0]**4 - x[...,0]**2 + 1)
        return res.squeeze() 
    
class p_4th_order():
    def __init__(self,):
        pass

    def __call__(self, x):
        #n = np.sqrt(1/x.shape[-1]) * np.linalg.norm(x, axis=-1)
        #n = 1/x.shape[-1] *np.sum(x, axis = -1)
        n =  x
        
        #res = (n**4 - n**2 + 1)
        res = (np.sum(n**4,axis=-1) - np.sum(n**2,axis=-1) + 1)
        return res.squeeze() 
    
class Quadratic():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return np.linalg.norm(self.alpha*x, axis=-1)**2
    
class Banana():
    def __init__(self, m=0, sigma=0.5, sigma_prior=2):
        self.m = m
        self.sigma = sigma
        self.sigma_prior = sigma_prior
    
    def __call__(self, x):
        G = ((x[...,1]-1)**2-(x[...,0]-2.5) -1)
        Phi = 0.5/(self.sigma**2)*(G - self.m)**2
        I = Phi + 0.5/(self.sigma_prior**2)*np.linalg.norm(x,axis=-1)**2
        
        return I

class Bimodal():
    def __init__(self, a=[1., 1.5], b=[-1.2, -0.7]):
        self.a = a
        self.b = b
    
    def __call__(self, x):
        a = self.a
        b = self.b         
        ret = -np.log(np.exp(-((x[...,0]-a[0])**2 + (x[...,1]-a[1])**2/0.2)) \
                      + 0.5*np.exp( -(x[...,0]-b[0])**2/8 - (x[...,1]-b[1])**2/0.5 ))
        return ret
        

class Unimodal():
    def __init__(self, a=[-1.2, -0.7]):
        self.a = a
    
    def __call__(self, x):
        a = self.a
        ret = -np.log(0.5*np.exp( -(x[...,0]-a[0])**2/8 - (x[...,1]-a[1])**2/0.5 ))
        
        return ret
                