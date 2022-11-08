import numpy as np
from scipy.special import logsumexp

def init_particles(num_particles=100, d=2, x_min=0.0, x_max = 1.0, delta=1.0, method="uniform"):
    if method == "uniform":
        x = np.random.uniform(x_min, x_max, (num_particles,d))
    elif method == "normal":
        x = np.random.multivariate_normal(np.array([0,0]),delta*np.eye(d),(num_particles,))
    else:
        raise Exception('Unknown method for init_particles specified!')
        
    return x
        

           
           
class config:
    def __init__(self, T=10, tau = 0.01, num_particles = 100):
        self.T = T
        self.tau = tau
        self.num_particles = num_particles
        self.save2disk = False
        
        

    
def found_minimas(m , z, thresh = 0.25):
    d = m.shape[1]
    num_particles = m.shape[0]
    num_minimas = z.shape[0]
    
    dists = np.linalg.norm(np.reshape(m, (num_particles, 1, d)) - np.reshape(z, (1,num_minimas, d)),axis=2, ord=np.inf)
    dists_min_idx = np.argmin(dists, axis=1)
    
    succes_idx = np.where(dists[np.arange(num_particles), dists_min_idx] < thresh, dists_min_idx,num_minimas)
    
    found = 0
    sc = np.unique(succes_idx)
    for j in range(num_minimas):
        if j in sc:
            found += 1
    
    return found

        
