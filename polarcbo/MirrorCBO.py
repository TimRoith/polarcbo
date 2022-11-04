import numpy as np
import matplotlib.pyplot as plt
import MirrorMaps as mm
import test_functions as tf

T = 1000
tau = 0.1
beta = 100
std = 0.5
effective_particle_size = False

#%% Objective function
offset = 0.1

def V(x):
    y = x.copy()
    y -= offset
    # return y**4-y**2 + np.sin(5*y)**2
    return np.sin(5*y)**2


#%% Mirror function
phi = mm.LogBarrier()
# phi = mm.NonsmoothBarrier()        
# phi = mm.ElasticNet(lamda = 10.)   
# phi = mm.L2() 
        
#%% Initialize x#
num_points = 200
x = np.random.uniform(-1, 1, (num_points,)) 
y = phi.grad(x)

def compute_mean(x, V, beta=1):
    m_n = np.sum(x * np.exp(-beta*V(x)))
    m_d = np.sum(np.exp(-beta*V(x)))
    #
    return m_n/m_d

x_range=np.linspace(-1,1,50)

beta_loc = beta
for i in range(T):
    
    w = np.exp(-beta_loc * V(x))
    Jeff = (np.sum(w)/np.linalg.norm(w))**2
        
    if effective_particle_size is True:
        if Jeff > 0.1*num_points:
            beta_loc *=1.05
        else:
            beta_loc*=0.95
            
        print('Effective particles {}, beta {}'.format(Jeff, beta_loc))    
    
    
    m = compute_mean(x, V, beta=beta_loc)
    
    # update
    sigma = std * np.random.normal(loc=0, scale=np.sqrt(tau), size=x.shape)
    y = y - tau * (x - m) + np.abs(x - m) * sigma
    x = phi.grad_conj(y)
    
    plt.clf()
    plt.scatter(x,V(x))
    plt.plot(x_range,V(x_range))
    plt.xlim(-1,1)
    plt.ylim(-0.5,1)    
    plt.title('t = {:.2f}'.format(tau*i))
    plt.pause(0.01)
    