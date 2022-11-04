import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import kernelcbo as kcbo
import kernelcbo.particledynamic as pdyn

#%%
cur_path = os.path.dirname(os.path.realpath(__file__))

#%% set parameters
conf = kcbo.utils.config()
conf.save2disk = False
conf.T = 2001
conf.tau=0.01
conf.x_max = 7
conf.x_min = -7
conf.random_seed = 42
conf.d = 2
conf.beta = 1
conf.sigma = 1.0
conf.heavy_correction = False
conf.num_particles = 200
conf.factor = 1.0
conf.noise = kcbo.noise.normal_noise(tau=conf.tau)
conf.eta = 0.5

conf.kappa = .5
conf.kernel = kcbo.kernels.Gaussian_kernel(kappa=conf.kappa)    


snapshots = [0, 100, 500, 1000, 2000]


z = np.array([[3.,2.],[0,0], [-1,-3.5]])
alphas = np.array([1,1,1])

# z = np.array([[0,0]])
# alphas = np.array([1])

# z = np.pad(z, [[0,0], [0,conf.d-2]])

# conf.V = tf.Ackley_multimodal(alpha=alphas,z=z)
# conf.V = tf.Rastrigin()
conf.V = kcbo.objectives.Rastrigin_multimodal(alpha=alphas,z=z)

#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = kcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
    
#%% init optimizer and scheduler
opt = pdyn.KernelCBO(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                       beta = conf.beta, kernel=conf.kernel)

beta_sched = kcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e7)

#%% define variance
def compute_variance(self):    
    var = 0
    for i in range(len(z)):
        var = var + 0.5*np.sum(np.linalg.norm(self.x-z[i,:], axis=-1)**2)
    var = var / len(self.x)    
    return var  

from scipy.special import logsumexp

def compute_kernelized_variance(self, ind=None):
    m_beta = self.compute_mean()
    var = np.zeros(len(self.x))
    self.energy = self.V(self.x)
    
    for i in range(len(self.x)):
        neg_log_kernel = -self.kernel.neg_log(self.x[i,:], self.x)
        weights = - neg_log_kernel - self.beta*self.energy
        coeffs = np.expand_dims(np.exp(weights-logsumexp(weights)),axis=1)
        var = 0.5*np.sum(np.linalg.norm(self.x-m_beta[i,:])**2*coeffs)
    
    var = np.sum(var)# * np.exp(-self.beta * self.energy))    
    
    return var 
    
#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,2, squeeze=False)
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 200
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = opt.V(Z)
lsp = np.linspace(np.min(ZZ),np.max(ZZ),15)
cf = ax[0,0].contourf(XX,YY,ZZ, levels=lsp)
#plt.colorbar(cf)
ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)

#plot points and quiver
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
# scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[2], s=30)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
#scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[0], s=50)
ax[0,0].scatter(z[:,0], z[:,1], marker='d', color=colors[3], s=24)

time = 0.0
#%% main loop
variances = []
var = compute_kernelized_variance(opt)
variances.append(var)

for i in range(conf.T):
    # plot
    if i%100 == 0:
        scx.set_offsets(opt.x[:, 0:2])
        # scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        # plt.title('Time = ' + str(time) + ' beta: ' + str(opt.beta) + ' kappa: ' + str(opt.kernel.kappa))
        plt.title('Time = ' + str(time))
        ax[0,1].plot(variances)
        plt.pause(0.1)
        plt.show()
        
        if conf.save2disk is True and i in snapshots:
            fig.savefig(cur_path+"\\visualizations\\Rastrigin\\Rastrigin-i-" \
                        + str(i) + "-kappa-" + str(conf.kappa)  \
                           + ".pdf",bbox_inches="tight")
    
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
    beta_sched.update()
    variances.append(compute_kernelized_variance(opt))
