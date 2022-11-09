import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import matplotlib as mpl
import matplotlib.cm as cmp


#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import polarcbo as pcbo
import polarcbo.particledynamic as pdyn

#%% set parameters
conf = pcbo.utils.config()
conf.T = 5000
conf.tau=0.01
conf.x_max = 2
conf.x_min = -2
conf.random_seed = 42
conf.d = 2
conf.beta = 1.
conf.sigma = 1.0

conf.kappa = 0.2

conf.heavy_correction = False
conf.num_particles = 500
conf.factor = 1.01
conf.noise = pcbo.noise.normal_noise(tau=conf.tau)#pcbo.noise.comp_noise(tau=conf.tau)
conf.eta = 0.5
conf.kernel = pcbo.functional.Gaussian_kernel(kappa=conf.kappa)

conf.repulsion_scale = 5.

conf.optim = "PolarCBO"
conf.num_means = 7
conf.M = int(conf.num_particles)
#%%
class Snowflake():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        x = self.alpha * x 
        r = np.linalg.norm(x,axis=-1)
        phi = np.arctan2(x[...,1], x[...,0])
        
        res = np.ones((x.shape[:-1]))
        for psi in [0, np.pi/3, np.pi*2/3]:
            g = r**8 - r**4 + np.abs(np.cos(phi+psi))**0.5*r**0.3
            res = np.minimum(res, g)
        
        res = np.minimum(res, .8)
        return res
    
conf.V = Snowflake(alpha=.55)
#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = pcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
#%% init optimizer and scheduler
if conf.optim == "PolarCBO":
    opt = pdyn.PolarCBO(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                        beta = conf.beta, kernel=conf.kernel)
else:
    opt = pdyn.CCBO(x, conf.V, conf.noise, num_means=conf.num_means, sigma=conf.sigma, tau=conf.tau,\
                          beta = conf.beta, kernel=conf.kernel,\
                          repulsion_scale = conf.repulsion_scale,
                          M=conf.M)
# beta_sched = ut.beta_eff(opt, eta=conf.eta, factor=conf.factor)
beta_sched = pcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e3)
#%%
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 500
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = conf.V(Z)#**0.1
lsp = np.min(ZZ) + (np.max(ZZ) - np.min(ZZ))*np.linspace(0, 1, 50)**5

cf = ax[0,0].contourf(XX,YY,ZZ, levels=lsp, cmap=cmp.get_cmap('Blues'))
#plt.colorbar(cf)
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color='w', s=12, alpha=.5)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color='w', scale=20)
scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='2', color=colors[3], s=50, alpha=.5)

ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)
plt.colorbar(cf)

time = 0.0
save_plots = False

plot_mod = 1

#%% main loop
for i in range(conf.T):
    
   
    # plot
    if i%plot_mod== 0:
        plot_mod = min(2*plot_mod, 20)
        scx.set_offsets(opt.x[:, 0:2])
        scm.set_offsets(opt.m_beta[:, 0:2])
        #scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        plt.title('Time = ' + str(time) + ' beta: ' + str(opt.beta) + ' kappa: ' + str(opt.kernel.kappa))
        plt.pause(0.5)
        plt.show()
        
        # if i in snapshots and save_plots:
        #     fig.savefig(cur_path+"\\visualizations\\Ackley\\Ackley-i-" + str(i) + "-kappa-" + str(conf.kappa) + ".pdf",bbox_inches="tight")
    
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
    beta_sched.update()
