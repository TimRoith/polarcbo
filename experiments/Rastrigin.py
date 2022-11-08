import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import polarcbo as pcbo
import polarcbo.particledynamic as pdyn

#%%
cur_path = os.path.dirname(os.path.realpath(__file__))

#%% set parameters
conf = pcbo.utils.config()
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
conf.noise = pcbo.noise.normal_noise(tau=conf.tau)
conf.eta = 0.5


kernel = 'Gaussian'

if kernel == 'Gaussian':
    conf.kappa = 0.5
    conf.kernel = pcbo.kernels.Gaussian_kernel(kappa=conf.kappa)    
elif kernel == 'Laplace':
    conf.kappa = .05
    conf.kernel = pcbo.kernels.Laplace_kernel(kappa=conf.kappa)    
elif kernel == 'Constant':
    conf.kappa = 2
    conf.kernel = pcbo.kernels.Constant_kernel(kappa=conf.kappa)        
elif kernel == 'InverseQuadratic':
    conf.kappa = 1e-7
    conf.kernel = pcbo.kernels.InverseQuadratic_kernel(kappa=conf.kappa)        
else:
    raise ValueError('Unknown kernel')


snapshots = [0, 100, 500, 1000, 2000]


z = np.array([[3.,2.],[0,0], [-1,-3.5]])
alphas = np.array([1,1,1])

conf.V = pcbo.objectives.Rastrigin_multimodal(alpha=alphas,z=z)

#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = pcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
    
#%% init optimizer and scheduler
opt = pdyn.polarcbo(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                       beta = conf.beta, kernel=conf.kernel)

beta_sched = pcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e7)

#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)
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
ax[0,0].scatter(z[:,0], z[:,1], marker='D', s=50, facecolors='none', edgecolors=colors[2])

time = 0.0
#%% main loop
if conf.save2disk:
    path = cur_path+"\\visualizations\\Rastrigin\\"
    os.makedirs(path, exist_ok=True) 
    
for i in range(conf.T):
    # plot
    if i%100 == 0:
        scx.set_offsets(opt.x[:, 0:2])
        # scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        # plt.title('Time = ' + str(time) + ' beta: ' + str(opt.beta) + ' kappa: ' + str(opt.kernel.kappa))
        plt.title('Time = ' + str(time))
        plt.pause(0.1)
        plt.show()
        
        if conf.save2disk is True and i in snapshots:
            fig.savefig(path+"Rastrigin-i-" \
                        + str(i) + "-kappa-" + str(conf.kappa)  \
                           + "-kernel-" + kernel + ".pdf",bbox_inches="tight")
    
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
    beta_sched.update()
