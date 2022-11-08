import numpy as np
import matplotlib.pyplot as plt

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
conf.x_max = 7
conf.x_min = -7
conf.random_seed = 225
conf.d = 30
conf.beta=30
conf.sigma=8.1

conf.kappa = .1#0.000755

conf.heavy_correction = False
conf.num_particles = 1000
conf.factor = 1.01
conf.noise = pcbo.noise.comp_noise(tau=conf.tau)
conf.eta = 0.5
conf.kernel = pcbo.kernels.Gaussian_kernel(kappa=conf.kappa)

conf.repulsion_scale = 5.

conf.optim = "CCBO"
conf.num_means = 5
conf.M = int(conf.num_particles*.9)
#conf.kernel = ut.Taz_kernel(kappa=0.0001)
# conf.kernel = ut.Vesuvio_kernel(kappa=0.001*conf.kappa)



# target function
uni_modal = False
if uni_modal:   
    z = np.array([[3.,2.]])
    alphas = np.array([1])
    z = np.pad(z, [[0,0], [0,conf.d-2]])
else:
    #z = np.array([[-4.,-2],[3.,-4.],[4.,2.]])
    z = np.zeros((3, conf.d))
    z[0,:] = np.array([[-2,1] for i in range(conf.d//2)]).ravel()
    z[1,:] = np.array([[2,-1] for i in range(conf.d//2)]).ravel()
    z[2,:] = np.array([[-1,-3] for i in range(conf.d//2)]).ravel()

    alphas = np.array([1,1,1])

#z[:, 2:] = np.random.uniform(low=conf.x_min, high=conf.x_max, size=(z.shape[0], conf.d-2))

conf.V = pcbo.objectives.Ackley_multimodal(alpha=alphas,z=z)

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
#beta_sched = ut.beta_eff(opt, eta=conf.eta, factor=conf.factor)
beta_sched = pcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e7)

#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)
#rc('font',**{'family':'serif','serif':['Times'],'size':14})
#rc('text', usetex=True)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 500
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = opt.V(Z)**0.1
lsp = np.linspace(np.min(ZZ),np.max(ZZ),15)
cf = ax[0,0].contourf(XX,YY,ZZ, levels=lsp)
#plt.colorbar(cf)
ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)

snapshots = [0, 100, 1000, 3000]
#plot points and quiver
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[2], s=30)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
#scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[0], s=50)
ax[0,0].scatter(z[:,0], z[:,1], marker='d', color=colors[3], s=24)

time = 0.0
save_plots = False
#%% main loop
   
for i in range(conf.T):
    # plot
    if i%20 == 0:
        scx.set_offsets(opt.x[:, 0:2])
        scm.set_offsets(opt.m_beta[:, 0:2])
        #scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        plt.title('Time = ' + str(time) + ' beta: ' + str(opt.beta) + ' kappa: ' + str(opt.kernel.kappa))
        plt.pause(0.1)
        plt.show()
        
        
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
    beta_sched.update()
