import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import polarcbo as pcbo
import polarcbo.dynamic as dyn
import os
#%%
cur_path = os.path.dirname(os.path.realpath(__file__)) #%% for save2disk=True
#%% set parameters
conf = pcbo.utils.config()
conf.save2disk = False
conf.T = 3001
conf.tau=0.01
conf.x_max = 4.5
conf.x_min = -1
conf.random_seed = 24
conf.d = 2
conf.beta = 1.0
conf.kappa = 0.5 #*np.inf 
conf.kernel = pcbo.functional.Gaussian_kernel(kappa=conf.kappa)

conf.num_particles = 300


# target function
conf.V = pcbo.objectives.Banana()
#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = pcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
#%% init optimizer and scheduler
opt = dyn.PolarCBS(x, conf.V, beta=conf.beta, tau=conf.tau, mode="sampling",\
                   kernel=conf.kernel)   

#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 2000
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = opt.V(Z)
ZZ = np.exp(-opt.beta * opt.V(Z))

# lsp = np.linspace(np.min(ZZ),np.max(ZZ),30)
cf = ax[0,0].contourf(XX,YY,ZZ)#, levels=lsp)
#plt.colorbar(cf)
ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)

snapshots = [0, 100, 1000, 3000]
#plot points and quiver
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[0], s=50)

time = 0.0
#%% main loop
if conf.save2disk:
    path = cur_path+"\\visualizations\\Banana_1\\"
    os.makedirs(path, exist_ok=True) 

for i in range(conf.T):
    # plot
    if i%100 == 0:
        scx.set_offsets(opt.x[:, 0:2])
        scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        plt.title('Time = {:.2f}'.format(time))
        plt.pause(0.1)
        plt.show()        
        
        if conf.save2disk is True and i in snapshots:
            fig.savefig(path + "Banana-i-" + str(i) + "-kappa-" + str(conf.kappa) + ".pdf",bbox_inches="tight")
    
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
