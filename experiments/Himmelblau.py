import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import polarcbo as pcbo
import polarcbo.dynamic as dyn


#%% set parameters
conf = pcbo.utils.config()
conf.save2disk = False
conf.T = 3000
conf.tau=0.01
conf.x_max = 6
conf.x_min = -6
conf.random_seed = 309
conf.d = 2
conf.V = pcbo.objectives.Himmelblau()
conf.beta=1.0
conf.sigma=1.0
conf.heavy_correction = False
conf.num_particles = 300
conf.factor = 1.0
conf.eta = 0.5

conf.noise = pcbo.noise.normal_noise(tau=conf.tau)
conf.kappa = 0.8 #np.inf
conf.kernel = pcbo.functional.Gaussian_kernel(kappa=conf.kappa)


#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = pcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
#%% init optimizer and scheduler
opt = dyn.PolarCBO(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                    kernel=conf.kernel)
# opt = dyn.CCBO(x, conf.V, conf.noise, beta=conf.beta, tau=conf.tau,\
#                       sigma=conf.sigma, kernel=conf.kernel,\
#                       num_means = 4, repulsion_scale = 1)    
beta_sched = pcbo.scheduler.beta_eff(opt, eta=conf.eta, factor=conf.factor)

#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 1000
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = opt.V(Z)**0.2
lsp = np.linspace(0,5.5,15)
cf = ax[0,0].contourf(XX,YY,ZZ, levels=lsp)
#plt.colorbar(cf)
ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)

snapshots = [0, 100, 1000]
#plot points and quiver
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
#scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[0], s=50)

time = 0.0
#%% main loop
if conf.save2disk:
    path = cur_path+"\\visualizations\\Himmelblau\\"
    os.makedirs(path, exist_ok=True) 
    
for i in range(conf.T):
    # plot
    if i%100 == 0:
        scx.set_offsets(opt.x[:, 0:2])
        #scm.set_offsets(opt.m_beta[:, 0:2])
        quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
        quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
        plt.title('Time = {:.2f}'.format(time))
        plt.pause(0.1)
        plt.show()
        
        if conf.save2disk is True and i in snapshots:
            fig.savefig(path+"Himmelblau-i-" + str(i) + "-kappa-" + str(conf.kappa) + ".pdf",bbox_inches="tight")
    
    # update step
    time = conf.tau*(i+1)
    opt.step(time=time)
    beta_sched.update()
        