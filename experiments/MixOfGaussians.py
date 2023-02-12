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
import polarcbo.dynamic as dyn

#%%
cur_path = os.path.dirname(os.path.realpath(__file__))

#%% set parameters
conf = pcbo.utils.config()
conf.save2disk = False
conf.T = 4001
conf.tau=0.01
conf.x_max = 5
conf.x_min = -5
conf.random_seed = 24
conf.d = 2
conf.beta = 1.0
kappas = [0.4, 0.6, 0.8, np.inf]

conf.num_particles = 400




# target function
case = 'Far'

if case == 'Far':
    a = [1., 3.]    #center of first cluster
    b = [-1.2, -2.5] #center of second cluster
    conf.V = pcbo.objectives.Bimodal(a, b)
elif case == 'Close':
    conf.V = pcbo.objectives.Bimodal()
else: 
    raise ValueError('Unknown case')
    

#%%
for kappa in kappas:
    conf.kappa = kappa
    #%% initialize scheme
    np.random.seed(seed=conf.random_seed)
    x = pcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                          x_min=conf.x_min, x_max=conf.x_max)
    #%% init optimizer 
    opt = dyn.PolarCBS(x, conf.V, beta=conf.beta, tau=conf.tau, mode="sampling",\
                     kernel=pcbo.functional.Gaussian_kernel(kappa=conf.kappa))   
    
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
    ZZ = np.exp(-opt.V(Z))
    
    # lsp = np.linspace(np.min(ZZ),np.max(ZZ),30)
    cf = ax[0,0].contourf(XX,YY,ZZ)#, levels=lsp)
    #plt.colorbar(cf)
    ax[0,0].axis('square')
    ax[0,0].set_xlim(conf.x_min,conf.x_max)
    ax[0,0].set_ylim(conf.x_min,conf.x_max)
    
    snapshots = [0, 100, 1000, 2000, 3000, 4000]
    #plot points and quiver
    scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
    quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
    # scm = ax[0,0].scatter(opt.means[:,0], opt.means[:,1], marker='x', color=colors[2], s=50)
    # scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[2], s=50)
    
    time = 0.0
    #%% main loop
    if conf.save2disk:
        path = cur_path+"\\visualizations\\MixOfGaussians\\"
        os.makedirs(path, exist_ok=True) 
        
    for i in range(conf.T):
        # plot
        if i%20 == 0:
            scx.set_offsets(opt.x[:, 0:2])
            # scm.set_offsets(opt.means[:, 0:2])
            # scm.set_offsets(opt.m_beta[:, 0:2])
            quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
            quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
            plt.title('Time = {:.2f}'.format(time))
            plt.pause(0.1)
            plt.show()
            
            if conf.save2disk is True and i in snapshots:
                filename = cur_path + \
                    "MixOfGaussians" + "-" + case \
                        + "-i-" +str(i) + "-kappa-" + str(conf.kappa) \
                        + ".pdf"
                fig.savefig(filename,bbox_inches="tight")
        
        # update step
        time = conf.tau*(i+1)
        opt.step(time=time)
