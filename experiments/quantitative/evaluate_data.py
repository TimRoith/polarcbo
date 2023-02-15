import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from polarcbo import utils as ut

#%% get files and extract data
cur_path = os.path.dirname(os.path.realpath(__file__))
data_path = cur_path + "\\data\\Ackley-Exp-30d-CCBO"

lst = os.listdir(data_path)
lst.sort()
files = [f for f in lst if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]

conf = ut.config()

found_hists = []

for j in range(len(files)):
    skip_ctr = 0
    minimas = []
    file_idx = j
    
    with open(data_path + '\\' +  files[file_idx]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if row[0] == "minima":
                local_minima = []
                for d in range(conf.d):
                    local_minima.append(float(row[1+d]))
                minimas.append(local_minima)
                skip_ctr+=1
    
            elif row[0] != "0":
                try:
                    num = float(row[1])
                    if num.is_integer():
                        num = int(num)
                    setattr(conf, row[0], num)
                except:
                    pass
                skip_ctr+=1
            else:
                break
            
    G = np.genfromtxt(data_path +  '\\' +  files[file_idx], delimiter=',', skip_header=skip_ctr)
    G = G[:,1:]
    m_betas = np.zeros((conf.num_runs, conf.num_particles, conf.d))
    for i in range(conf.num_runs):
        m_betas[i, :, :] = G[conf.d * i: conf.d * (i+1), :].T
    
    minimas = np.array(minimas)   
    num_minimas = minimas.shape[0] 
    
    #%%
    thresh = 0.25
    hist = np.zeros((num_minimas+1))
    found = []
    
    for num_run in range(conf.num_runs):
        m = m_betas[num_run,:,:]
        dists = np.linalg.norm(np.reshape(m, (conf.num_particles,1,conf.d)) - np.reshape(minimas, (1,num_minimas,conf.d)),axis=2, ord=np.inf)
        dists_min_idx = np.argmin(dists, axis=1)
        succes_idx = np.where(dists[np.arange(conf.num_particles), dists_min_idx] < thresh, dists_min_idx,num_minimas)
        local_hist = np.bincount(succes_idx, minlength=num_minimas+1)
        hist += local_hist
        
        found_minmas = 0
        sc = np.unique(succes_idx)
        for j in range(num_minimas):
            if j in sc:
                found_minmas += 1
                
        found.append(found_minmas)
    print(hist/np.sum(hist))
        
        
    #%%
    
    hist = hist/np.sum(hist)
    found_hist = 100*(np.bincount(found, minlength=num_minimas+1)/conf.num_runs)
    found_hists.append([conf.kappa, conf.num_particles] + list(found_hist))
    
    show_plots = False
    if show_plots:
        plt.bar(np.arange(4),found_hist)
        plt.title('kappa: ' + str(conf.kappa) + ' J: '+str(conf.num_particles))
    #print("<>"*20)
    #print("Evaluation for kappa: " + str(conf.kappa) + ' J: '+str(conf.num_particles))
    #print('Found Minimas: ' + str(found_hist))
    
#%%
sfh = sorted(found_hists)
file = data_path + "\\eval_table.txt"
old_kappa = None
with open(file, 'w') as f:
    for j in range(len(sfh)):
        new_kappa = sfh[j][0]
        if new_kappa != old_kappa:
            if new_kappa == np.inf:
                kappa_str = '\infty'
            else:
                kappa_str = str(new_kappa)
            
            f.write('\\\ \n')
            f.write('%' + '<>'*40 + '\n')
            f.write('$\kappa = ' + kappa_str + "$")
            old_kappa = new_kappa
            
            
        for i in range(3, len(sfh[j])):
            f.write(' & ')
            val = sum(sfh[j][i:])
            f.write("$" + str(int(val)) + "\%$")
    f.write('\\\ \n')
    f.write('\hline')
            
            
        