import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca']
modality_names = ["EEG", "MEG"]

l1_vals = np.logspace(-3,2,11)
l2_vals = np.logspace(-5,0,11)

num_iter_outer = 5
num_iter_inner = 100

num_comps = np.arange(2,11)

fix,ax = plt.subplots(3,len(num_comps))
best_loss = np.zeros((3,len(num_comps)))

for m,modeltype in enumerate(modeltypes):
    for k,K in enumerate(num_comps):
        loss = np.zeros((num_iter_outer,len(l1_vals),len(l2_vals)))
        for outer in range(num_iter_outer):
            allinner = np.zeros((num_iter_inner,len(l1_vals),len(l2_vals)))
            for inner in range(num_iter_inner):
                try:
                    allinner[inner]=np.genfromtxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                except:
                    allinner[inner] = np.nan
            loss[outer] = np.nanmin(allinner,axis=0)
        ax[m,k].imshow(np.nanmean(loss,axis=0),vmin=0,vmax=100)
        ax[m,k].set_title(modeltype+', K='+str(K))
        if m==2:
            ax[m,k].set_xticks(ticks=np.arange(len(l1_vals)),labels=l1_vals)
        else:
            ax[m,k].set_xticks([])
        if k==0:
            ax[m,k].set_yticks(ticks=np.arange(len(l2_vals)),labels=l2_vals)
        else:
            ax[m,k].set_yticks([])
        if k==9:
            ax[m,k].colorbar()
        
        best_loss[m,k] = np.nanmin(np.nanmean(loss,axis=0))

plt.figure()
plt.plot(num_comps,best_loss.T)
h = 7
