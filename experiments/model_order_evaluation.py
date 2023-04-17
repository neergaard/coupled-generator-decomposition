import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca','pca','ica','daa']
modality_names = ["EEG", "MEG"]

num_iter_outer = 5
num_iter_inner = 20

num_comps = np.arange(2,21)

best_train_loss = np.zeros((len(modeltypes),len(num_comps)))
best_test_loss = np.zeros((len(modeltypes),len(num_comps)))

for k,K in enumerate(num_comps):
    for m,modeltype in enumerate(modeltypes):

        if modeltype is 'pca':
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_PCA_K='+str(K)+'.txt')
            best_train_loss[m,k] = loss[0]
            best_test_loss[m,k] = loss[1]
        elif modeltype is 'ica':
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_ICA_K='+str(K)+'.txt')
            best_train_loss[m,k] = loss[0]
            best_test_loss[m,k] = loss[1]
        elif modeltype is 'daa':
            train_loss = np.genfromtxt('data/DAA_results/train_loss_ICA_K='+str(K)+'.txt')


        loss = np.zeros((num_iter_outer))
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
            ax[m,k].set_xticks(ticks=np.arange(len(l1_vals)),labels=l1_vals,rotation = 45)
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
