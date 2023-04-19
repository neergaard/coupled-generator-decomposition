import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from sklearn.decomposition import SparsePCA
#from ica import ica1
#from TMMSAA.TMMSAA import SSE
torch.set_num_threads(16)

modeltype = 'group_spca_lars'

modality_names = ["EEG", "MEG"]

Xtrain_mmmsmc = {} # split by modality, subject, condition
Xtrain_mmms = {} # split by modality, subject
Xtrain_mm = {} # split by modality
for m in modality_names:
    Xtrain_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
    Xtrain_mmms[m] = torch.cat((Xtrain_mmmsmc[m][:, 0], Xtrain_mmmsmc[m][:, 1],Xtrain_mmmsmc[m][:, 2],),dim=-1)
    Xtrain_mm[m] = torch.reshape(Xtrain_mmms[m],(16*Xtrain_mmms[m].shape[-2],540))
Xtrain_group_spca = torch.cat((Xtrain_mm['EEG'],Xtrain_mm['MEG']),dim=-2)
Xtrain = {'group_spca_lars':np.array(Xtrain_group_spca)}

Xtest_mmmsmc = {} # split by modality, subject, condition
Xtest_mmms = {} # split by modality, subject
Xtest_mm = {} # split by modality
for m in modality_names:
    Xtest_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
    Xtest_mmms[m] = torch.cat((Xtest_mmmsmc[m][:, 0], Xtest_mmmsmc[m][:, 1],Xtest_mmmsmc[m][:, 2],),dim=-1)
    Xtest_mm[m] = torch.reshape(Xtest_mmms[m],(16*Xtest_mmms[m].shape[-2],540))
Xtest_group_spca = torch.cat((Xtest_mm['EEG'],Xtest_mm['MEG']),dim=-2)
Xtest = {'group_spca_lars':np.array(Xtest_group_spca)}

times = torch.load("data/MEEGtimes.pt")
dims = {'group_spca_lars':Xtrain_group_spca.shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,2,8))))
l2_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,2,8))))
#l2_vals = l2_vals[2:]

num_iter_outer = 1
num_iter_inner = 10

# Model: group PCA
#U_group_pca,Sigma_group_pca,V_group_pca = torch.pca_lowrank(Xtrain['group_spca'],q=K,niter=100)
#init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}
#group_pca_train_loss = torch.norm(Xtrain['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2
#group_pca_test_loss = torch.norm(Xtest['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2

# Model: group ICA
#A_group_ica,S_group_ica,_ = ica1(x_raw=Xtrain_group_spca,ncomp=K)
#group_ica_train_loss = torch.norm(Xtrain['group_spca']-A_group_ica@S_group_ica)**2
#group_ica_test_loss = torch.norm(Xtest['group_spca']-A_group_ica@S_group_ica)**2

for K in range(2,31):
    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            print(str(K)+'_'+str(outer)+'_'+str(inner))

            all_train_loss = np.zeros((1,1))
            all_test_loss = np.zeros((1,1))
            for l2,lambda2 in enumerate(l2_vals):
                for l1,lambda1 in enumerate(l1_vals):
                    if l2>0 or l1>0:
                        continue    
                    model = SparsePCA(n_components=K,alpha=lambda1,ridge_alpha=lambda2,method='lars')
                    fit = model.inverse_transform(model.fit_transform(Xtrain[modeltype]))
                    loss_train = np.linalg.norm(Xtrain[modeltype]-fit)**2
                    loss_test = np.linalg.norm(Xtest[modeltype]-fit)**2
                    
                    all_test_loss[l2,l1] = loss_test
                    all_train_loss[l2,l1] = loss_train
            np.savetxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_train_loss,delimiter=',')
            np.savetxt("data/SPCA_results/test_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test_loss,delimiter=',')
