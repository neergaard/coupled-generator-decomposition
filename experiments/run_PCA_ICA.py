import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from ica import ica1
#import matplotlib.pyplot as plt
#from TMMSAA.TMMSAA import SSE
torch.set_num_threads(16)

modality_names = ["EEG", "MEG"]

Xtrain_mmmsmc = {} # split by modality, subject, condition
Xtrain_mmms = {} # split by modality, subject
Xtrain_mm = {} # split by modality
for m in modality_names:
    Xtrain_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
    Xtrain_mmms[m] = torch.cat((Xtrain_mmmsmc[m][:, 0], Xtrain_mmmsmc[m][:, 1],Xtrain_mmmsmc[m][:, 2],),dim=-1)
    Xtrain_mm[m] = torch.reshape(Xtrain_mmms[m],(16*Xtrain_mmms[m].shape[-2],540))
Xtrain_group_daa = torch.cat((Xtrain_mm['EEG'],Xtrain_mm['MEG']),dim=-2)
Xtrain = {'group_daa':Xtrain_group_daa,'mm_daa':Xtrain_mm,'mmms_daa':Xtrain_mmms}

Xtest_mmmsmc = {} # split by modality, subject, condition
Xtest_mmms = {} # split by modality, subject
Xtest_mm = {} # split by modality
for m in modality_names:
    Xtest_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
    Xtest_mmms[m] = torch.cat((Xtest_mmmsmc[m][:, 0], Xtest_mmmsmc[m][:, 1],Xtest_mmmsmc[m][:, 2],),dim=-1)
    Xtest_mm[m] = torch.reshape(Xtest_mmms[m],(16*Xtest_mmms[m].shape[-2],540))
Xtest_group_daa = torch.cat((Xtest_mm['EEG'],Xtest_mm['MEG']),dim=-2)
Xtest = {'group_daa':Xtest_group_daa,'mm_daa':Xtest_mm,'mmms_daa':Xtest_mmms}

Xtraintilde_mmmsmc = {} # split by modality, subject, condition
Xtraintilde_mmms = {} # split by modality, subject
Xtraintilde_mm = {} # split by modality
for m in modality_names:
    Xtraintilde_mmmsmc[m] = torch.load("data/concatenatedData/Xf_" + m + "_FT0.pt")
    Xtraintilde_mmms[m] = torch.cat((Xtraintilde_mmmsmc[m][:, 0], Xtraintilde_mmmsmc[m][:, 1],Xtraintilde_mmmsmc[m][:, 2],),dim=-1)
    Xtraintilde_mm[m] = torch.reshape(Xtraintilde_mmms[m],(16*Xtraintilde_mmms[m].shape[-2],540))
Xtraintilde_group_daa = torch.cat((Xtraintilde_mm['EEG'],Xtraintilde_mm['MEG']),dim=-2)
Xtraintilde = {'group_daa':Xtraintilde_group_daa,'mm_daa':Xtraintilde_mm,'mmms_daa':Xtraintilde_mmms}

daa_modeltypes=['group_daa','mm_daa','mmms_daa']
num_modalities=[1,2,2]
times = torch.load("data/MEEGtimes.pt")
dims = {'group_daa':Xtrain_group_daa.shape,'mm_daa':Xtrain_mm["EEG"].shape,'mmms_daa':Xtrain_mmms["EEG"].shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
l2_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
#l2_vals = l2_vals[2:]

num_iter_outer = 5
num_iter_inner = 20

for K in range(2,31):

    # Model: group PCA
    U_group_pca,Sigma_group_pca,V_group_pca = torch.pca_lowrank(Xtrain['group_daa'],q=K,niter=100)
    group_pca_train_loss = torch.norm(Xtrain['group_daa']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2
    group_pca_test_loss = torch.norm(Xtest['group_daa']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2

    np.savetxt("data/PCAICA_results/train_test_loss_PCA_K="+str(K)+'.txt',np.array((group_pca_train_loss,group_pca_test_loss)),delimiter=',')

    # Model: group ICA
    A_group_ica,S_group_ica,_ = ica1(x_raw=Xtrain_group_daa,ncomp=K)
    group_ica_train_loss = torch.norm(Xtrain['group_daa']-A_group_ica@S_group_ica)**2
    group_ica_test_loss = torch.norm(Xtest['group_daa']-A_group_ica@S_group_ica)**2
    np.savetxt("data/PCAICA_results/train_test_loss_ICA_K="+str(K)+'.txt',np.array((group_ica_train_loss,group_ica_test_loss)),delimiter=',')
    
