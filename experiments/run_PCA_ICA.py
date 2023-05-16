import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from load_data import load_data
from sklearn.decomposition import FastICA
#import matplotlib.pyplot as plt
#from TMMSAA.TMMSAA import SSE

modality_names = ["EEG", "MEG"]

Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

dims = {'group_spca':Xtrain['group_spca'].shape,'mm_spca':Xtrain['mm_spca']["EEG"].shape,'mmms_spca':Xtrain['mmms_spca']["EEG"].shape}

num_iter_outer = 5
num_iter_inner = 50

for K in range(2,31):

    # Model: group PCA
    U_group_pca,Sigma_group_pca,V_group_pca = torch.pca_lowrank(Xtrain['group_spca'],q=K,niter=100)
    group_pca_train_loss = torch.norm(Xtrain['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2
    group_pca_test_loss = torch.norm(Xtest['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2

    np.savetxt("data/PCAICA_results/train_test_loss_PCA_K="+str(K)+'.txt',np.array((group_pca_train_loss,group_pca_test_loss)),delimiter=',')

    # Model: group ICA
    ica = FastICA(n_components=K)
    X_transformed = ica.inverse_transform(ica.fit_transform(Xtrain['group_spca']))
    group_ica_train_loss = torch.norm(Xtrain['group_spca']-X_transformed)**2
    group_ica_test_loss = torch.norm(Xtest['group_spca']-X_transformed)**2
    # A_group_ica,S_group_ica,_ = ica1(x_raw=Xtrain['group_spca'],ncomp=K)
    # group_ica_train_loss = torch.norm(Xtrain['group_spca']-A_group_ica@S_group_ica)**2
    # group_ica_test_loss = torch.norm(Xtest1['group_spca']-A_group_ica@S_group_ica)**2
    np.savetxt("data/PCAICA_results/train_test_loss_ICA_K="+str(K)+'.txt',np.array((group_ica_train_loss,group_ica_test_loss)),delimiter=',')
    
    print('Done with '+str(K))
