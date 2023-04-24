import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from load_data import load_data
import matplotlib.pyplot as plt
#from ica import ica1
#from TMMSAA.TMMSAA import SSE
modeltypes = ['group_spca','mm_spca','mmms_spca']
num_modalities = [1,2,2]

Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

dims = {'group_spca':Xtrain['group_spca'].shape,'mm_spca':Xtrain['mm_spca']["EEG"].shape,'mmms_spca':Xtrain['mmms_spca']["EEG"].shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = torch.logspace(-5,-2,4)
#l2_vals = torch.logspace(-5,2,8)
lambda2 = 0.01

num_iter_outer = 5
num_iter_inner = 5

K=5
modeltype = modeltypes[2]

# Model: group PCA
#_,_,V_group_pca = torch.pca_lowrank(Xtrain['group_spca'],q=K,niter=100)
#init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}

for outer in range(num_iter_outer):
    losses = np.zeros((num_iter_inner,4))
    for inner in range(num_iter_inner):
        for l1,lambda1 in enumerate(l1_vals):
            if l1==0:
                model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=None)
            else:
                model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-4,threshold_mode='abs',min_lr=0.0001,patience=100)
            loss,best_loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],optimizer=optimizer,scheduler=scheduler,max_iter=30000,tol=1e-4)
            C,S,Bp,Bn = model.get_model_params(X=Xtrain[modeltype])
            init={'Bp':Bp,'Bn':Bn}

        losses[inner,0]=best_loss
        losses[inner,1]=model.eval_model(Xtrain=Xtrain1[modeltype],Xtraintilde=Xtrain1[modeltype],Xtest=Xtest1[modeltype])
        losses[inner,2]=model.eval_model(Xtrain=Xtrain2[modeltype],Xtraintilde=Xtrain2[modeltype],Xtest=Xtest2[modeltype])
        losses[inner,3]=model.eval_model(Xtrain=Xtrain[modeltype],Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
        
        np.savetxt("data/SPCA_results_selectedregu/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_train_loss,delimiter=',')
            
        h=7