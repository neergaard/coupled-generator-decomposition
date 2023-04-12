import sys
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
import matplotlib.pyplot as plt

modeltypes=['group_spca','mm_spca','mmms_spca']
num_modalities=[1,2,2]
    
modality_names = ["EEG", "MEG"]

X_train_mmmsmc = {} # split by modality, subject, condition
X_train_mmms = {} # split by modality, subject
X_train_mm = {} # split by modality
for m in modality_names:
    X_train_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob.pt")
    X_train_mmms[m] = torch.cat((X_train_mmmsmc[m][:, 0], X_train_mmmsmc[m][:, 1],X_train_mmmsmc[m][:, 2],),dim=-1)
    X_train_mm[m] = torch.reshape(X_train_mmms[m],(16*X_train_mmms[m].shape[-2],540))

X_train_group_spca = torch.cat((X_train_mm['EEG'],X_train_mm['MEG']),dim=-2)

X = {'group_spca':X_train_group_spca,'mm_spca':X_train_mm,'mmms_spca':X_train_mmms}

times = torch.load("data/MEEGtimes.pt")
dims = {'group_spca':X_train_group_spca.shape,'mm_spca':X_train_mm["EEG"].shape,'mmms_spca':X_train_mmms["EEG"].shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = torch.logspace(-3,2,11)
#l1_vals = l1_vals[0:1]
l2 = 0.01
K = 5

num_iter_outer = 1
num_iter_inner = 10

# Group pca initialization
_,_,V = torch.pca_lowrank(X['group_spca'],q=K)
init0 = {'Bp':torch.nn.functional.relu(V),'Bn':torch.nn.functional.relu(-V)}

modeltype_best_models = []
best_models = []
best_losses = np.zeros((3,num_iter_outer))
for m,modeltype in enumerate(modeltypes):
    if m==0:
        continue
    for outer in range(num_iter_outer):
        best_loss = 10000000
        for inner in range(num_iter_inner):
            fitloss = np.zeros(len(l1_vals))
            for l1,lambda1 in enumerate(l1_vals):
                if l1==0:
                    model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_modalities=num_modalities[m],num_comp=K,model='SPCA',lambda1=lambda1,lambda2=l2,init=init0)
                else:
                    model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_modalities=num_modalities[m],num_comp=K,model='SPCA',lambda1=lambda1,lambda2=l2,init=init)
                optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
                #optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                loss = TMMSAA_trainer.Optimizationloop(model=model,X=X[modeltype],Optimizer=optimizer,max_iter=100,tol=1e-6)
                C,S,Bp,Bn = model.get_model_params(X=X[modeltype])
                init={'Bp':Bp,'Bn':Bn}
                fitloss[l1] = model.eval_model(X[modeltype],X[modeltype])

                print('Done with '+modeltype+', outer='+str(outer)+', inner='+str(inner)+', l1_idx='+str(l1))
                
            if loss[-1]<best_loss:
                best_loss = loss[-1]
                best_model = model
        
        best_models.append(best_model)
        best_losses[m,outer] = best_loss
    best_of_best = np.argmin(best_losses[m])
    modeltype_best_models.append(best_models[best_of_best])
    
# plotting
h=7