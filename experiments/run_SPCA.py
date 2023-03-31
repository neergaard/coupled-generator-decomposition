import matplotlib.pyplot as plt
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer, visualize_AA_trajectory

modality_names = ["EEG", "MEG"]
M = len(modality_names)
model_types = ['group_spca','mm_spca','mmms_spca']
num_modalities = [1,2,2]

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
B, N, P = X_train_mmms["EEG"].shape
dims = {'group_spca':X_train_group_spca.shape,'mm_spca':X_train_mm["EEG"].shape,'mmms_spca':X_train_mmms["EEG"].shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = torch.logspace(-3,2,11)
l2_vals = torch.logspace(-3,2,11)

num_iter_outer = 5
num_iter_inner = 100

num_comp = torch.arange(2,11)




for outer in range(num_iter_outer):
    for inner in range(num_iter_inner):
        for m1,modeltype in enumerate(model_types):
            for k1,K in enumerate(num_comp):
                all_loss = torch.zeros(len(l2_vals),len(l1_vals))
                for l2,lambda2 in enumerate(l2_vals):
                    for l1,lambda1 in enumerate(l1_vals):
                        if l1==0:
                            model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities[m1],model='SPCA',lambda1=lambda1,lambda2=lambda2,init=None)
                        else:
                            model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities[m1],model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                        loss = TMMSAA_trainer.Optimizationloop(model=model,X=X[modeltype],Optimizer=optimizer,max_iter=10000,tol=1e-3)
                        C,S,Bp,Bn = model.get_model_params(X=X[modeltype])
                        init={'Bp':Bp,'Bn':Bn}
                        all_loss[l2,l1] = loss[-1]
                
