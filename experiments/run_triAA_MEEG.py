import matplotlib.pyplot as plt
import torch
import numpy as np
from TMMSAA import TMMSAA,TMMSAA_trainer

modalitynames = ['EEG','MEG']
M = len(modalitynames)
#K = 5

# Full AA
X_train = {}
X_test = {}
X_train_cat = {}
X_test_cat = {}
X_train_l2 = {}
X_test_l2 = {}
X_train_l2_cat = {}
X_test_l2_cat = {}
for m in range(M):
    X_train[modalitynames[m]] = torch.load('data/X_'+modalitynames[m]+'_split0.pt')
    X_test[modalitynames[m]] = torch.load('data/X_'+modalitynames[m]+'_split1.pt')
    X_train_cat[modalitynames[m]] = torch.cat((X_train[modalitynames[m]][:,0],X_train[modalitynames[m]][:,1],X_train[modalitynames[m]][:,2]),dim=-1)
    X_test_cat[modalitynames[m]] = torch.cat((X_test[modalitynames[m]][:,0],X_test[modalitynames[m]][:,1],X_test[modalitynames[m]][:,2]),dim=-1)

    X_train_l2[modalitynames[m]] = torch.load('data/Xf_'+modalitynames[m]+'_split0.pt')
    X_test_l2[modalitynames[m]] = torch.load('data/Xf_'+modalitynames[m]+'_split1.pt')
    X_train_l2_cat[modalitynames[m]] = torch.cat((X_train_l2[modalitynames[m]][:,0],X_train_l2[modalitynames[m]][:,1],X_train_l2[modalitynames[m]][:,2]),dim=-1)
    X_test_l2_cat[modalitynames[m]] = torch.cat((X_test_l2[modalitynames[m]][:,0],X_test_l2[modalitynames[m]][:,1],X_test_l2[modalitynames[m]][:,2]),dim=-1)
    
times = torch.load('data/MEEGtimes.pt')
B,L,N,P = X_train['EEG'].shape
C_idx=torch.hstack((torch.zeros(20,dtype=torch.bool),torch.ones(160,dtype=torch.bool)))

# %% DAA fit

model_full = TMMSAA.TMMSAA(dimensions=(B,L,N,P), num_modalities=M,num_comp=5,model='DAA',C_idx=C_idx)
optimizer = torch.optim.Adam(model_full.parameters(), lr = 0.1)
loss_full = TMMSAA_trainer.Optimizationloop(model=model_full,X=X_train,Xtilde=X_train_l2,Optimizer=optimizer,max_iter=10000,tol=1e-16)

model_equalarchetypes = TMMSAA.TMMSAA(dimensions=(B,N,int(P*3)), num_modalities=M,num_comp=5,model='DAA',C_idx=torch.squeeze(torch.tile(C_idx,(1,3))))
optimizer = torch.optim.Adam(model_equalarchetypes.parameters(), lr = 0.1)
loss_equalarchetypes = TMMSAA_trainer.Optimizationloop(model=model_equalarchetypes,X=X_train_cat,Xtilde=X_train_l2_cat,Optimizer=optimizer,max_iter=10000,tol=1e-16)

# %% basic AA fit with lr=0.1 (figure of LRs in reports folder)
model_full = TMMSAA.TMMSAA(dimensions=(B,L,N,P), num_modalities=M,num_comp=6,model='AA',C_idx=C_idx)
optimizer = torch.optim.Adam(model_full.parameters(), lr = 0.1)
loss_full = TMMSAA_trainer.Optimizationloop(model=model_full,X=X_train,Optimizer=optimizer,max_iter=10000,tol=1e-16)

model_equalarchetypes = TMMSAA.TMMSAA(dimensions=(B,N,int(P*3)), num_modalities=M,num_comp=6,model='AA',C_idx=torch.squeeze(torch.tile(C_idx,(1,3))))
optimizer = torch.optim.Adam(model_equalarchetypes.parameters(), lr = 0.1)
loss_equalarchetypes = TMMSAA_trainer.Optimizationloop(model=model_equalarchetypes,X=X_train_cat,Optimizer=optimizer,max_iter=10000,tol=1e-16)


# %% test likelihood over K
loss_test_full = []
loss_test_equalarchetypes = []
K_eval = np.arange(2,10)
for K in K_eval:
    print(K)
    model_full = TMMSAA.TMMSAA(dimensions=(B,L,N,P), num_modalities=M,num_comp=K)
    optimizer = torch.optim.Adam(model_full.parameters(), lr = 0.1)
    loss_full = TMMSAA_trainer.Optimizationloop(model=model_full,X=X_train,Optimizer=optimizer,max_iter=10000,tol=1e-16)
    loss_test_full.append(model_full.eval_model(X_test))

    model_equalarchetypes = TMMSAA.TMMSAA(dimensions=(B,N,P*3), num_modalities=M,num_comp=K)
    optimizer = torch.optim.Adam(model_equalarchetypes.parameters(), lr = 0.1)
    loss_equalarchetypes = TMMSAA_trainer.Optimizationloop(model=model_equalarchetypes,X=X_train_cat,Optimizer=optimizer,max_iter=10000,tol=1e-16)
    loss_test_equalarchetypes.append(model_equalarchetypes.eval_model(X_test_cat))

plt.figure()
plt.plot(K_eval,loss_test_full,label='full')
plt.plot(K_eval,loss_test_equalarchetypes,label='equalarchetypes')
plt.legend()

# %% various plots

# plots over LR if evaluating LR
plt.figure()
for lr in loss_full:
    plt.plot(loss_full[lr],label=lr)
plt.legend()

# plots for one subject
C,S = model_full.get_model_params()

plt.figure()
plt.plot(loss_full)

plt.figure()
plt.plot(times[C_idx],C.detach())

fig,axs = plt.subplots(2,3)
bottom=torch.zeros(180)
for m in range(M):
    for l in range(L):
        for k in range(C.shape[1]):
            axs[m,l].bar(times,torch.mean(S[m,:,l,k,:].detach(),dim=0),width=(max(times)-min(times))/len(times),bottom=bottom)
            bottom += torch.mean(S[m,:,l,k,:].detach(),dim=0)

# plots for one subject
C,S = model_equalarchetypes.get_model_params()

plt.figure()
plt.plot(loss_equalarchetypes)

figC,ax = plt.subplots(3,1)
ax[0].plot(times[C_idx],C[0:160].detach())
ax[1].plot(times[C_idx],C[160:320].detach())
ax[2].plot(times[C_idx],C[320:].detach())

fig,axs = plt.subplots(2,3)
bottom=torch.zeros(180)
delims = (0,180,360,540)
for m in range(M):
    for l in range(L):
        for k in range(C.shape[1]):
            axs[m,l].bar(times,torch.mean(S[m,:,k,delims[l]:delims[l+1]].detach(),dim=0),width=(max(times)-min(times))/len(times),bottom=bottom)
            bottom += torch.mean(S[m,:,k,delims[l]:delims[l+1]].detach(),dim=0)

y=7