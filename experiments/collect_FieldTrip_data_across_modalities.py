import torch
import numpy as np
import mne
import matplotlib.pyplot as plt

num_subjects = 16
conditions = ['Famous','Scrambled','Unfamiliar']
for split in range(2):
    # %% evoked potentials, frobenius normalized
    X_EEG = torch.zeros(num_subjects,len(conditions),70,180,dtype=torch.double)
    X_MEG = torch.zeros(num_subjects,len(conditions),102,180,dtype=torch.double)

    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            
            # EEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'EEG'+cond+'_frobnorm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_EEG[sub,cond_idx] = data

            # MEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'MEGMAG'+cond+'_frobnorm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_MEG[sub,cond_idx] = data

    torch.save(X_EEG,'data/concatenatedData/X_EEG_FT_frob'+str(split)+'.pt')
    torch.save(X_MEG,'data/concatenatedData/X_MEG_FT_frob'+str(split)+'.pt')

for split in range(2):
    # %% evoked potentials, l2 normalized
    X_EEG = torch.zeros(num_subjects,len(conditions),70,180,dtype=torch.double)
    X_MEG = torch.zeros(num_subjects,len(conditions),102,180,dtype=torch.double)

    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            
            # EEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'EEG'+cond+'_l2norm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_EEG[sub,cond_idx] = data

            # MEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'MEGMAG'+cond+'_l2norm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_MEG[sub,cond_idx] = data

    torch.save(X_EEG,'data/concatenatedData/X_EEG_FT_l2'+str(split)+'.pt')
    torch.save(X_MEG,'data/concatenatedData/X_MEG_FT_l2'+str(split)+'.pt')

# %% flipped EEG, l2 normalized
X_EEG = torch.zeros(2,num_subjects,len(conditions),70,180,dtype=torch.double)
    
for split in range(2):
    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            # EEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'EEG'+cond+'_l2norm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_EEG[split,sub,cond_idx] = data
        
U,S,V = torch.pca_lowrank(torch.mean(X_EEG,(0,1,2))-torch.mean(torch.mean(X_EEG,(0,1,2))),q=1)
score = (torch.mean(X_EEG,(0,1,2))-torch.mean(torch.mean(X_EEG,(0,1,2))))@V

for split in range(2):
    Xf_EEG = torch.zeros(num_subjects,len(conditions),70,180,dtype=torch.double)
    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            projections = torch.sum(X_EEG[split,sub,cond_idx]*score,dim=0)
            to_flip = 2*(projections < 0) -1
            Xf_EEG[sub,cond_idx] = X_EEG[split,sub,cond_idx]*to_flip
    torch.save(Xf_EEG,'data/concatenatedData/Xf_EEG_FT'+str(split)+'.pt')

# %% flipped MEG, l2 normalized
X_MEG = torch.zeros(2,num_subjects,len(conditions),102,180,dtype=torch.double)
    
for split in range(2):
    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            # MEG
            data = np.genfromtxt('data/FieldTripProcessed/sub'+str(sub+1)+'MEGMAG'+cond+'_l2norm'+str(split+1)+'_12-Apr-2023.csv',delimiter=',')
            data = torch.tensor(data)
            X_MEG[split,sub,cond_idx] = data
        
U,S,V = torch.pca_lowrank(torch.mean(X_MEG,(0,1,2))-torch.mean(torch.mean(X_MEG,(0,1,2))),q=1)
score = (torch.mean(X_MEG,(0,1,2))-torch.mean(torch.mean(X_MEG,(0,1,2))))@V

for split in range(2):
    Xf_MEG = torch.zeros(num_subjects,len(conditions),102,180,dtype=torch.double)
    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            projections = torch.sum(X_MEG[split,sub,cond_idx]*score,dim=0)
            to_flip = 2*(projections < 0) -1
            Xf_MEG[sub,cond_idx] = X_MEG[split,sub,cond_idx]*to_flip
    torch.save(Xf_MEG,'data/concatenatedData/Xf_MEG_FT'+str(split)+'.pt')