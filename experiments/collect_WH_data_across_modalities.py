import torch
import mne
import matplotlib.pyplot as plt

num_subjects = 16
conditions = ['famous','scrambled','unfamiliar']
# %% evoked potentials, frobenius normalized
for split in range(2):
    X_EEG = torch.zeros(num_subjects,len(conditions),70,180,dtype=torch.double)
    X_MEG = torch.zeros(num_subjects,len(conditions),102,180,dtype=torch.double)

    
    for sub in range(num_subjects):
        for cond_idx,cond in enumerate(conditions):
            
            sub_id = f'{(sub+1):02d}'
            
            # EEG
            evo = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                   +"task-facerecognition_proc-p_cond-"+cond+"_split-"+str(split)+"_evo.fif")
            evo = evo[0]
            evo.pick('eeg')
            data = torch.tensor(evo.data)
            X_EEG[sub,cond_idx,:,:] = data/torch.norm(data,p='fro')

            # MEG
            evo = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                   +"task-facerecognition_proc-p_cond-"+cond+"_split-"+str(split)+"_evo.fif")
            evo = evo[0]
            evo.pick_types('mag')
            data = torch.tensor(evo.data)
            X_MEG[sub,cond_idx,:,:] = data/torch.norm(data,p='fro')
    torch.save(X_EEG,'data/X_EEG_split'+str(split)+'.pt')
    torch.save(X_MEG,'data/X_MEG_split'+str(split)+'.pt')
    torch.save(evo.times,'data/MEEGtimes.pt')

# %% flipped EEG, l2 normalized
X_EEG = torch.zeros(num_subjects,len(conditions),70,360,dtype=torch.double)
    
for sub in range(num_subjects):
    for cond_idx,cond in enumerate(conditions):
        
        sub_id = f'{(sub+1):02d}'
        
        # EEG
        evo0 = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                +"task-facerecognition_proc-p_cond-"+cond+"_split-0_evo.fif")
        evo0 = evo0[0]
        evo0.pick('eeg')
        data0 = torch.tensor(evo0.data)

        evo1 = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                +"task-facerecognition_proc-p_cond-"+cond+"_split-1_evo.fif")
        evo1 = evo1[0]
        evo1.pick('eeg')
        data1 = torch.tensor(evo1.data)

        for P in range(180):
            data0[:,P] = data0[:,P]/torch.linalg.vector_norm(data0[:,P])
            data1[:,P] = data1[:,P]/torch.linalg.vector_norm(data1[:,P])
        
        X_EEG[sub,cond_idx] = torch.cat((data0,data1),dim=1)
        
U,S,V = torch.pca_lowrank(torch.mean(X_EEG,(0,1))-torch.mean(torch.mean(X_EEG,(0,1))),q=1)
score = (torch.mean(X_EEG,(0,1))-torch.mean(torch.mean(X_EEG,(0,1))))@V

Xf_EEG = torch.zeros(num_subjects,len(conditions),70,360,dtype=torch.double)
for sub in range(num_subjects):
    for cond_idx,cond in enumerate(conditions):
        projections = torch.sum(X_EEG[sub,cond_idx]*score,dim=0)
        to_flip = 2*(projections < 0) -1
        Xf_EEG[sub,cond_idx] = X_EEG[sub,cond_idx]*to_flip
torch.save(Xf_EEG[:,:,:,:180],'data/Xf_EEG_split'+str(0)+'.pt')
torch.save(Xf_EEG[:,:,:,180:],'data/Xf_EEG_split'+str(1)+'.pt')

# %% flipped MEG, l2 normalized
X_MEG = torch.zeros(num_subjects,len(conditions),102,360,dtype=torch.double)
    
for sub in range(num_subjects):
    for cond_idx,cond in enumerate(conditions):
        
        sub_id = f'{(sub+1):02d}'
        
        # EEG
        evo0 = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                +"task-facerecognition_proc-p_cond-"+cond+"_split-0_evo.fif")
        evo0 = evo0[0]
        evo0.pick_types('mag')
        data0 = torch.tensor(evo0.data)

        evo1 = mne.read_evokeds("data/sub-"+sub_id+"/ses-meg/stage-preprocess/"\
                                +"task-facerecognition_proc-p_cond-"+cond+"_split-1_evo.fif")
        evo1 = evo1[0]
        evo1.pick_types('mag')
        data1 = torch.tensor(evo1.data)

        for P in range(180):
            data0[:,P] = data0[:,P]/torch.linalg.vector_norm(data0[:,P])
            data1[:,P] = data1[:,P]/torch.linalg.vector_norm(data1[:,P])
        
        X_MEG[sub,cond_idx] = torch.cat((data0,data1),dim=1)

U,S,V = torch.pca_lowrank(torch.mean(X_MEG,(0,1))-torch.mean(torch.mean(X_MEG,(0,1))),q=1)
score = (torch.mean(X_MEG,(0,1))-torch.mean(torch.mean(X_MEG,(0,1))))@V

Xf_MEG = torch.zeros(num_subjects,len(conditions),102,360,dtype=torch.double)
for sub in range(num_subjects):
    for cond_idx,cond in enumerate(conditions):
        projections = torch.sum(X_MEG[sub,cond_idx]*score,dim=0)
        to_flip = 2*(projections < 0) -1
        Xf_MEG[sub,cond_idx] = X_MEG[sub,cond_idx]*to_flip
torch.save(Xf_MEG[:,:,:,:180],'data/Xf_MEG_split'+str(0)+'.pt')
torch.save(Xf_MEG[:,:,:,180:],'data/Xf_MEG_split'+str(1)+'.pt')
