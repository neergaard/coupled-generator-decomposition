import torch
import matplotlib.pyplot as plt

modeltypes=['group_spca','mm_spca','mmms_spca']
num_modalities=[1,2,2]
    
modality_names = ["EEG", "MEG"]

X_train_mmmsmc = {} # split by modality, subject, condition
X_train_mmms = {} # split by modality, subject
X_train_mm = {} # split by modality
for m in modality_names:
    # X_train_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_split0.pt")
    X_train_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
    X_train_mmms[m] = torch.cat((X_train_mmmsmc[m][:, 0], X_train_mmmsmc[m][:, 1],X_train_mmmsmc[m][:, 2],),dim=-1)
    X_train_mm[m] = torch.reshape(X_train_mmms[m],(16*X_train_mmms[m].shape[-2],540))
X_train_group_spca = torch.cat((X_train_mm['EEG'],X_train_mm['MEG']),dim=-2)
X_train = {'group_spca':X_train_group_spca,'mm_spca':X_train_mm,'mmms_spca':X_train_mmms}

X_test_mmmsmc = {} # split by modality, subject, condition
X_test_mmms = {} # split by modality, subject
X_test_mm = {} # split by modality
for m in modality_names:
    # X_test_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_split1.pt")
    X_test_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
    X_test_mmms[m] = torch.cat((X_test_mmmsmc[m][:, 0], X_test_mmmsmc[m][:, 1],X_test_mmmsmc[m][:, 2],),dim=-1)
    X_test_mm[m] = torch.reshape(X_test_mmms[m],(16*X_test_mmms[m].shape[-2],540))
X_test_group_spca = torch.cat((X_test_mm['EEG'],X_test_mm['MEG']),dim=-2)
X_test = {'group_spca':X_test_group_spca,'mm_spca':X_test_mm,'mmms_spca':X_test_mmms}

for s in range(16):
    fig,axs = plt.subplots(2,2)
    axs[0,0].plot(X_train_mmms['MEG'][s].T)
    axs[1,0].plot(X_test_mmms['MEG'][s].T)
    axs[0,1].plot(X_train_mmms['EEG'][s].T)
    axs[1,1].plot(X_test_mmms['EEG'][s].T)
    plt.savefig('figures_tmp/sub'+str(s+1)+'.png')
h=7