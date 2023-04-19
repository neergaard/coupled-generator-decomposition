import torch
def load_data():
    modality_names = ["EEG", "MEG"]

    ######################### Xtrain, all 16 subjects
    Xtrain_mmmsmc = {} # split by modality, subject, condition
    Xtrain_mmms = {} # split by modality, subject
    Xtrain_mm = {} # split by modality
    for m in modality_names:
        Xtrain_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
        Xtrain_mmms[m] = torch.cat((Xtrain_mmmsmc[m][:, 0], Xtrain_mmmsmc[m][:, 1],Xtrain_mmmsmc[m][:, 2],),dim=-1)
        Xtrain_mm[m] = torch.reshape(Xtrain_mmms[m],(16*Xtrain_mmms[m].shape[-2],540))
    Xtrain_group_spca = torch.cat((Xtrain_mm['EEG'],Xtrain_mm['MEG']),dim=-2)
    Xtrain = {'group_spca':Xtrain_group_spca,'mm_spca':Xtrain_mm,'mmms_spca':Xtrain_mmms}

    ######################### Xtrain, first 8 subjects
    Xtrain1_mmmsmc = {} # split by modality, subject, condition
    Xtrain1_mmms = {} # split by modality, subject
    Xtrain1_mm = {} # split by modality
    for m in modality_names:
        Xtrain1_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
        Xtrain1_mmms[m] = torch.cat((Xtrain1_mmmsmc[m][:8, 0], Xtrain1_mmmsmc[m][:8, 1],Xtrain1_mmmsmc[m][:8, 2],),dim=-1)
        Xtrain1_mm[m] = torch.reshape(Xtrain1_mmms[m],(8*Xtrain1_mmms[m].shape[-2],540))
    Xtrain1_group_spca = torch.cat((Xtrain1_mm['EEG'],Xtrain1_mm['MEG']),dim=-2)
    Xtrain1 = {'group_spca':Xtrain1_group_spca,'mm_spca':Xtrain1_mm,'mmms_spca':Xtrain1_mmms}

    ######################### Xtrain, second 8 subjects
    Xtrain2_mmmsmc = {} # split by modality, subject, condition
    Xtrain2_mmms = {} # split by modality, subject
    Xtrain2_mm = {} # split by modality
    for m in modality_names:
        Xtrain2_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
        Xtrain2_mmms[m] = torch.cat((Xtrain2_mmmsmc[m][8:, 0], Xtrain2_mmmsmc[m][8:, 1],Xtrain2_mmmsmc[m][8:, 2],),dim=-1)
        Xtrain2_mm[m] = torch.reshape(Xtrain2_mmms[m],(8*Xtrain2_mmms[m].shape[-2],540))
    Xtrain2_group_spca = torch.cat((Xtrain2_mm['EEG'],Xtrain2_mm['MEG']),dim=-2)
    Xtrain2 = {'group_spca':Xtrain2_group_spca,'mm_spca':Xtrain2_mm,'mmms_spca':Xtrain2_mmms}

    Xtest_mmmsmc = {} # split by modality, subject, condition
    Xtest_mmms = {} # split by modality, subject
    Xtest_mm = {} # split by modality
    for m in modality_names:
        Xtest_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
        Xtest_mmms[m] = torch.cat((Xtest_mmmsmc[m][:, 0], Xtest_mmmsmc[m][:, 1],Xtest_mmmsmc[m][:, 2],),dim=-1)
        Xtest_mm[m] = torch.reshape(Xtest_mmms[m],(16*Xtest_mmms[m].shape[-2],540))
    Xtest_group_spca = torch.cat((Xtest_mm['EEG'],Xtest_mm['MEG']),dim=-2)
    Xtest = {'group_spca':Xtest_group_spca,'mm_spca':Xtest_mm,'mmms_spca':Xtest_mmms}

  
    Xtest1_mmmsmc = {} # split by modality, subject, condition
    Xtest1_mmms = {} # split by modality, subject
    Xtest1_mm = {} # split by modality
    for m in modality_names:
        Xtest1_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
        Xtest1_mmms[m] = torch.cat((Xtest1_mmmsmc[m][:8, 0], Xtest1_mmmsmc[m][:8, 1],Xtest1_mmmsmc[m][:8, 2],),dim=-1)
        Xtest1_mm[m] = torch.reshape(Xtest1_mmms[m],(8*Xtest1_mmms[m].shape[-2],540))
    Xtest1_group_spca = torch.cat((Xtest1_mm['EEG'],Xtest1_mm['MEG']),dim=-2)
    Xtest1 = {'group_spca':Xtest1_group_spca,'mm_spca':Xtest1_mm,'mmms_spca':Xtest1_mmms}


    Xtest2_mmmsmc = {} # split by modality, subject, condition
    Xtest2_mmms = {} # split by modality, subject
    Xtest2_mm = {} # split by modality
    for m in modality_names:
        Xtest2_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
        Xtest2_mmms[m] = torch.cat((Xtest2_mmmsmc[m][8:, 0], Xtest2_mmmsmc[m][8:, 1],Xtest2_mmmsmc[m][8:, 2],),dim=-1)
        Xtest2_mm[m] = torch.reshape(Xtest2_mmms[m],(8*Xtest2_mmms[m].shape[-2],540))
    Xtest2_group_spca = torch.cat((Xtest2_mm['EEG'],Xtest2_mm['MEG']),dim=-2)
    Xtest2 = {'group_spca':Xtest2_group_spca,'mm_spca':Xtest2_mm,'mmms_spca':Xtest2_mmms}

    return Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2