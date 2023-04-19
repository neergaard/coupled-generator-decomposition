import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
#from ica import ica1
#from TMMSAA.TMMSAA import SSE
torch.set_num_threads(16)
def run_model(M,K):
    if M==0:
        modeltype='group_spca'
        num_modalities=1
    elif M==1:
        modeltype='mm_spca'
        num_modalities=2
    elif M==2:
        modeltype='mmms_spca'
        num_modalities=2
    
    modality_names = ["EEG", "MEG"]

    Xtrain_mmmsmc = {} # split by modality, subject, condition
    Xtrain_mmms = {} # split by modality, subject
    Xtrain_mm = {} # split by modality
    for m in modality_names:
        Xtrain_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
        Xtrain_mmms[m] = torch.cat((Xtrain_mmmsmc[m][:, 0], Xtrain_mmmsmc[m][:, 1],Xtrain_mmmsmc[m][:, 2],),dim=-1)
        Xtrain_mm[m] = torch.reshape(Xtrain_mmms[m],(16*Xtrain_mmms[m].shape[-2],540))
    Xtrain_group_spca = torch.cat((Xtrain_mm['EEG'],Xtrain_mm['MEG']),dim=-2)
    Xtrain = {'group_spca':Xtrain_group_spca,'mm_spca':Xtrain_mm,'mmms_spca':Xtrain_mmms}

    Xtest_mmmsmc = {} # split by modality, subject, condition
    Xtest_mmms = {} # split by modality, subject
    Xtest_mm = {} # split by modality
    for m in modality_names:
        Xtest_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
        Xtest_mmms[m] = torch.cat((Xtest_mmmsmc[m][:, 0], Xtest_mmmsmc[m][:, 1],Xtest_mmmsmc[m][:, 2],),dim=-1)
        Xtest_mm[m] = torch.reshape(Xtest_mmms[m],(16*Xtest_mmms[m].shape[-2],540))
    Xtest_group_spca = torch.cat((Xtest_mm['EEG'],Xtest_mm['MEG']),dim=-2)
    Xtest = {'group_spca':Xtest_group_spca,'mm_spca':Xtest_mm,'mmms_spca':Xtest_mmms}

    times = torch.load("data/MEEGtimes.pt")
    dims = {'group_spca':Xtrain_group_spca.shape,'mm_spca':Xtrain_mm["EEG"].shape,'mmms_spca':Xtrain_mmms["EEG"].shape}
    #C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

    l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
    l2_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
    #l2_vals = l2_vals[2:]

    num_iter_outer = 5
    num_iter_inner = 20

    # Model: group PCA
    U_group_pca,Sigma_group_pca,V_group_pca = torch.pca_lowrank(Xtrain['group_spca'],q=K,niter=100)
    init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}
    #group_pca_train_loss = torch.norm(Xtrain['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2
    #group_pca_test_loss = torch.norm(Xtest['group_spca']-U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T)**2

    # Model: group ICA
    #A_group_ica,S_group_ica,_ = ica1(x_raw=Xtrain_group_spca,ncomp=K)
    #group_ica_train_loss = torch.norm(Xtrain['group_spca']-A_group_ica@S_group_ica)**2
    #group_ica_test_loss = torch.norm(Xtest['group_spca']-A_group_ica@S_group_ica)**2

    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            if os.path.isfile("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt'):
                continue

            all_train_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test_loss = np.zeros((len(l2_vals),len(l1_vals)))
            for l2,lambda2 in enumerate(l2_vals):
                for l1,lambda1 in enumerate(l1_vals):
                    if l1==0 or l1==1:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init0)
                    else:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init)
                    
                    optimizer = torch.optim.SGD(model.parameters(), lr=1)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1, steps_per_epoch=10, epochs=1000,div_factor=10000,pct_start=0.3)
                    loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],optimizer=optimizer,scheduler=scheduler,max_iter=10000,tol=1e-3)
                    C,S,Bp,Bn = model.get_model_params(X=Xtrain[modeltype])
                    init={'Bp':Bp,'Bn':Bn}

                    all_test_loss[l2,l1] = model.eval_model(Xtrain=Xtrain[modeltype],Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
                    all_train_loss[l2,l1] = loss[-1]
            np.savetxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_train_loss,delimiter=',')
            np.savetxt("data/SPCA_results/test_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test_loss,delimiter=',')

if __name__=="__main__":
    if len(sys.argv)>1:
        run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    else:
        run_model(M=2,K=5)