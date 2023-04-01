import sys
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
torch.set_num_threads(8)

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
    l2_vals = torch.logspace(-3,2,11)

    num_iter_outer = 5
    num_iter_inner = 100

    _,_,V = torch.pca_lowrank(X['group_spca'],q=K)
    init0 = {'Bp':torch.nn.functional.softplus(torch.nn.functional.relu(V)),'Bn':torch.nn.functional.softplus(torch.nn.functional.relu(-V))}

    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            all_loss = np.zeros((len(l2_vals),len(l1_vals)))
            for l2,lambda2 in enumerate(l2_vals):
                for l1,lambda1 in enumerate(l1_vals):
                    if l1==0:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init0)
                    else:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                    loss = TMMSAA_trainer.Optimizationloop(model=model,X=X[modeltype],Optimizer=optimizer,max_iter=10000,tol=1e-3)
                    C,S,Bp,Bn = model.get_model_params(X=X[modeltype])
                    init={'Bp':Bp,'Bn':Bn}
                    all_loss[l2,l1] = loss[-1]
            np.savetxt("data/SPCA_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner),all_loss,delimiter=',')

if __name__=="__main__":
    #run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    run_model(M=0,K=2)