import sys
import os.path
import torch
import numpy as np
from TMMSAA import larsqp,larsqp_trainer
from load_data import load_data

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
    
    Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

    times = torch.load("data/MEEGtimes.pt")
    dims = {'group_spca':Xtrain['group_spca'].shape,'mm_spca':Xtrain['mm_spca']["EEG"].shape,'mmms_spca':Xtrain['mmms_spca']["EEG"].shape}
    #C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

    l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
    l2_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))

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
            all_test1_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test2_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test12_loss = np.zeros((len(l2_vals),len(l1_vals)))
            for l2,lambda2 in enumerate(l2_vals):
                for l1,lambda1 in enumerate(l1_vals):
                    


                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    loss,best_loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],optimizer=optimizer,max_iter=30000,tol=1e-8)
                    C,S,Bp,Bn = model.get_model_params(X=Xtrain[modeltype])
                    init={'Bp':Bp,'Bn':Bn}

                    all_test1_loss[l2,l1] = model.eval_model(Xtrain=Xtrain1[modeltype],Xtraintilde=Xtrain1[modeltype],Xtest=Xtest1[modeltype])
                    all_test2_loss[l2,l1] = model.eval_model(Xtrain=Xtrain2[modeltype],Xtraintilde=Xtrain2[modeltype],Xtest=Xtest2[modeltype])
                    all_test12_loss[l2,l1] = model.eval_model(Xtrain=Xtrain[modeltype],Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
                    all_train_loss[l2,l1] = best_loss
            np.savetxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_train_loss,delimiter=',')
            np.savetxt("data/SPCA_results/test1_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test1_loss,delimiter=',')
            np.savetxt("data/SPCA_results/test2_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test2_loss,delimiter=',')
            np.savetxt("data/SPCA_results/test12_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test12_loss,delimiter=',')

if __name__=="__main__":
    if len(sys.argv)>1:
        run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    else:
        run_model(M=2,K=5)