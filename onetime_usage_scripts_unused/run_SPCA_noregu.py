import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from load_data import load_data
torch.set_num_threads(4)
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

    dims = {'group_spca':Xtrain['group_spca'].shape,'mm_spca':Xtrain['mm_spca']["EEG"].shape,'mmms_spca':Xtrain['mmms_spca']["EEG"].shape}

    num_iter_outer = 5
    num_iter_inner = 50

    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            #if os.path.isfile("data/SPCA_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt'):
            #    continue

            losses = np.zeros(4)
            model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities,model='SPCA',lambda1=0,lambda2=0,init=None)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-4,threshold_mode='abs',min_lr=0.0001,patience=100)
            loss,best_loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],optimizer=optimizer,scheduler=scheduler,max_iter=30000,tol=1e-4)

            losses[0] = best_loss
            losses[1] = model.eval_model(Xtrain=Xtrain1[modeltype],Xtraintilde=Xtrain1[modeltype],Xtest=Xtest1[modeltype])
            losses[2] = model.eval_model(Xtrain=Xtrain2[modeltype],Xtraintilde=Xtrain2[modeltype],Xtest=Xtest2[modeltype])
            losses[3] = model.eval_model(Xtrain=Xtrain[modeltype],Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
            
            np.savetxt("data/SPCA_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',losses,delimiter=',')

if __name__=="__main__":
    if len(sys.argv)>1:
        run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    else:
        run_model(M=2,K=5)