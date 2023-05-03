import torch
import sys
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer, larsqp_trainer
from load_data import load_data

def run_model(M1,M2):
    modeltypes = ['group_spca','mm_spca','mmms_spca']
    num_modalities = [1,2,2]

    Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

    dims = {'group_spca':Xtrain['group_spca'].shape,'mm_spca':Xtrain['mm_spca']["EEG"].shape,'mmms_spca':Xtrain['mmms_spca']["EEG"].shape}
    #C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

    num_iter_outer = 5
    num_iter_inner = 50

    K=5

    if M1==0:
    ##### For group_spca
        modeltype = modeltypes[0]
        M = num_modalities[0]
        l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,1,19)))
        l1_vals = l1_vals[:11]
        lambda2 = torch.tensor(0.1)
    elif M1==1:
        ##### For mmms_spca
        modeltype = modeltypes[2]
        M = num_modalities[2]
        l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,1,19)))
        l1_vals = l1_vals[:12]
        lambda2 = torch.tensor(1)

    # Model: group PCA
    #_,_,V_group_pca = torch.pca_lowrank(Xtrain['group_spca'],q=K,niter=100)
    #init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}

    for outer in range(num_iter_outer):
        losses = np.zeros((num_iter_inner,4))
        for inner in range(num_iter_inner):
            Bp = None # random initialization, then annealing
            Bn = None
            for l1,lambda1 in enumerate(l1_vals):
                if M2==0:
                    inf_type = 'torch'
                    if l1==0:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=M,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=None)
                    else:
                        model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=M,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-4,threshold_mode='abs',min_lr=0.0001,patience=100)
                    loss,best_loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],optimizer=optimizer,scheduler=scheduler,max_iter=30000,tol=1e-4)
                    C,S,Bp,Bn = model.get_model_params(X=Xtrain[modeltype])
                    init={'Bp':Bp,'Bn':Bn}
                elif M2==1:
                    inf_type = 'QP'
                    loss_lars,Bp,Bn,S = larsqp_trainer.Optimizationloop(X=Xtrain[modeltype],num_comp=K,lambda1=np.array(lambda1),lambda2=np.array(lambda2),max_iter=10000, tol=1e-6,Bp_init=Bp,Bn_init=Bn)
                    C = Bp-Bn
            np.savetxt('data/SPCA_results_selectedregu/'+inf_type+'C_'+modeltype+'_'+str(s)+'_rep_'+str(outer)+'_'+str(inner)+'.txt',C,delimiter=',')
            if modeltype=='mmms_spca':
                for m in range(M):
                    for s in range(16):
                        np.savetxt('data/SPCA_results_selectedregu/'+inf_type+'S_'+modeltype+'_'+str(m)+'_'+str(s)+'_rep_'+str(outer)+'_'+str(inner)+'.txt',S[m,s],delimiter=',')
            else:
                np.savetxt('data/SPCA_results_selectedregu/'+inf_type+'S_'+modeltype+'_rep'+str(outer)+'_'+str(inner)+'.txt',S,delimiter=',')
        losses[inner,0]=best_loss
        losses[inner,1]=model.eval_model(Xtrain=Xtrain1[modeltype],Xtraintilde=Xtrain1[modeltype],Xtest=Xtest1[modeltype])
        losses[inner,2]=model.eval_model(Xtrain=Xtrain2[modeltype],Xtraintilde=Xtrain2[modeltype],Xtest=Xtest2[modeltype])
        losses[inner,3]=model.eval_model(Xtrain=Xtrain[modeltype],Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
        np.savetxt('data/SPCA_results_selectedregu/'+inf_type+'loss_'+modeltype+'_K='+str(K)+'_rep_'+str(outer)+'.txt',losses,delimiter=',')
            

if __name__=="__main__":
    run_model(M1=int(sys.argv[1]),M2=int(sys.argv[2]))