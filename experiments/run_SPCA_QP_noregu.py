import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer,larsqp_trainer
from load_data import load_data
torch.set_num_threads(168)
def run_model(M,K):
    if M==0:
        modeltype='group_spca'
    elif M==1:
        modeltype='mm_spca'
    elif M==2:
        modeltype='mmms_spca'
    
    Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

    num_iter_outer = 5
    num_iter_inner = 50

    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            if os.path.isfile("data/SPCA_QP_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt'):
                continue

            losses = np.zeros(4)
            loss_lars,Bp,Bn,S = larsqp_trainer.Optimizationloop(X=Xtrain[modeltype],num_comp=K,lambda1=0,lambda2=0,max_iter=10000, tol=1e-6,Bp_init=None,Bn_init=None)
            losses[0] = loss_lars[-1]
            losses[1] = larsqp_trainer.larsqp_eval(X=Xtrain1[modeltype],Xtest=Xtest1[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
            losses[2] = larsqp_trainer.larsqp_eval(X=Xtrain2[modeltype],Xtest=Xtest2[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
            losses[3] = larsqp_trainer.larsqp_eval(X=Xtrain[modeltype],Xtest=Xtest[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
            
            np.savetxt("data/SPCA_QP_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',losses,delimiter=',')

if __name__=="__main__":
    if len(sys.argv)>1:
        run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    else:
        run_model(M=2,K=5)