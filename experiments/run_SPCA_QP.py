import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer,larsqp,larsqp_trainer
from load_data import load_data

torch.set_num_threads(168)

def run_model(M,K):
    modeltypes = ['group_spca','mm_spca','mmms_spca']
    modeltype = modeltypes[M]
    
    Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()

    l1_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,2,8))))
    l2_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,2,8))))
    # l2_vals = l2_vals[4:]
    
    num_iter_outer = 5
    num_iter_inner = 10

    for outer in range(num_iter_outer):
        for inner in range(num_iter_inner):
            all_train_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test1_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test2_loss = np.zeros((len(l2_vals),len(l1_vals)))
            all_test12_loss = np.zeros((len(l2_vals),len(l1_vals)))
            for l2,lambda2 in enumerate(l2_vals):
                Bp = None # random initialization, then annealing
                Bn = None
                loss_curves = []
                loss_curve_lengths = []
                for l1,lambda1 in enumerate(l1_vals):
                    loss_lars,Bp,Bn,S = larsqp_trainer.Optimizationloop(X=Xtrain[modeltype],num_comp=K,lambda1=lambda1,lambda2=lambda2,max_iter=10000, tol=1e-6,Bp_init=Bp,Bn_init=Bp)
                    all_train_loss[l2,l1] = loss_lars[-1]
                    all_test1_loss[l2,l1] = larsqp_trainer.larsqp_eval(X=Xtrain1[modeltype],Xtest=Xtest1[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
                    all_test2_loss[l2,l1] = larsqp_trainer.larsqp_eval(X=Xtrain2[modeltype],Xtest=Xtest2[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
                    all_test12_loss[l2,l1] = larsqp_trainer.larsqp_eval(X=Xtrain[modeltype],Xtest=Xtest[modeltype],Bp=Bp,Bn=Bn,num_comp=K)
                    loss_curves.extend(loss_lars)
                    loss_curve_lengths.append(len(loss_lars))
                np.savetxt("data/SPCA_QP_results/train_loss_curve"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',loss_curves,delimiter=',')
                np.savetxt("data/SPCA_QP,results/train_loss_curve_len"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',loss_curve_lengths,delimiter=',')
            np.savetxt("data/SPCA_QP_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_train_loss,delimiter=',')
            np.savetxt("data/SPCA_QP_results/test1_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test1_loss,delimiter=',')
            np.savetxt("data/SPCA_QP_results/test2_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test2_loss,delimiter=',')
            np.savetxt("data/SPCA_QP_results/test12_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',all_test12_loss,delimiter=',')

if __name__=="__main__":
    if len(sys.argv)>1:
        run_model(M=int(sys.argv[1]),K=int(sys.argv[2]))
    else:
        run_model(M=2,K=5)