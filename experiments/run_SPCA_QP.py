import sys
import torch
import os
import numpy as np
import pandas as pd
import larsqp_trainer
from load_data import load_data
from load_config import load_config
os.environ["OMP_NUM_THREADS"] = "8"

def run_model(modeltype,K,init_method='group_PCA'):
    
    # load parameters from params.json file
    config = load_config()

    # times = torch.load('data/MEEGtimes.pt')
    # C_idx = np.tile(times>=0.0,3)
    C_idx = None

    # common initialization
    X_traingroup,_ = load_data(data_pool='all',type='group',preproc='split')

    l1_vals = np.hstack((np.array(0),np.logspace(config['lowest_l1_log'],config['highest_l1_log'],config['highest_l1_log']-config['lowest_l1_log']+1)))
    l2_vals = np.hstack((np.array(0),np.logspace(config['lowest_l2_log'],config['highest_l2_log'],config['highest_l2_log']-config['lowest_l2_log']+1)))

    # if df exists already, load, otherwise make new
    try:
        df = pd.read_csv('data/SPCA_results/SPCA_QP_results_K='+str(K)+'_'+modeltype+'_'+init_method+'.csv')
    except:
        df = pd.DataFrame(columns=['modeltype','init_method','K','lambda1','lambda2','inner','iter','train_loss','val_loss','test_loss'])

    # loop over group, multimodal, and multimodal+multisubject
    X_train,X_test = load_data(data_pool='all',type=modeltype,preproc='split',as_numpy_array=True)
    X_train1,X_train2,X_test1,X_test2 = load_data(data_pool='half',type=modeltype,preproc='split',as_numpy_array=True)

    for inner in range(config['num_iter_LR_selection']):  
        # if inner already done, skip
        if len(df[df['inner']==inner])>0:
            continue
        if init_method == 'group_PCA':
            print('Calculating group PCA initialization...')
            _,_,V_group_pca = torch.pca_lowrank(X_traingroup['all'],q=K)
            Bp_init0 = np.array(torch.nn.functional.relu(V_group_pca))
            Bn_init0 = np.array(torch.nn.functional.relu(-V_group_pca))
        else:
            Bp_init0 = None
            Bn_init0 = None
        rows_list = []
        for lambda2 in l2_vals:
            for l1,lambda1 in enumerate(l1_vals):
                print('Beginning modeltype=',modeltype,'K=',K,'lambda1=',lambda1,'lambda2=',lambda2,'inner=',inner)
                if l1==0:
                    loss,Bp,Bn,S = larsqp_trainer.Optimizationloop(X=X_train,num_comp=K,lambda1=lambda1,lambda2=lambda2,Bp_init=Bp_init0,Bn_init=Bn_init0,max_iter=config['max_iterations'], tol=config['tolerance'],C_idx=C_idx)
                else:
                    loss,Bp,Bn,S = larsqp_trainer.Optimizationloop(X=X_train,num_comp=K,lambda1=lambda1,lambda2=lambda2,Bp_init=Bp,Bn_init=Bn,max_iter=config['max_iterations'], tol=config['tolerance'],C_idx=C_idx)
                
                # test_loss = larsqp_trainer.larsqp_eval(Xtrain=X_train,Xtraintilde=X_train,Xtest=X_test,Bp=Bp,Bn=Bn,num_comp=K,C_idx=C_idx)
                val_loss = larsqp_trainer.larsqp_eval(Xtrain=X_train1,Xtraintilde=None,C_idx=C_idx,Xtest=X_test1,Bp=Bp,Bn=Bn,num_comp=K)
                test_loss = larsqp_trainer.larsqp_eval(Xtrain=X_train2,Xtraintilde=None,C_idx=C_idx,Xtest=X_test2,Bp=Bp,Bn=Bn,num_comp=K)
                entry = {'modeltype':modeltype,'init_method':init_method,'K':K,'lambda2':lambda2,'lambda1':lambda1,'inner':inner,'iter':len(loss),'train_loss':np.min(np.array(loss)),'val_loss':val_loss,'test_loss':test_loss}
                rows_list.append(entry)
        df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
        df.to_csv('data/SPCA_results/SPCA_QP_results_K='+str(K)+'_'+modeltype+'_'+init_method+'.csv',index=False)

if __name__=="__main__":
    modeltype = ['group','mm','mmms']

    if len(sys.argv)>1:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        init_method = int(sys.argv[3])
        if init_method==0:
            init_method = 'group_PCA'
        elif init_method==1:
            init_method = 'random'
        run_model(modeltype=modeltype[M],K=K,init_method=init_method)
    else:
        run_model(modeltype='mmms',K=2,init_method='group_PCA')