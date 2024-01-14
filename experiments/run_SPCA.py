import sys
import torch
import numpy as np
import pandas as pd
import TMMSAA
import TMMSAA_trainer
from load_data import load_data
from load_config import load_config
torch.set_num_threads(8)

def run_model(modeltype,K):
    
    # load parameters from params.json file
    config = load_config()

    # times = torch.load('data/MEEGtimes.pt')
    # C_idx = torch.tensor(np.tile(times>=0.0,3))
    C_idx = None

    # common initialization
    X_train,_ = load_data(data_pool='all',type='group',preproc='FT_frob')
    print('Calculating group PCA initialization...')
    # _,_,V_group_pca = torch.pca_lowrank(X_train['all'][...,C_idx],q=K,niter=100)
    _,_,V_group_pca = torch.pca_lowrank(X_train['all'],q=K,niter=100)
    init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}

    l1_vals = torch.hstack((torch.tensor(0),torch.logspace(config['lowest_l1_log'],config['highest_l1_log'],config['highest_l1_log']-config['lowest_l1_log']+1)))
    l2_vals = torch.hstack((torch.tensor(0),torch.logspace(config['lowest_l2_log'],config['highest_l2_log'],config['highest_l2_log']-config['lowest_l2_log']+1)))

    # if df exists already, load, otherwise make new
    try:
        df = pd.read_csv('data/SPCA_results/SPCA_results_K='+str(K)+'_'+modeltype+'.csv')
    except:
        df = pd.DataFrame(columns=['modeltype','K','lambda1','lambda2','inner','iter','train_loss','test_loss'])

    # loop over group, multimodal, and multimodal+multisubject
    X_train,X_test = load_data(data_pool='all',type=modeltype,preproc='FT_frob')

    for inner in range(config['num_iter_LR_selection']):  
        # if inner already done, skip
        if len(df[df['inner']==inner])>0:
            continue
        rows_list = []
        for lambda2 in l2_vals:
            for l1,lambda1 in enumerate(l1_vals):
                print('Beginning modeltype=',modeltype,'K=',K,'lambda1=',lambda1,'lambda2=',lambda2,'inner=',inner)
                if l1==0:
                    model = TMMSAA.TMMSAA(X=X_train,num_comp=K,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init0,C_idx=C_idx)
                else:
                    model = TMMSAA.TMMSAA(X=X_train,num_comp=K,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init,C_idx=C_idx)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
                loss,_ = TMMSAA_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=config['max_iterations'],tol=config['tolerance'],disable_output=True)
                C,S,Bp,Bn = model.get_model_params()
                init={'Bp':Bp,'Bn':Bn}

                test_loss = model.eval_model(Xtrain=X_train,Xtraintilde=None,C_idx=C_idx,Xtest=X_test)
                entry = {'modeltype':modeltype,'K':K,'lambda2':lambda2.item(),'lambda1':lambda1.item(),'inner':inner,'iter':len(loss),'train_loss':np.min(np.array(loss)),'test_loss':test_loss}
                rows_list.append(entry)
        df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
        df.to_csv('data/SPCA_results/SPCA_results_K='+str(K)+'_'+modeltype+'.csv',index=False)

if __name__=="__main__":
    modeltype = ['group','mm','mmms']

    if len(sys.argv)>1:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        run_model(modeltype=modeltype[M],K=K)
    else:
        run_model(modeltype='group',K=5)