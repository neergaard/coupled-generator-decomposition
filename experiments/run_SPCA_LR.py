import torch
import numpy as np
import pandas as pd
from CGD import CGD, CGD_trainer
from load_data import load_data
from load_config import load_config
torch.set_num_threads(16)

def main(modeltype):

    # load parameters from params.json file
    config = load_config()

    K=5
    # common initialization
    C_idx = None
    X_train,_ = load_data(data_pool='all',type='group',preproc='FT_frob')
    print('Calculating group PCA initialization...')
    # _,_,V_group_pca = torch.pca_lowrank(X_train['all'][...,C_idx],q=K,niter=100)
    _,_,V_group_pca = torch.pca_lowrank(X_train['all'],q=K,niter=100)
    print('Calculating group PCA initialization...')
    _,_,V_group_pca = torch.pca_lowrank(X_train['all'],q=K,niter=100)
    init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}

    l1_vals = torch.hstack((torch.tensor(0),torch.logspace(config['lowest_l1_log'],config['highest_l1_log'],config['highest_l1_log']-config['lowest_l1_log']+1)))
    l2_vals = torch.hstack((torch.tensor(0),torch.logspace(config['lowest_l2_log'],config['highest_l2_log'],config['highest_l2_log']-config['lowest_l2_log']+1)))
    lrs = torch.logspace(-5,-1,5) #-6 is too small

    try:
        df = pd.read_csv('data/SPCA_LR_results_K='+str(K)+'_'+modeltype+'.csv')
    except:
        df = pd.DataFrame(columns=['modeltype','K','lambda1','lambda2','lr','inner','iter','train_loss','test_loss'])

    # loop over group, multimodal, and multimodal+multisubject
    X_train,X_test = load_data(data_pool='all',type=modeltype,preproc='FT_frob') #this one on all data...
    # X_train1,X_train2,X_test1_X_test2 = load_data(data_pool='half',type=modeltype,preproc='FT_frob')

    for inner in range(config['num_iter_LR_selection']):
        if len(df[df['inner']==inner])>0:
            continue
        for lr in lrs:
            dftmp = df[df['inner']==inner]
            if len(dftmp[dftmp['lr']==lr])>0:
                continue
            rows_list = []
            for lambda2 in l2_vals:
                for l1,lambda1 in enumerate(l1_vals):
                    print('Beginning modeltype=',modeltype,'K=',K,'lambda1=',lambda1,'lambda2=',lambda2,'lr=',lr,'inner=',inner)
                    if l1==0:
                        model = CGD.CGD(X=X_train,num_comp=K,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init0,G_idx=C_idx)
                    else:
                        model = CGD.CGD(X=X_train,num_comp=K,model='SPCA',lambda1=lambda1,lambda2=lambda2,init=init,G_idx=C_idx)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    loss,_ = CGD_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=config['max_iterations'],tol=config['tolerance'],disable_output=True)
                    C,S,Bp,Bn = model.get_model_params()
                    init={'Bp':Bp,'Bn':Bn}

                    test_loss = model.eval_model(Xtrain=X_train,Xtraintilde=X_train,Xtest=X_test)
                    entry = {'modeltype':modeltype,'lr':lr.item(),'K':K,'lambda2':lambda2.item(),'lambda1':lambda1.item(),'inner':inner,'iter':len(loss),'train_loss':np.min(np.array(loss)),'test_loss':test_loss}
                    rows_list.append(entry)
            df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
            df.to_csv('data/SPCA_LR_results_K='+str(K)+'_'+modeltype+'.csv',index=False)

if __name__ == '__main__':
    modeltype = ['group','mm','mmms']
    main(modeltype[2])
    