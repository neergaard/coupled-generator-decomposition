import sys
import torch
import numpy as np
import pandas as pd
from CGD import CGD, CGD_trainer
from load_data import load_data
from load_config import load_config
torch.set_num_threads(8)

def run_model(modeltype,K):
    
    if modeltype=='AA':
        model1='AA'
        datatype = 'mmms'
    else:
        model1='SPCA'
        datatype = modeltype

    # load parameters from params.json file
    config = load_config()

    times = torch.load('data/MEEGtimes.pt')
    C_idx = torch.tensor(np.tile(times>=0.0,3))
    # C_idx = None

    # common initialization

    if model1 == 'SPCA':
        l1_vals = torch.hstack((torch.tensor(0),torch.logspace(config['lowest_l1_log'],-2,-2-config['lowest_l1_log']+1)))
        lambda2 = 0.1
    else:
        lambda1 = torch.tensor(0)
        lambda2 = 0
    # loop over group, multimodal, and multimodal+multisubject
    X_train_group,_ = load_data(data_pool='all',type='group',preproc='split')
    X_train,_ = load_data(data_pool='all',type=datatype,preproc='split')
    X_train1,X_train2,X_test1,X_test2 = load_data(data_pool='half',type=datatype,preproc='split')

    try:
        df = pd.read_csv('data/SPCA_selectedregu_results/SPCA_results_selectedregu'+'K='+str(K)+'.csv')
    except:
        df = pd.DataFrame(columns=['modeltype','K','lambda1','lambda2','inner','iter','train_loss','val_loss','test_loss'])

    for inner in range(config['num_iter_LR_selection']): 
        # if inner already done, skip
        df1 = df[df['modeltype']==modeltype]
        if len(df1[df1['inner']==inner])>0:
            continue
        rows_list = []
        if model1 == 'SPCA':
            print('Calculating group PCA initialization...')
            _,_,V_group_pca = torch.pca_lowrank(X_train_group['all'][...,C_idx],q=K)
            init0 = {'Bp':torch.nn.functional.relu(V_group_pca),'Bn':torch.nn.functional.relu(-V_group_pca)}
            for l1,lambda1 in enumerate(l1_vals):
                print('Beginning modeltype=',modeltype,'K=',K,'lambda1=',lambda1,'lambda2=',lambda2,'inner=',inner)
                if l1==0:
                    model = CGD.CGD(X=X_train,num_comp=K,model=model1,lambda1=lambda1,lambda2=lambda2,init=init0,G_idx=C_idx)
                else:
                    model = CGD.CGD(X=X_train,num_comp=K,model=model1,lambda1=lambda1,lambda2=lambda2,init=init,G_idx=C_idx)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
                loss,_ = CGD_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=config['max_iterations'],tol=config['tolerance'],disable_output=False)
                C,S,Bp,Bn = model.get_model_params()
                init={'Bp':Bp,'Bn':Bn}
        else:
            init0 = None
            model = CGD.CGD(X=X_train,num_comp=K,model=model1,init=init0,G_idx=C_idx)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            loss,_ = CGD_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=config['max_iterations'],tol=config['tolerance'],disable_output=False)
            C,S = model.get_model_params()

        val_loss = model.eval_model(Xtrain=X_train1,Xtraintilde=None,G_idx=C_idx,Xtest=X_test1,AAsubjects=torch.tensor([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],dtype=torch.bool))
        test_loss = model.eval_model(Xtrain=X_train2,Xtraintilde=None,G_idx=C_idx,Xtest=X_test2,AAsubjects=torch.tensor([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],dtype=torch.bool))
    
        
        entry = {'modeltype':modeltype,'K':K,'lambda2':lambda2,'lambda1':lambda1.item(),'inner':inner,
                 'iter':len(loss),'train_loss':np.min(np.array(loss)),'val_loss':val_loss,'test_loss':test_loss}
        rows_list.append(entry)
        df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
        np.save('data/SPCA_selectedregu_results/SPCA_results_selectedregu_C_'+modeltype+str(inner)+'K='+str(K)+'.npy',C)
        np.save('data/SPCA_selectedregu_results/SPCA_results_selectedregu_S_'+modeltype+str(inner)+'K='+str(K)+'.npy',S)
        df.to_csv('data/SPCA_selectedregu_results/SPCA_results_selectedregu'+'K='+str(K)+'.csv',index=False)
        

if __name__=="__main__":
    modeltype = ['group','mm','mmms','AA']
    # modeltype = ['AA']

    for model in modeltype:
        run_model(modeltype=model,K=6)