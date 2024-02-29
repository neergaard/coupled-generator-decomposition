import sys
import torch
import numpy as np
import pandas as pd
from CGD import CGD, CGD_trainer
from load_data import load_data
from load_config import load_config
torch.set_num_threads(8)

def run_model(modeltype,K):
    
    # load parameters from params.json file
    config = load_config()

    # times = torch.load('data/MEEGtimes.pt')
    # C_idx = torch.tensor(np.tile(times>=0.0,3))
    C_idx = None

    # if df exists already, load, otherwise make new
    try:
        df = pd.read_csv('data/AA_results/AA_results_K='+str(K)+'_'+modeltype+'.csv')
    except:
        df = pd.DataFrame(columns=['modeltype','K','inner','iter','train_loss','val_loss','test_loss'])

    # loop over group, multimodal, and multimodal+multisubject
    X_train,_ = load_data(data_pool='all',type=modeltype,preproc='split')
    X_train1,X_train2,X_test1,X_test2 = load_data(data_pool='half',type=modeltype,preproc='split')

    for inner in range(config['num_iter_LR_selection']):  
        # if inner already done, skip
        if len(df[df['inner']==inner])>0:
            continue
        rows_list = []
        print('Beginning modeltype=',modeltype,'K=',K,'inner=',inner)
        model = CGD.CGD(X=X_train,num_comp=K,model='AA',init=None,G_idx=C_idx)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss,_ = CGD_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=config['max_iterations'],tol=config['tolerance'],disable_output=False)
        C,S = model.get_model_params()
    
        val_loss = model.eval_model(Xtrain=X_train1,Xtraintilde=None,G_idx=C_idx,Xtest=X_test1)
        test_loss = model.eval_model(Xtrain=X_train2,Xtraintilde=None,G_idx=C_idx,Xtest=X_test2)
        entry = {'modeltype':modeltype,'K':K,'inner':inner,'iter':len(loss),'train_loss':np.min(np.array(loss)),'val_loss':val_loss,'test_loss':test_loss}
        rows_list.append(entry)
        df = pd.concat([df,pd.DataFrame(rows_list)],ignore_index=True)
    df.to_csv('data/AA_results/AA_results_K='+str(K)+'_'+modeltype+'.csv',index=False)

if __name__=="__main__":
    modeltype = ['group','mm','mmms']

    if len(sys.argv)>1:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        run_model(modeltype=modeltype[M],K=K)
    else:
        run_model(modeltype='mmms',K=5)