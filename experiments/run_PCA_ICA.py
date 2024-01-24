import torch
import pandas as pd
from load_data import load_data
from load_config import load_config
from sklearn.decomposition import FastICA
config = load_config()

X_train,X_test = load_data(data_pool='all',type='group',preproc='split')
X_train1,X_train2,X_test1,X_test2 = load_data(data_pool='half',type='group',preproc='split')

rows_list = []
for inner in range(10):
    for K in range(2,21):

        # Model: group PCA
        U_group_pca,Sigma_group_pca,V_group_pca = torch.pca_lowrank(X_train['all'],q=K)
        recon = U_group_pca@torch.diag(Sigma_group_pca)@V_group_pca.T
        group_pca_train_loss = torch.norm(X_train['all']-recon)**2
        group_pca_val_loss = torch.norm(X_test1['all']-recon[:172*8])**2
        group_pca_test_loss = torch.norm(X_test2['all']-recon[172*8:])**2
        entry = {'modeltype':'PCA','K':K,'inner':inner,'train_loss':group_pca_train_loss.item(),'val_loss':group_pca_val_loss.item(),'test_loss':group_pca_test_loss.item()}
        rows_list.append(entry)

        # # Model: group ICA
        # ica = FastICA(n_components=K)
        # X_transformed = ica.inverse_transform(ica.fit_transform(X_train['all']))
        # group_ica_train_loss = torch.norm(X_train['all']-X_transformed)**2
        # group_ica_test_loss = torch.norm(X_test['all']-X_transformed)**2
        # entry = {'modeltype':'ICA','K':K,'train_loss':group_pca_train_loss.item(),'test_loss':group_pca_test_loss.item()}
        # rows_list.append(entry)
        
        print('Done with '+str(K))
df = pd.DataFrame(rows_list)
df.to_csv('data/PCA_ICA_results.csv',index=False)