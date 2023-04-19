import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca','group_pca','group_ica','mmms_daa','group_spca_lars']
modality_names = ["EEG", "MEG"]

num_iter_outer = 5
num_iter_inner = 20

num_comps = np.arange(2,31)

best_train_loss = np.zeros((len(modeltypes),len(num_comps)))
best_test_loss = np.zeros((len(modeltypes),len(num_comps)))

for k,K in enumerate(num_comps):
    for m,modeltype in enumerate(modeltypes):

        if modeltype == 'group_pca':
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_PCA_K='+str(K)+'.txt')
            best_train_loss[m,k] = loss[0]
            best_test_loss[m,k] = loss[1]
        elif modeltype == 'group_ica':
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_ICA_K='+str(K)+'.txt')
            best_train_loss[m,k] = loss[0]
            best_test_loss[m,k] = loss[1]
        elif modeltype == 'mmms_daa':
            continue
            train_loss = np.genfromtxt('data/DAA_results/train_loss_mmms_daa_K='+str(K)+'_SSE.txt')
            test_loss = np.genfromtxt('data/DAA_results/train_loss_mmms_daa_K='+str(K)+'_SSE.txt')
            best_train_loss[m,k] = np.nanmean(np.nanmin(train_loss,axis=1))
            best_test_loss[m,k] = np.nanmean(np.nanmin(test_loss,axis=1))
        else:

            train_loss = np.zeros((num_iter_outer,num_iter_inner))
            test_loss = np.zeros((num_iter_outer,num_iter_inner))
            for outer in range(num_iter_outer):
                for inner in range(num_iter_inner):
                    try:
                        loss=np.genfromtxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                        train_loss[outer,inner] = loss[0,0]
                        loss=np.genfromtxt("data/SPCA_results/test_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                        test_loss[outer,inner] = loss[0,0]
                    except:
                        try:
                            loss=np.genfromtxt("data/SPCA_results/train_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                            train_loss[outer,inner] = loss
                            loss=np.genfromtxt("data/SPCA_results/test_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                            test_loss[outer,inner] = loss
                        except:
                            train_loss[outer,inner] = np.nan
                            test_loss[outer,inner] = np.nan
                try:
                    best_train_inner_idx = np.nanargmin(train_loss,axis=1)
                    best_test_inner = test_loss[best_train_inner_idx]
                except:
                    best_test_inner = np.nanmin(test_loss,axis=1)
            best_train_loss[m,k] = np.nanmean(np.nanmin(train_loss,axis=1))
            best_test_loss[m,k] = np.nanmean(best_test_inner)
        
        
plt.figure()
plt.plot(num_comps,best_train_loss.T)
plt.legend(modeltypes)
plt.ylim(10,60)
plt.title('Train loss')
plt.xlabel('Model order (K)')
plt.figure()
plt.plot(num_comps,best_test_loss.T)
plt.legend(modeltypes)
plt.ylim(10,60)
plt.title('Test loss')
plt.xlabel('Model order (K)')
h = 7
