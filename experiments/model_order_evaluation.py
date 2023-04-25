import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca','group_pca','group_ica','mmms_daa','group_spca','mm_spca','mmms_spca']
modality_names = ["EEG", "MEG"]

num_iter_outer = 5
num_iter_inner = 50

num_comps = np.arange(2,31)

mean_train_loss = np.zeros((len(modeltypes),len(num_comps)))
mean_test_loss = np.zeros((len(modeltypes),len(num_comps)))
std_train_loss = np.zeros((len(modeltypes),len(num_comps)))
std_test_loss = np.zeros((len(modeltypes),len(num_comps)))

fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(12, 6))
for k,K in enumerate(num_comps):
    for m,modeltype in enumerate(modeltypes):

        if m==3:
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_PCA_K='+str(K)+'.txt')
            mean_train_loss[m,k] = loss[0]
            mean_test_loss[m,k] = loss[1]
        elif m==4:
            loss = np.genfromtxt('data/PCAICA_results/train_test_loss_ICA_K='+str(K)+'.txt')
            mean_train_loss[m,k] = loss[0]
            mean_test_loss[m,k] = loss[1]
        elif m==5:
            continue
            train_loss = np.genfromtxt('data/DAA_results/train_loss_mmms_daa_K='+str(K)+'_SSE.txt')
            test_loss = np.genfromtxt('data/DAA_results/train_loss_mmms_daa_K='+str(K)+'_SSE.txt')
            mean_train_loss[m,k] = np.nanmean(np.nanmin(train_loss,axis=1))
            mean_test_loss[m,k] = np.nanmean(np.nanmin(test_loss,axis=1))
        else:

            train_loss = np.zeros((num_iter_outer,num_iter_inner))
            test_loss = np.zeros((num_iter_outer,num_iter_inner))
            for outer in range(num_iter_outer):
                for inner in range(num_iter_inner):
                    try:
                        if m<3:
                            loss=np.genfromtxt("data/SPCA_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                        elif m>5:
                            loss=np.genfromtxt("data/SPCA_QP_noregu_results/loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                        train_loss[outer,inner] = loss[0]
                        test_loss[outer,inner] = loss[1]
                    except:
                        train_loss[outer,inner] = np.nan
                        test_loss[outer,inner] = np.nan
                try:
                    best_train_inner_idx = np.nanargmin(train_loss,axis=1)
                    best_test_inner = test_loss[best_train_inner_idx]
                except:
                    best_test_inner = np.nanmin(test_loss,axis=1)
            mean_train_loss[m,k] = np.nanmean(np.nanmin(train_loss,axis=1))
            mean_test_loss[m,k] = np.nanmean(best_test_inner)
            std_train_loss[m,k] = np.nanstd(np.nanmin(train_loss,axis=1))
            std_test_loss[m,k] = np.nanstd(best_test_inner)
    #ax1.fill_between(num_comps,mean_train_loss[m]-std_train_loss[m],mean_train_loss[m]+std_train_loss[m])
    #ax1.plot(num_comps,mean_train_loss[m])
    #ax2.fill_between(num_comps,mean_test_loss[m]-std_test_loss[m],mean_test_loss[m]+std_test_loss[m])
    #ax2.plot(num_comps,mean_test_loss[m])
        
ax1.plot(num_comps,mean_train_loss.T)
ax2.plot(num_comps,mean_test_loss.T)
ax1.legend(modeltypes)
ax1.set_ylim(0,60)
ax2.set_ylim(10,30)
ax1.set_title('Train loss')
ax2.set_title('Test loss')
ax1.set_xlabel('Model order (K)')
ax2.set_xlabel('Model order (K)')
h = 7
