import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca']
modality_names = ["EEG", "MEG"]

l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
l2_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))

num_iter_outer = 5
num_iter_inner = 10

num_comps = np.arange(2,31)

plt.figure()
linestyles = ['-','--']
for inf_type in range(2):
    
    if inf_type==0:
        folder = 'SPCA_results'
    elif inf_type==1:
        folder = 'SPCA_QP_results'
        l1_vals = l1_vals[1:]
        l2_vals = l2_vals[1:]
    for m,modeltype in enumerate(modeltypes):
        order_loss = np.zeros((len(num_comps),len(l1_vals),len(l2_vals)))
        best_loss = np.zeros((len(num_comps)))
        order_l1l2 = np.zeros((len(num_comps),2))
        for k,K in enumerate(num_comps):
            loss = np.zeros((num_iter_outer,len(l1_vals),len(l2_vals)))
            for outer in range(num_iter_outer):
                allinner = np.zeros((num_iter_inner,len(l1_vals),len(l2_vals)))
                for inner in range(num_iter_inner):
                    try:
                        allinner[inner]=np.genfromtxt("data/"+folder+"/test1_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                    except:
                        allinner[inner] = np.nan
                loss[outer] = np.nanmin(allinner,axis=0)
            
            order_loss[k] = np.nanmean(loss,axis=0)
            best_loss[k] = np.nanmin(order_loss[k])
            try:
                order_l1l2[k] = np.where(order_loss[k]==best_loss[k])
            except:
                order_l1l2[k] = np.nan

            
        plt.plot(num_comps,best_loss,linestyle=linestyles[inf_type])
plt.legend(modeltypes)

h = 7
