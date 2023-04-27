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

linestyles = ['-','--']

K=5
fig, axs = plt.subplots(2,3, figsize=(12, 4))
for inf_type in range(2):
    
    if inf_type==0:
        folder = 'SPCA_results'
    elif inf_type==1:
        folder = 'SPCA_QP_results'

    for m,modeltype in enumerate(modeltypes):
        order_loss = np.zeros((len(l1_vals),len(l2_vals)))
        order_l1l2 = np.zeros(2)
        
        loss = np.zeros((num_iter_outer,len(l1_vals),len(l2_vals)))
        for outer in range(num_iter_outer):
            allinner = np.zeros((num_iter_inner,len(l1_vals),len(l2_vals)))
            for inner in range(num_iter_inner):
                try:
                    allinner[inner]=np.genfromtxt("data/"+folder+"/test12_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                except:
                    allinner[inner] = np.nan
            loss[outer] = np.nanmin(allinner,axis=0)
        
        order_loss = np.nanmean(loss,axis=0)
        best_loss = np.nanmin(order_loss)
        try:
            order_l1l2 = np.where(order_loss==best_loss)
        except:
            order_l1l2 = np.nan

        # axs[inf_type,m].imshow(order_loss[:-1,:-1],vmin=35.83,vmax=35.85)
        for l2,_ in enumerate(l2_vals):
            axs[inf_type,m].semilogx(np.array(l1_vals),order_loss[l2])
        if m==0:
            axs[inf_type,m].set_ylim((35.7,36.2))
        elif m==1:
            axs[inf_type,m].set_ylim((34.7,36.15))
        elif m==2:
            axs[inf_type,m].set_ylim((29.7,32))
        axs[inf_type,m].legend(np.array(l2_vals))
plt.show()
h = 7
