import torch
import numpy as np
import matplotlib.pyplot as plt

modeltypes = ['group_spca','mm_spca','mmms_spca']
# modeltypes = ['group_spca','mmms_spca']
modality_names = ["EEG", "MEG"]

l1_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,1,19))))
l2_vals = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,1,7))))
l1_vals_plot = np.array(torch.hstack((torch.tensor(0),torch.logspace(-5,0,16))))
l2_vals_plot = np.array(torch.hstack((torch.tensor(0),torch.logspace(-4,0,5))))

num_iter_outer = 5
num_iter_inner = 15

num_comps = np.arange(2,31)

linestyles = ['-','--']

K=5
fig, axs = plt.subplots(2,1, figsize=(5, 6),layout='constrained')
fig2, axs2 = plt.subplots(2,1, figsize=(5, 6),layout='constrained')
for inf_type in range(2):
    
    if inf_type==0:
        folder = 'SPCA_results'
    elif inf_type==1:
        folder = 'SPCA_QP_results'

    for m,modeltype in enumerate(modeltypes):
        order_loss = np.zeros((len(l2_vals),len(l1_vals)))
        order_l1l2 = np.zeros(2)
        
        loss = np.zeros((num_iter_outer,len(l2_vals),len(l1_vals)))
        for outer in range(num_iter_outer):
            allinner = np.zeros((num_iter_inner,len(l2_vals),len(l1_vals)))
            for inner in range(num_iter_inner):
                try:
                    allinner[inner]=np.genfromtxt("data/"+folder+"/test12_loss_"+modeltype+"_K="+str(K)+"_rep_"+str(outer)+"_"+str(inner)+'.txt',delimiter=',')
                except:
                    allinner[inner] = np.nan
            loss[outer] = np.nanmin(allinner,axis=0)
        
        order_loss = np.nanmean(loss,axis=0)
        order_loss = order_loss[np.isin(l2_vals,l2_vals_plot)]
        order_loss = order_loss[:,np.isin(l1_vals,l1_vals_plot)]
        best_loss = np.nanmin(order_loss)
        try:
            order_l1l2 = np.where(order_loss==best_loss)
        except:
            order_l1l2 = np.nan

        # axs[inf_type,m].imshow(order_loss[:-1,:-1],vmin=35.83,vmax=35.85)
        
        if inf_type==0:
            if m==0:
                axs[0].set_ylabel('QP SSE')
                for l2,_ in enumerate(l2_vals_plot):
                    axs[0].semilogx(np.array(l1_vals_plot),order_loss[l2])
                axs[0].set_ylim((35.8,35.95))
                axs[0].legend(np.array(l2_vals_plot),title=r'$\lambda_2$ regu.',loc='lower right')
                axs[0].set_title('Group sparse PCA, K=5')
                axs[0].set_xlabel(r'$\lambda_1$ regularization')
                axs[0].set_ylabel('Test SSE')
                axs[0].set_yticks([35.80,35.85,35.90,35.95])
                minval = np.min(order_loss)
                minloc = np.where(order_loss==minval)[1]
                axs[0].scatter(l1_vals_plot[minloc],minval,s=80,facecolors='none',edgecolors='red')
            elif m==2:
                for l2,_ in enumerate(l2_vals_plot):
                    axs[1].semilogx(np.array(l1_vals_plot),order_loss[l2])
                axs[1].set_ylim((29.7,32))
                axs[1].set_title('Multimodal, multisubject sparse PCA, K=5')
                axs[1].set_xlabel(r'$\lambda_1$ regularization')
                axs[1].set_ylabel('Test SSE')
                minval = np.min(order_loss)
                minloc = np.where(order_loss==minval)[1]
                axs[1].scatter(l1_vals_plot[minloc],minval,s=80,facecolors='none',edgecolors='red')
        elif inf_type==1:
            if m==0:
                axs2[0].set_ylabel('QP SSE')
                for l2,_ in enumerate(l2_vals_plot):
                    axs2[0].semilogx(np.array(l1_vals_plot),order_loss[l2])
                axs2[0].set_ylim((35.8,35.95))
                axs2[0].legend(np.array(l2_vals_plot),title=r'$\lambda_2$ regu.',loc='lower right')
                axs2[0].set_title('Group sparse PCA, K=5')
                axs2[0].set_xlabel(r'$\lambda_1$ regularization')
                axs2[0].set_ylabel('Test SSE')
                axs2[0].set_yticks([35.80,35.85,35.90,35.95])
                minval = np.min(order_loss)
                minloc = np.where(order_loss==minval)[1]
                axs2[0].scatter(l1_vals_plot[minloc],minval,s=80,facecolors='none',edgecolors='red')
            elif m==2:
                for l2,_ in enumerate(l2_vals_plot):
                    axs2[1].semilogx(np.array(l1_vals_plot),order_loss[l2])
                axs2[1].set_ylim((29.7,32))
                axs2[1].set_title('Multimodal, multisubject sparse PCA, K=5')
                axs2[1].set_xlabel(r'$\lambda_1$ regularization')
                axs2[1].set_ylabel('Test SSE')
                minval = np.min(order_loss)
                minloc = np.where(order_loss==minval)[1]
                axs2[1].scatter(l1_vals_plot[minloc],minval,s=80,facecolors='none',edgecolors='red')
            
        

fig.savefig('reports/regularization_fig_torch.png',dpi=600,bbox_inches='tight')
fig2.savefig('reports/regularization_fig_QP.png',dpi=600,bbox_inches='tight')
h = 7
