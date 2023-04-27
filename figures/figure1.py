import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from load_data import load_data
import matplotlib.pyplot as plt

Xtrain,Xtest,Xtrain1,Xtrain2,Xtest1,Xtest2 = load_data()


##################### Figure 1a
plt.figure(figsize=(4,8))

times = torch.load('data/MEEGtimes.pt')
subdelim = 0.015
modalitydelim = 0.005
x_spacing = 10
t0_idx = 20
ms500_idx = 120

for sub in range(16):
    for m,modality in enumerate(Xtrain['mmms_spca']):
        plt.plot(torch.arange(0,180),torch.mean(Xtrain['mmms_spca'][modality][sub,:,:180],dim=0)+modalitydelim*m+subdelim*sub,color='green',alpha=0.5)
        plt.plot(torch.arange(180+x_spacing,370),torch.mean(Xtrain['mmms_spca'][modality][sub,:,180:360],dim=0)+modalitydelim*m+subdelim*sub,color='black',alpha=0.5)
        plt.plot(torch.arange(360+x_spacing*2,560),torch.mean(Xtrain['mmms_spca'][modality][sub,:,360:],dim=0)+modalitydelim*m+subdelim*sub,color='red',alpha=0.5)

plt.vlines([t0_idx,180+x_spacing+t0_idx,360+x_spacing*2+t0_idx],ymin=-modalitydelim/2,ymax=subdelim*15+2*modalitydelim,linestyles='dashed',colors='k',alpha=0.4)

plt.ylabel('')
plt.xticks([t0_idx,120,180+x_spacing+t0_idx,180+ms500_idx+x_spacing,360+x_spacing*2+t0_idx,360+x_spacing*2+ms500_idx],
           labels=['Stim','0.5s','Stim','0.5s','Stim','0.5s'],rotation=45)
plt.yticks(np.arange(0+modalitydelim/2,16*subdelim,subdelim),labels=np.arange(1,17))
plt.savefig('reports/Figure1a.png',bbox_inches='tight',dpi=600)
# plt.box('off')

################### Figure 1b






stop=7