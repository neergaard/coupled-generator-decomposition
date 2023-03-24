import matplotlib.pyplot as plt
import torch
import numpy as np
from TMMSAA import TMMSAA,TMMSAA_trainer

P = 300
B = 10
L = 3
N = 2
modalitynames = ['EEG','MEG','fMRI']
X = {}
for m in range(3):
    norm1 = np.random.multivariate_normal([5, 0], [[1, 1], [1, 2]], size = (int(P/3),B,L))
    norm2 = np.random.multivariate_normal([17, 0], [[3, -1], [-1, 7]], size = (int(P/3),B,L))
    norm3 = np.random.multivariate_normal([15, 10], [[10, 0.2], [0.2, 5]], size = (int(P/3),B,L))
    X[modalitynames[m]] = torch.tensor(np.transpose(np.concatenate((norm1,norm2,norm3),axis=0),(3,0,1,2)))


M = len(X)
K = 3

model = TMMSAA.TMMSAA(num_dimensions=P, num_comp=K,num_subjects=B,num_conditions=L,num_modalities=M)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
loss = TMMSAA_trainer.Optimizationloop(model=model,X=X,Optimizer=optimizer,max_iter=5000,tol=1e-10)

# plots for one subject
C = torch.nn.functional.softmax(model.state_dict()['C'],dim=0)
S = torch.nn.functional.softmax(model.state_dict()['S'],dim=0)

sub = 0
cond = 0
mod = 'EEG'
mod2 = 0
XC = (X[mod][:,:,sub,cond]@C).T.detach()
plt.figure()
plt.plot(X[mod][0,:,sub,cond].detach(),X[mod][1,:,sub,cond].detach(),'.')
plt.plot(XC[:,0], XC[:,1],'X',alpha=1)
plt.fill(XC[:,0], XC[:,1], facecolor='none', edgecolor='purple', linewidth=1)

plt.figure()
plt.plot(loss)

y=7