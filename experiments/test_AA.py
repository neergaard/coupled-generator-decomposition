import torch
import numpy as np
from CGD import CGD, CGD_trainer
import matplotlib.pyplot as plt

# construct uniform 2D data (so, 4 ground truth archetypes)
X = np.random.uniform(low=0, high=1, size=(2,1000))
data = {'X':torch.tensor(X)}

# initialize an AA model with 4 components, where the data is provided in a dictionary
K = 4
model = CGD.CGD(X=data,num_comp=K,model='AA')

# specify an optimizer (don't set LR too high)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train the model, the model stops training when the difference between lowest and second-lowest of the 5 latest losses is less than 1e-8
loss,_ = CGD_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=10000,tol=1e-4)

# get and detach parameters
C,S = model.get_model_params()
C = C.detach().numpy()
S = S.detach().numpy()

# plots
fig,axs = plt.subplots(1,5,figsize=(20,5),layout='constrained')
axs[0].scatter(X[0,:],X[1,:])
axs[0].set_title('Synthetic data')
axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')
for k in range(K):
    axs[1].plot(C[:,k]+k)
axs[1].set_title('C (transposed)')
axs[1].set_yticks([0,1,2,3])
axs[1].set_yticklabels(['1','2','3','4'])
axs[1].set_ylabel('Component')
axs[1].set_xlabel('Sample index')

axs[2].scatter(X[0,:],X[1,:])
#plot the archetypes on top of real data
XC = X@C
axs[2].scatter(XC[0,:],XC[1,:],c='r')
axs[2].set_title('Archetypes (in red)')
axs[2].set_xlabel('Feature 1')
axs[2].set_ylabel('Feature 2')

for k in range(K):
    axs[3].plot(S[0,k,:]+k)
axs[3].set_title('S')
axs[3].set_yticks([0,1,2,3])
axs[1].set_yticklabels(['1','2','3','4'])
axs[3].set_ylabel('Component')
axs[1].set_xlabel('Sample index')

axs[4].plot(loss)
axs[4].set_title('Train loss')
axs[4].set_xlabel('Iteration')
axs[4].set_ylabel('SSE')
fig.savefig('AA_example.png')