# Torch file specifying trimodal, multisubject AA
import torch
import time

class TMMSAA(torch.nn.Module):
    def __init__(self,num_dimensions, num_comp=3,num_subjects=1,num_conditions=1,num_modalities=1):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)

        self.P = num_dimensions
        self.K = num_comp
        self.B = num_subjects
        self.L = num_conditions
        self.M = num_modalities
        
        # perhaps torch double?
        self.C = torch.nn.Parameter(self.softmax(-torch.log(torch.rand((self.P,self.K),dtype=torch.double))))
        self.S = torch.nn.Parameter(self.softmax(-torch.log(torch.rand((self.K,self.P,self.B,self.L,self.M),dtype=torch.double))))

    def forward(self,X):
        # X is a list of tensors of size NxPxBxL where N is num_samples
        # this function broadcasts matrix mult for all subjects/conditions
        XCS = {}
        for m,key in enumerate(X):
            XCS[key] = torch.permute(torch.permute(X[key],(2,3,0,1))@self.softmax(self.C)@torch.permute(self.softmax(self.S[:,:,:,:,m]),(2,3,0,1)),(2,3,0,1))
            
        #XCS = {}
        #for m,key in enumerate(X):
        #    XCS[key] = torch.zeros(X[key].shape,dtype=torch.double)
        #    for l in range(self.L):
        #        for b in range(self.B):
        #            XCS[key][:,:,b,l] = X[key][:,:,b,l]@self.softmax(self.C)@self.softmax(self.S[:,:,b,l,m])
        
        return XCS
