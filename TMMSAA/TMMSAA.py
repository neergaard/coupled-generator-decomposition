# Torch file specifying trimodal, multisubject AA
import torch
import time

class TMMSAA(torch.nn.Module):
    '''
    Coupled generator decomposition model for variable number of tensor dimensions.
    
    The input data X should be either a torch tensor or a dictionary of torch tensors.
    Each tensor should be of size (*,N,P), where:
    * - indicates an arbitrary number of dimensions, e.g., subjects or conditions. 
    N - the dimension that may vary across modalities if X is a dictionary of tensors, 
    e.g., the (variable) number sensors in EEG/MEG for a temporal decomposition or the 
    number of time points in EEG/fMRI for a spatial decomposition.
    P - the dimension that is fixed and assumed equal across modalities. If a temporal 
    decomposition, this would be the number of samples. If a spatial decomposition, this
    would be the number of voxels or surface points. 

    The model learns a shared generator matrix C (P,K) and a mixing matrix S (*,K,P) with
    different properties depending on the model:

    SpPCA: Sparse Principal Component Analysis

    CCD: Convec Cone Decomposition
    S is assumed non-negative

    AA: Archetypal Analysis.
    C and S are non-negative and sum-to-one across the first dimension. S is then assumed
    to be on the simplex and the archetypes learned constitute extremes or corners in the data

    The number of components K is a required input to the mode
    '''


    def __init__(self,dimensions, num_modalities=1,num_comp=3,model='AA'):
        super().__init__()
        
        self.M = num_modalities
        self.tensordims = dimensions
        self.K = num_comp
        self.model = model

        self.softmaxC = torch.nn.Softmax(dim=0)
        self.softmaxS = torch.nn.Softmax(dim=-2)

        self.C = torch.nn.Parameter(self.softmaxC(-torch.log(torch.rand((self.tensordims[-1],self.K),dtype=torch.double))))
        S_size = torch.hstack((torch.tensor(self.M),torch.tensor(self.tensordims[:-2]),torch.tensor(self.K),torch.tensor(self.tensordims[-1])))
        self.S = torch.nn.Parameter(self.softmaxS(-torch.log(torch.rand(tuple(S_size.tolist()),dtype=torch.double))))
        

    def forwardAA(self,X):
        # X is a list of tensors of size NxPxBxL where N is num_samples
        # this function broadcasts matrix mult for all subjects/conditions
        if type(X) is dict:
            XCS = {}
            for m,key in enumerate(X):
                XCS[key] = X[key]@self.softmaxC(self.C)@self.softmaxS(self.S[m])
        else:
            XCS = X@self.softmaxC(self.C)@self.softmaxS(self.S)
        
            
        # check proper implementation
        #XCS2 = {}
        #for m,key in enumerate(X):
        #    XCS2[key] = torch.zeros(X[key].shape,dtype=torch.double)
        #    for l in range(X[key].shape[1]):
        #        for b in range(X[key].shape[0]):
        #            XCS2[key][b,l] = X[key][b,l]@self.softmaxC(self.C)@self.softmaxS(self.S[m,b,l])
        return XCS
    def forward(self,X):
        if self.model == 'AA':
            return self.forwardAA(X)
        
