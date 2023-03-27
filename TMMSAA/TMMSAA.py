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
    variable number of time points in EEG/fMRI for a spatial decomposition.
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


    def __init__(self,dimensions, num_modalities=1,num_comp=3,model='AA',C_idx=None):
        super().__init__()
        
        self.model = model
        self.C_idx = C_idx

        self.softmaxC = torch.nn.Softmax(dim=0)
        self.softmaxS = torch.nn.Softmax(dim=-2)

        # Allow for the shared generator matrix to only learn from part of the data (dimension P), 
        # such as the post-stimulus period in evoked-responses, while S covers the whole signal
        if C_idx is None:
            self.C = torch.nn.Parameter(self.softmaxC(-torch.log(torch.rand((dimensions[-1],num_comp),dtype=torch.double))))
        else:
            self.C = torch.nn.Parameter(self.softmaxC(-torch.log(torch.rand((torch.sum(C_idx),num_comp),dtype=torch.double))))
        
        # Get ready for some ugly code
        S_size = torch.hstack((torch.tensor(num_modalities),torch.tensor(dimensions[:-2]),torch.tensor(num_comp),torch.tensor(dimensions[-1])))
        self.S = torch.nn.Parameter(self.softmaxS(-torch.log(torch.rand(tuple(S_size.tolist()),dtype=torch.double))))
        
    def get_model_params(self):
        with torch.no_grad():
            if self.model=='AA':
                return self.softmaxC(self.C),self.softmaxS(self.S)
            if self.model=='DAA':
                return self.softmaxC(self.C),self.softmaxS(self.S)
    
    def eval_model(self,X,Xtilde):
        with torch.no_grad():
            if self.model=='AA':
                if type(X) is dict:
                    loss=0
                    for m,key in enumerate(X):
                        loss += torch.norm(X[key]-Xtilde[key]@self.softmaxC(self.C)@self.softmaxS(self.S[m]),p='fro')**2
                else:
                    loss = torch.norm(X-Xtilde@self.softmaxC(self.C)@self.softmaxS(self.S),p='fro')
        return loss

    def forwardAA(self,X,Xtilde):
        # X is a list of tensors of size NxPxBxL where N is num_samples
        # this function broadcasts matrix mult for all subjects/conditions
        
        #SSE = 0
        #if type(X) is dict:
        #    XCS = {}
        #    for m,key in enumerate(X):
        #        XCS[key] = Xtilde[key]@self.softmaxC(self.C)@self.softmaxS(self.S[m])
        #        SSE += torch.norm(X[key]-XCS[key])**2
        #else:
        #    XCS = Xtilde@self.softmaxC(self.C)@self.softmaxS(self.S)
        #    SSE = torch.norm(X-XCS)**2
            
        # efficient implementation
        S_soft = self.softmaxS(self.S)
        if type(X) is dict:
            SSE = 0
            for m,key in enumerate(X):
                XC = Xtilde[key]@self.softmaxC(self.C)
                XCtXC = torch.swapaxes(XC,-2,-1)@XC
                XtXC = torch.swapaxes(X[key],-2,-1)@XC

                SSE+=torch.sum(torch.linalg.matrix_norm(X[key]))-2*torch.sum(torch.swapaxes(XtXC,-2,-1)*S_soft[m])+torch.sum(XCtXC@S_soft[m]*S_soft[m])
        else:
            XC = Xtilde@self.softmaxC(self.C)
            XCtXC = torch.swapaxes(XC,-2,-1)@XC
            XtXC = torch.swapaxes(X[key],-2,-1)@XC

            SSE=torch.sum(torch.linalg.matrix_norm(X))-2*torch.sum(torch.swapaxes(XtXC,-2,-1)*S_soft)+torch.sum(XCtXC@S_soft*S_soft)
        
        
        # check proper implementation
        #loss_loop = 0
        #for m,key in enumerate(X):
        #    for l in range(X[key].shape[1]):
        #        for b in range(X[key].shape[0]):
        #            loss_loop += torch.norm(X[key][b,l]-X[key][b,l]@self.softmaxC(self.C)@self.softmaxS(self.S[m,b,l]),p='fro')**2
        return SSE
    
    def forwardDAA(self,X,Xtilde):
        S_soft = self.softmaxS(self.S)

        #WatsonError = 0
        #for m,key in enumerate(X):
        #    for b in range(X[key].shape[0]):
        #        for l in range(X[key].shape[1]):
        #            XC = Xtilde[key][b,l]@self.softmaxC(self.C)
        #            XCtXC = XC.T@XC
        #            XtXC = X[key][b,l].T@XC

        #            q = torch.sum(XCtXC@S_soft[m,b,l]*S_soft[m,b,l],dim=0)
        #            z = torch.sum(XtXC.T*S_soft[m,b,l],dim=0)
        #            v = (1/torch.sqrt(q))*z

        #            WatsonError+=-torch.sum(v**2)
        #            if torch.sum(torch.isnan(v))>0:
        #                y=8
        #return WatsonError

        if type(X) is dict:
            WatsonError = 0
            for m,key in enumerate(X):
                XC = Xtilde[key]@self.softmaxC(self.C)
                XCtXC = torch.swapaxes(XC,-2,-1)@XC
                XtXC = torch.swapaxes(X[key],-2,-1)@XC

                q = torch.sum(XCtXC@S_soft[m]*S_soft[m],dim=-2)
                z = torch.sum(torch.swapaxes(XtXC,-2,-1)*S_soft[m],dim=-2)
                v = (1/torch.sqrt(q))*z

                WatsonError+=-torch.sum(v**2)
                if torch.isnan(WatsonError):
                    y=8
        else:
            XC = Xtilde@self.softmaxC(self.C)
            XCtXC = torch.swapaxes(XC,-2,-1)@XC
            XtXC = torch.swapaxes(X[key],-2,-1)@XC

            q = torch.sum(XCtXC@S_soft*S_soft,dim=0)
            z = torch.sum(torch.swapaxes(XtXC,-2,-1)*S_soft,dim=0)
            v = (1/torch.sqrt(q))*z

            WatsonError=-torch.sum(v**2)
        return WatsonError
    
    def forward(self,X,Xtilde):
        
        if self.model == 'AA':
            return self.forwardAA(X,Xtilde)
        elif self.model == 'DAA':
            return self.forwardDAA(X,Xtilde)
        
