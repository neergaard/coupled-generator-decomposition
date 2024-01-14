# Torch file specifying trimodal, multisubject AA
import torch
from time import time

class TMMSAA(torch.nn.Module):
    """
    Coupled generator decomposition model for variable number of tensor dimensions.

    The input data X should be a dictionary of torch tensors. 
    Each tensor should be of size (*,N,P), where:
    * - indicates an arbitrary number of dimensions, e.g., subjects or conditions.
    N - the dimension that may vary across modalities if X is a dictionary of tensors,
    e.g., the (variable) number sensors in EEG/MEG for a temporal decomposition or the
    variable number of time points in EEG/fMRI for a spatial decomposition.
    P - the dimension that is fixed and assumed equal across modalities. If a temporal
    decomposition, this would be the number of samples. If a spatial decomposition, this
    would be the number of voxels or surface points.

    Each entry in the dictionary may represent a different modality, subject, or condition, 
    where the number of samples differs. THUS, say you have a multimodal, multisubject setup where 
    N differs between modalities but not subjects, input a dictionary "data", with entries, 
    e.g., data['mod1'] and data['mod2'], where data['mod1'].shape = (*,N1,P) and data['mod2'].shape = (*,N2,P).

    The model learns a shared generator matrix C (P,K) and a mixing matrix S (*,K,P) with
    different properties depending on the model:

    SpPCA: Sparse Principal Component Analysis

    CCD: Convec Cone Decomposition
    S is assumed non-negative

    AA: Archetypal Analysis.
    C and S are non-negative and sum-to-one across the first dimension. S is then assumed
    to be on the simplex and the archetypes learned constitute extremes or corners in the data

    The number of components K is a required input to the model. 
    """

    def __init__(self,  num_comp, X, Xtilde=None, model="SPCA", C_idx=None,lambda1=None,lambda2=None,init=None):
        super().__init__()
        print('Initializing model: '+model)
        t1 = time()
        self.model = model
        self.keys = X.keys()

        num_modalities = len(X)
        P = X[list(self.keys)[0]].shape[-1]
        other_dims = list(X[list(self.keys)[0]].shape[:-2])
        self.keys = X.keys()

        # Allow for the shared generator matrix to only learn from part of the data (dimension P),
        # such as the post-stimulus period in evoked-responses, while S covers the whole signal
        if C_idx is None:
            C_idx = torch.ones(P,dtype=torch.bool)
        self.C_idx = C_idx
        
        self.X = X
        if Xtilde is None:
            self.Xtilde = {}
            for key in self.keys:
                self.Xtilde[key] = X[key][..., self.C_idx].clone()
        
        if model == "SPCA" or model == "AA":
            self.Xsqnorm = torch.zeros(num_modalities,dtype=torch.double)
            self.XtXtilde = torch.zeros((num_modalities,*other_dims,P,torch.sum(C_idx)),dtype=torch.double)
            for m,key in enumerate(self.keys):
                self.Xsqnorm[m] = torch.sum(torch.linalg.matrix_norm(self.X[key],ord='fro')**2)
                self.XtXtilde[m] = torch.transpose(self.X[key], -2, -1) @ self.Xtilde[key]

        self.S_size = [num_modalities, *other_dims, num_comp, P]
        
        if self.model=='AA' or self.model=='DAA':
            self.softmaxC = torch.nn.Softmax(dim=0)
            self.softmaxS = torch.nn.Softmax(dim=-2)

            if init is None:
                self.C = torch.nn.Parameter(
                    self.softmaxC(-torch.log(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double)))
                )
                # squeeze if num_modalities is 1 and if no multiple subjects or conditions are presented
                self.S = torch.nn.Parameter(
                    self.softmaxS(torch.squeeze(-torch.log(torch.rand(self.S_size, dtype=torch.double))))
                )
            else:
                self.C = init['C'].clone()
                self.S = init['S'].clone()
        elif self.model=='SPCA':
            if lambda1 is None or lambda2 is None:
                raise ValueError('lambda1 and lambda2 must be specified for SPCA')
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.softplus = torch.nn.Softplus()
            if init is None:
                self.Bp = torch.nn.Parameter(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double))
                self.Bn = torch.nn.Parameter(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double))
            else:
                self.Bp = torch.nn.Parameter(init['Bp'].clone())
                self.Bn = torch.nn.Parameter(init['Bn'].clone())
        t2 = time()
        print('Model initialized in '+str(t2-t1)+' seconds')

    def get_model_params(self):
        with torch.no_grad():
            if self.model == "AA" or self.model=="DAA":
                return self.softmaxC(self.C).detach(), self.softmaxS(self.S).detach()
            elif self.model == "SPCA":
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                C = Bpsoft - Bnsoft 
                S = torch.zeros(self.S_size,dtype=torch.double)
                U,_,Vt = torch.linalg.svd(self.XtXtilde @ C,full_matrices=False)
                S = torch.transpose(U@Vt,-2,-1)
                return C, S,self.Bp.detach(),self.Bn.detach()

    def eval_model(self, Xtrain,Xtest,Xtraintilde=None,C_idx=None):
        with torch.no_grad():
            if C_idx is None:
                C_idx = torch.ones(Xtrain[list(self.keys)[0]].shape[-1],dtype=torch.bool)
            if Xtraintilde is None:
                Xtraintilde = {}
                for key in self.keys:
                    Xtraintilde[key] = Xtrain[key][..., C_idx].clone()
            if self.model == 'AA' or self.model == 'DAA':
                S = self.softmaxS(self.S)
                C = self.softmaxC(self.C)
            elif self.model=='SPCA':
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                C = Bpsoft - Bnsoft 
            
            loss = 0
            for key in self.keys:
                if self.model == 'SPCA':
                    U,_,Vt = torch.linalg.svd(torch.transpose(Xtrain[key], -2, -1) @ Xtraintilde[key]@C,full_matrices=False)
                    S = torch.transpose(U@Vt,-2,-1)
                loss += torch.sum(torch.linalg.matrix_norm(Xtest[key]-Xtraintilde[key]@C@S)**2)
        return loss.item()

    def forwardDAA(self, X, Xtilde,C_soft,S_soft):
        loss = 0 
        for key in self.keys:
            XC = Xtilde[key] @ C_soft
            XCtXC = torch.swapaxes(XC, -2, -1) @ XC
            XCtXC = torch.swapaxes(XC, -2, -1) @ XC
            XtXC = torch.swapaxes(X[key], -2, -1) @ XC

            q = torch.sum(XCtXC @ S_soft * S_soft, dim=-2)
            z = torch.sum(torch.swapaxes(XtXC, -2, -1) * S_soft, dim=-2)
            v = (1 / torch.sqrt(q)) * z

            loss += -torch.sum(v**2)
        return loss
    
    def SSE(self,XtXtilde,Xtilde,C,S,Xsqnorm):
        XC = Xtilde @ C
        XtXC = XtXtilde @ C

        SSE = (
            Xsqnorm
            - 2 * torch.sum(torch.transpose(XtXC, -2, -1) * S)
            + torch.sum(XC*XC)
        )
        return SSE
    
    def forwardAA(self,C_soft,S_soft):
        loss = 0
        for key in self.keys:
            loss += self.SSE(self.XtXtilde[key],self.Xtilde[key],C_soft,S_soft,self.Xsqnorm[key])
        return loss
    
    def forwardSPCA(self,C):
        # in Zou, Hastie, Tibshirani, B is here C and A is here S
        loss = 0
        XtXC = self.XtXtilde @ C
        U,_,Vt = torch.linalg.svd(XtXC,full_matrices=False)
        S = torch.transpose(U@Vt,-2,-1)
        for m,key in enumerate(self.keys):
            XC = self.Xtilde[key] @ C
            SSE = self.Xsqnorm[m] - 2 * torch.sum(torch.transpose(XtXC[m], -2, -1) * S[m]) + torch.sum(XC*XC) #correct
            loss += SSE
        return loss

    def forward(self):

        if self.model == 'AA' or self.model == 'DAA':
            S_soft = self.softmaxS(self.S)
            C_soft = self.softmaxC(self.C)
        elif self.model=='SPCA':
            Bpsoft = self.softplus(self.Bp)
            Bnsoft = self.softplus(self.Bn)
            C = Bpsoft - Bnsoft 
        
        # loop through modalities   
        if self.model == "AA":
            loss = self.forwardAA(C_soft,S_soft)
        elif self.model == "DAA":
            loss = self.forwardDAA(self.X, self.Xtilde,C_soft,S_soft)
        elif self.model == 'SPCA':
            loss = self.forwardSPCA(C)
            loss+=self.lambda1*torch.sum((Bpsoft+Bnsoft))
            loss+=self.lambda2*torch.sum((Bpsoft**2+Bnsoft**2))

        if torch.isnan(loss):
            KeyboardInterrupt
        return loss