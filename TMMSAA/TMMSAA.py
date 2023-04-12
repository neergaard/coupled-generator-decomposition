# Torch file specifying trimodal, multisubject AA
import torch


class TMMSAA(torch.nn.Module):
    """
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
    """

    def __init__(self, dimensions, num_modalities=1, num_comp=3, model="AA", C_idx=None,lambda1=None,lambda2=None,init=None,Xsqnorm=None):
        super().__init__()

        self.model = model
        self.Xsqnorm = Xsqnorm

        if C_idx is None:
            C_idx = torch.ones(dimensions[-1],dtype=torch.bool)
        self.C_idx = C_idx
        # Get ready for some ugly code
        self.S_size = tuple(torch.hstack(
            (torch.tensor(num_modalities,dtype=torch.int16),
                torch.tensor(dimensions[:-2],dtype=torch.int16),
                torch.tensor(num_comp,dtype=torch.int16),
                torch.tensor(dimensions[-1],dtype=torch.int16))
        ).tolist())
        
        if self.model=='AA' or self.model=='DAA':
            self.softmaxC = torch.nn.Softmax(dim=0)
            self.softmaxS = torch.nn.Softmax(dim=-2)

            # Allow for the shared generator matrix to only learn from part of the data (dimension P),
            # such as the post-stimulus period in evoked-responses, while S covers the whole signal

            if init is None:
                # we don't need softmax upon initialization (?)
                self.C = torch.nn.Parameter(
                    self.softmaxC(-torch.log(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double)))
                )

                # squeeze if num_modalities is 1 and if no multiple subjects or conditions are presented
                self.S = torch.nn.Parameter(
                    self.softmaxS(torch.squeeze(-torch.log(torch.rand(self.S_size, dtype=torch.double))))
                )
            else:
                self.C = init['C']
                self.S = init['S']
        elif self.model=='SPCA':
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.softplus = torch.nn.Softplus()
            if init is None:
                self.Bp = torch.nn.Parameter(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double))
                self.Bn = torch.nn.Parameter(torch.rand((torch.sum(C_idx).int(), num_comp), dtype=torch.double))
            else:
                self.Bp = torch.nn.Parameter(init['Bp'].clone())
                self.Bn = torch.nn.Parameter(init['Bn'].clone())

    def get_model_params(self,X=None):
        with torch.no_grad():
            if self.model == "AA":
                return self.softmaxC(self.C), self.softmaxS(self.S)
            elif self.model == "DAA":
                return self.softmaxC(self.C), self.softmaxS(self.S)
            elif self.model == "SPCA":
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                C = Bpsoft - Bnsoft 
                S = torch.squeeze(torch.zeros(self.S_size,dtype=torch.double))
                if type(X) is dict:
                    for m,key in enumerate(X):
                        U,_,Vt = torch.linalg.svd(torch.swapaxes(X[key], -2, -1) @ X[key]@C,full_matrices=False)
                        S[m] = torch.swapaxes(U@Vt,-2,-1)
                else:
                    U,_,Vt = torch.linalg.svd(torch.swapaxes(X, -2, -1) @ X@C,full_matrices=False)
                    S = torch.swapaxes(U@Vt,-2,-1)

                return C, S,self.Bp.detach(),self.Bn.detach()

    def eval_model(self, Xtrain,Xtraintilde,Xtest):
        with torch.no_grad():
            if self.model == 'AA' or self.model == 'DAA':
                S_soft = self.softmaxS(self.S)
                C_soft = self.softmaxC(self.C)
            elif self.model=='SPCA':
                Bpsoft = self.softplus(self.Bp)
                Bnsoft = self.softplus(self.Bn)
                C = Bpsoft - Bnsoft 
                U,_,Vt = torch.linalg.svd(torch.transpose(Xtrain, -2, -1) @ Xtraintilde@C,full_matrices=False)
                S = torch.transpose(U@Vt,-2,-1)
            
            # loop through modalities
            if type(Xtest) is dict:
                loss = 0
                for m,key in enumerate(Xtest):
                    if self.model == "AA":
                        loss += self.SSE(Xtest[key], Xtraintilde[key],C_soft,S_soft)
                    elif self.model == "DAA":
                        loss += self.forwardDAA(Xtest[key], Xtraintilde[key],C_soft,S_soft)
                    elif self.model == 'SPCA':
                        loss += self.SSE(Xtest[key], Xtraintilde[key],C,S)
            else:
                if self.model == "AA":
                    loss = self.SSE(Xtest, Xtraintilde,C_soft,S_soft)
                elif self.model == "DAA":
                    loss = self.forwardDAA(Xtest, Xtraintilde,C_soft,S_soft)
                elif self.model == "SPCA":
                    loss = self.SSE(Xtest, Xtraintilde,C,S)
        return loss.item()
    
    def SSE(self,X,Xtilde,C,S,Xsqnorm=None):
        if Xsqnorm is None:
            Xsqnorm = torch.sum(torch.linalg.matrix_norm(X,ord='fro')**2)
        XC = Xtilde[..., self.C_idx] @ C
        XCtXC = torch.swapaxes(XC, -2, -1) @ XC
        XtXC = torch.swapaxes(X, -2, -1) @ XC

        SSE = (
            Xsqnorm
            - 2 * torch.sum(torch.transpose(XtXC, -2, -1) * S)
            + torch.sum(XCtXC @ S * S)
        )
        return SSE

    def forwardAA(self, X, Xtilde,C_soft,S_soft,Xsqnorm):
        return self.SSE(X,Xtilde,C_soft,S_soft,Xsqnorm)

    def forwardDAA(self, X, Xtilde,C_soft,S_soft):
        
        XC = Xtilde[..., self.C_idx] @ C_soft
        XCtXC = torch.swapaxes(XC, -2, -1) @ XC
        XtXC = torch.swapaxes(X, -2, -1) @ XC

        q = torch.sum(XCtXC @ S_soft * S_soft, dim=-2)
        z = torch.sum(torch.swapaxes(XtXC, -2, -1) * S_soft, dim=-2)
        v = (1 / torch.sqrt(q)) * z

        loss = -torch.sum(v**2)
        if torch.isnan(loss):
            KeyboardInterrupt
        return loss
    
    def forwardSPCA(self,X,Xtilde,Bpsoft,Bnsoft,Xsqnorm):
        # in Zou, Hastie, Tibshirani, B is here C and A is here S
        C = Bpsoft - Bnsoft #the minus is important here
        U,Sigma,Vt = torch.linalg.svd(torch.transpose(X, -2, -1) @ Xtilde@C,full_matrices=False)
        S = torch.transpose(U@Vt,-2,-1)

        return self.SSE(X,Xtilde,C,S,Xsqnorm)

    def forward(self, X, Xtilde):
        if self.Xsqnorm is None:
            if type(X) is dict:
                self.Xsqnorm = {}
                for key in X:
                    self.Xsqnorm[key] = torch.sum(torch.linalg.matrix_norm(X[key],ord='fro')**2)
            else:
                self.Xsqnorm = torch.sum(torch.linalg.matrix_norm(X,ord='fro')**2)

        if self.model == 'AA' or self.model == 'DAA':
            S_soft = self.softmaxS(self.S)
            C_soft = self.softmaxC(self.C)
        elif self.model=='SPCA':
            Bpsoft = self.softplus(self.Bp)
            Bnsoft = self.softplus(self.Bn)
        
        # loop through modalities
        if type(X) is dict:
            loss = 0
            for m,key in enumerate(X):
                if self.model == "AA":
                    loss += self.forwardAA(X[key], Xtilde[key],C_soft,S_soft,self.Xsqnorm)
                elif self.model == "DAA":
                    loss += self.forwardDAA(X[key], Xtilde[key],C_soft,S_soft)
                elif self.model == 'SPCA':
                    loss += self.forwardSPCA(X[key], Xtilde[key],Bpsoft,Bnsoft,self.Xsqnorm)
        else:
            if self.model == "AA":
                loss = self.forwardAA(X, Xtilde,C_soft,S_soft,self.Xsqnorm)
            elif self.model == "DAA":
                loss = self.forwardDAA(X, Xtilde,C_soft,S_soft)
            elif self.model == "SPCA":
                loss = self.forwardSPCA(X, Xtilde,Bpsoft,Bnsoft,self.Xsqnorm)
        if self.model=='SPCA':
            loss+=self.lambda1*torch.sum((Bpsoft+Bnsoft))
            loss+=self.lambda2*torch.sum((Bpsoft**2+Bnsoft**2))

        return loss