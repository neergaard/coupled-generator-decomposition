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

    def __init__(self, dimensions, num_modalities=1, num_comp=3, model="AA", C_idx=None):
        super().__init__()

        self.model = model
        self.C_idx = C_idx

        self.softmaxC = torch.nn.Softmax(dim=0)
        self.softmaxS = torch.nn.Softmax(dim=-2)

        # Allow for the shared generator matrix to only learn from part of the data (dimension P),
        # such as the post-stimulus period in evoked-responses, while S covers the whole signal
        if C_idx is None:
            self.C = torch.nn.Parameter(
                self.softmaxC(-torch.log(torch.rand((dimensions[-1], num_comp), dtype=torch.double)))
            )
        else:
            self.C = torch.nn.Parameter(
                self.softmaxC(-torch.log(torch.rand((torch.sum(C_idx), num_comp), dtype=torch.double)))
            )

        # Get ready for some ugly code
        S_size = torch.hstack(
            (torch.tensor(num_modalities),
                torch.tensor(dimensions[:-2]),
                torch.tensor(num_comp),
                torch.tensor(dimensions[-1]))
        )
        self.S = torch.nn.Parameter(
            self.softmaxS(-torch.log(torch.rand(tuple(S_size.tolist()), dtype=torch.double)))
        )

    def get_model_params(self):
        with torch.no_grad():
            if self.model == "AA":
                return self.softmaxC(self.C), self.softmaxS(self.S)
            if self.model == "DAA":
                return self.softmaxC(self.C), self.softmaxS(self.S)

    def eval_model(self, X, Xtilde):
        with torch.no_grad():
            loss = self.forward(X,Xtilde)
        return loss

    def forwardAA(self, X, Xtilde,S_soft,C_soft):
        # X is a list of tensors of size NxPxBxL where N is num_samples
        # this function broadcasts matrix mult for all subjects/conditions

        XC = Xtilde[..., self.C_idx] @ C_soft
        XCtXC = torch.swapaxes(XC, -2, -1) @ XC
        XtXC = torch.swapaxes(X, -2, -1) @ XC

        SSE = (
            torch.sum(torch.linalg.matrix_norm(X))
            - 2 * torch.sum(torch.swapaxes(XtXC, -2, -1) * S_soft)
            + torch.sum(XCtXC @ S_soft * S_soft)
        )

        return SSE

    def forwardDAA(self, X, Xtilde,S_soft,C_soft):
        
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

    def forward(self, X, Xtilde):
        S_soft = self.softmaxS(self.S)
        C_soft = self.softmaxC(self.C)
        
        if type(X) is dict:
            loss = 0
            for key in X:
                if self.model == "AA":
                    loss += self.forwardAA(X[key], Xtilde[key],S_soft,C_soft)
                elif self.model == "DAA":
                    loss += self.forwardDAA(X[key], Xtilde[key],S_soft,C_soft)
        else:
            if self.model == "AA":
                loss = self.forwardAA(X[key], Xtilde[key],S_soft,C_soft)
            elif self.model == "DAA":
                loss = self.forwardDAA(X[key], Xtilde[key],S_soft,C_soft)
        return loss
