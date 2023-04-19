import numpy as np
import torch
import quadprog

def larsqp(X,Xtilde,Bn,Bp):
    
    U,Sigma,Vt = np.linalg.svd(np.transpose(X, -2, -1) @ X@(Bp-Bn),full_matrices=False)
    S = torch.transpose(U@Vt,-2,-1)

    XtX = torch.transpose(X,dim=(-2,-1))@X
    XtX_lambda2 = 2*torch.sum(XtX,dim=0) + self.lambda2*torch.eye(X.shape[-1])
    SXtX = 2*S@XtX
    XtX_minus = -2*torch.sum(XtX,dim=0)

    P = torch.cat(XtX_lambda2,XtX_minus,dim=-1)
    P = torch.cat(P,torch.cat(XtX_minus,XtX_lambda2,dim=-1),dim=-2)

    Q = torch.cat(SXtX,-SXtX,dim=-1) #vektor, T lang
    G = -torch.eye(2*X.shape[-1])
    h = torch.zeros(2*X.shape[-1])
    for k in Bn.shape[1]:
        sol=quadprog.cvxopt_solve_qp(P,q=Q[k,:].T,G=G,h=h)
        Bp[k] = sol[:X.shape[-1]/2] #første halvdel
        Bn[k] = sol[X.shape[-1]/2:] #anden halvdel
    return Bp, Bn