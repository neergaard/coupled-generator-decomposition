# optimization loop
from tqdm import tqdm
import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False
import time
import os
os.environ["OMP_NUM_THREADS"] = "8"

def Optimizationloop(num_comp, X, Xtilde=None,C_idx=None,lambda1=None,lambda2=None,max_iter=100, tol=1e-10,Bp_init=None,Bn_init=None,disable_output=False):
    
    ##### LARS-QP initialization #####
    t1 = time.time()
    num_modalities = len(X)
    keys = X.keys()
    other_dims = list(X[list(keys)[0]].shape[:-2])
    P = X[list(keys)[0]].shape[-1]
    S = np.zeros([num_modalities, *other_dims, num_comp, P])

    if C_idx is None:
        C_idx = np.ones(P,dtype=bool)

    if Xtilde is None:
        Xtilde = {}
        for key in keys:
            Xtilde[key] = X[key][..., C_idx].copy()
    
    XtXtilde = np.zeros((num_modalities,*other_dims,P,np.sum(C_idx)))
    XtXtilde_sum = np.zeros((*other_dims,P,np.sum(C_idx)))
    for m,key in enumerate(keys):
        XtXtilde[m] = np.swapaxes(X[key], -1,-2) @ Xtilde[key]
        XtXtilde_sum += XtXtilde[m]

    if other_dims:
        XtXtilde_sum = np.sum(XtXtilde_sum,axis=tuple(np.arange(len(XtXtilde_sum.shape))[:-2]))
        
    XtX_lambda2 = 2*XtXtilde_sum + lambda2*np.eye(P)
    XtX_minus = -2*XtXtilde_sum
    P1 = np.concatenate((XtX_lambda2,XtX_minus),axis=1)
    P1 = cvxopt.matrix(np.concatenate((P1,np.concatenate((XtX_minus,XtX_lambda2),axis=-1)),axis=-2))
    G = cvxopt.matrix(-np.eye(2*P))
    h = cvxopt.matrix(np.zeros(2*P))

    if Bp_init is None and Bn_init is None:
        Bp = np.random.standard_normal(size=(P, num_comp))
        Bn = np.random.standard_normal(size=(P, num_comp))
    else:
        Bp = Bp_init 
        Bn = Bn_init
        
    t2 = time.time()
    print('Model initialized in '+str(t2-t1)+' seconds')

    ##### LARS-QP optimization #####
    all_loss = []
    for epoch in tqdm(range(max_iter),disable=disable_output):
        initvals = cvxopt.matrix(np.concatenate((Bp,Bn),axis=0))

        U,_,Vt = np.linalg.svd(XtXtilde @ (Bp-Bn),full_matrices=False)
        S = np.swapaxes(U@Vt,-2, -1)
        SXtXtilde = 2*np.sum(S@XtXtilde,axis=tuple(np.arange(len(XtXtilde.shape))[:-2]))

        Q = cvxopt.matrix(-np.concatenate((SXtXtilde,-SXtXtilde),axis=1)+lambda1) #vektor, 2T lang
        for k in range(num_comp):
            sol = np.array(cvxopt.solvers.qp(P1,Q[k,:].T,G,h,initvals=initvals[:,k])['x']).reshape(2*P,)
            #sol = quadprog.solve_qp(P,-Q[k],-G,h)[0]
            Bp[:,k] = sol[:P] #første halvdel
            Bn[:,k] = sol[P:] #anden halvdel
            cvxopt.solvers.options

        loss = 0
        for m,key in enumerate(X):
            loss += np.linalg.norm(X[key]-Xtilde[key]@(Bp-Bn)@S[m])**2
        all_loss.append(loss.item())

        if epoch>5:
            latest = np.array(all_loss[-5:])
            minval = np.min(latest)
            secondlowest = np.min(latest[latest!=minval])
            if (secondlowest-minval)/minval<tol:
                break
            # if (all_loss[-5] - loss)/all_loss[-5] < tol:
            #     break

    print("Tolerance reached at " + str(epoch) + " number of iterations")
    return all_loss,Bp,Bn,S

def larsqp_eval(Xtrain,Xtest,Bp,Bn,num_comp,Xtraintilde=None,C_idx=None):
    keys = Xtrain.keys()

    num_modalities = len(Xtrain)
    other_dims = list(Xtrain[list(keys)[0]].shape[:-2])
    P = Xtrain[list(keys)[0]].shape[-1]
    S = np.zeros([num_modalities, *other_dims, num_comp, P])

    if C_idx is None:
        C_idx = np.ones(P,dtype=bool)

    if Xtraintilde is None:
        Xtraintilde = {}
        for key in keys:
            Xtraintilde[key] = Xtrain[key][..., C_idx].copy()

    loss = 0
    for key in keys:
        XtXtilde = np.swapaxes(Xtrain[key],-2,-1)@Xtraintilde[key]
        U,_,Vt = np.linalg.svd(XtXtilde @ (Bp-Bn),full_matrices=False)
        S = np.swapaxes(U@Vt,-2, -1)
        loss += np.sum(np.linalg.norm(Xtest[key]-np.array(Xtraintilde[key])@(Bp-Bn)@S,axis=(-2,-1))**2)
        
    return loss