# optimization loop
from tqdm import tqdm
import torch
from larsqp import larsqp

def Optimizationloop(model, X,num_comp,max_iter=100, tol=1e-10):
    
    all_loss = []
    Bp = torch.rand((X.shape[-1], num_comp))
    Bn = torch.rand((X.shape[-1], num_comp))

    for epoch in tqdm(range(max_iter)):
        Bp,Bn = larsqp(X,Bp,Bn)
        loss = model(X, )
        all_loss.append(loss.item())

        if epoch>100:
            if all_loss[-5] - loss < tol:
                break

    print("Tolerance reached at " + str(epoch) + " number of iterations")
    best_loss = torch.min(all_loss)
    return all_loss, best_loss
