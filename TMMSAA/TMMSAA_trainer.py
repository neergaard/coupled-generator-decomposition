# optimization loop
import torch
from tqdm import tqdm

def Optimizationloop(model,X,Optimizer,Xtilde=None,max_iter=100,tol=1e-10):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device).train()
    
    all_loss = []

    if Xtilde is None:
        Xtilde = X.copy()
            
    if model.C_idx is not None:
        if type(Xtilde) is dict:
            for key in Xtilde:
                Xtilde[key]=Xtilde[key][...,model.C_idx]
        else:
            Xtilde = Xtilde[...,model.C_idx]

    #X = X.to(device)
    for epoch in tqdm(range(max_iter)):
        loss = model(X,Xtilde)
        #loss = model(X)

        all_loss.append(loss.detach())

        if epoch>5:
            if all_loss[-5]-loss<tol:
                print("Tolerance reached at "+str(epoch)+" number of iterations")
                return  all_loss
            
        Optimizer.zero_grad(set_to_none=True)
        loss.backward()
        Optimizer.step()
        
    return  all_loss