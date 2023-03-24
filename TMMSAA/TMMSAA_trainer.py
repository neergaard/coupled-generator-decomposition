# optimization loop
import torch
from tqdm import tqdm

def Optimizationloop(model,X,Optimizer,max_iter=100,tol=1e-10):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device).train()
    
    all_loss = []
    lossCriterion = torch.nn.MSELoss(reduction = "sum")

    #X = X.to(device)
    for epoch in tqdm(range(max_iter)):
        Xrecon = model(X)
        #loss = model(X)

        loss = 0
        for m,key in enumerate(X):
            loss += lossCriterion(Xrecon[key],X[key])
        
        all_loss.append(loss.detach())

        if epoch>1:
            if all_loss[-2]-loss<tol:
                print("Tolerance reached at "+str(epoch)+" number of iterations")
                return  all_loss
            
        Optimizer.zero_grad(set_to_none=True)
        loss.backward()
        Optimizer.step()
        
    return  all_loss