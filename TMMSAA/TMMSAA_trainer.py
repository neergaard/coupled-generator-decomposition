# optimization loop
from tqdm import tqdm
import torch


def Optimizationloop(model, X, Optimizer, Xtilde=None, max_iter=100, tol=1e-10):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device).train()

    all_loss = []
    if Xtilde is None:
        if type(X) is dict:
            Xtilde = X.copy()
        else:
            Xtilde = X.clone()
    def closure():
        Optimizer.zero_grad()
        loss = model(X,Xtilde)
        loss.backward()
        return loss
    # X = X.to(device)
    for epoch in tqdm(range(max_iter)):
        loss = model(X, Xtilde)
        all_loss.append(loss.item())
        if epoch > 5:
            if all_loss[-5] - loss < tol:
                break
        #Optimizer.zero_grad()
        #loss.backward()
        Optimizer.step(closure)
    print("Tolerance reached at " + str(epoch) + " number of iterations")
    return all_loss
