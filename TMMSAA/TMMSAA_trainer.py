# optimization loop
from tqdm import tqdm
import torch


def Optimizationloop(model, X, optimizer, scheduler=None,Xtilde=None, max_iter=100, tol=1e-10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()

    all_loss = []
    lrs = []
    if Xtilde is None:
        if type(X) is dict:
            Xtilde = X.copy()
        else:
            Xtilde = X.clone()
    #def closure():
    #    Optimizer.zero_grad()
    #    loss = model(X,Xtilde)
    #    loss.backward()
    #    return loss
    # X = X.to(device)
    for epoch in tqdm(range(max_iter)):
        loss = model(X, Xtilde)
        all_loss.append(loss.item())
        #if epoch > 100:
        #    if all_loss[-5] - loss < 0:
        #        break
        #if epoch>100:
        #    if all_loss[-100] - loss < tol:
        #        break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        if epoch>100:
            if scheduler is not None:
                scheduler.step(loss)
                if optimizer.param_groups[0]["lr"]<0.01:
                    break

    print("Tolerance reached at " + str(epoch) + " number of iterations")
    best_loss = min(all_loss)
    return all_loss, best_loss
