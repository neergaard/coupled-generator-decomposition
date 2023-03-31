# optimization loop
from tqdm import tqdm
import torch


def Optimizationloop(model, X, Optimizer, Xtilde=None, max_iter=100, tol=1e-10):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device).train()

    all_loss = []

    if Xtilde is None:
        Xtilde = X.clone()

    # X = X.to(device)
    for epoch in tqdm(range(max_iter)):
        loss = model(X, Xtilde)
        # loss = model(X)

        all_loss.append(loss.detach())

        if epoch > 5:
            if all_loss[-5] - loss < tol:
                break

        Optimizer.zero_grad(set_to_none=True)
        loss.backward()
        Optimizer.step()
    print("Tolerance reached at " + str(epoch) + " number of iterations")
    return all_loss
