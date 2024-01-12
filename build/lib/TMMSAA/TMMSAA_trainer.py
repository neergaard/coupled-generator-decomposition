# optimization loop
from tqdm import tqdm
import torch


def Optimizationloop(model, optimizer, scheduler=None, max_iter=100, tol=1e-10,disable_output=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).train()

    all_loss = []
    lrs = []

    for epoch in tqdm(range(max_iter),disable=disable_output):
        loss = model()
        all_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        if epoch>100:
            if scheduler is not None:
                scheduler.step(loss)
                if optimizer.param_groups[0]["lr"]<0.001:
                    break
            else: #specify relative tolerance threshold
                if (all_loss[-5]-all_loss[-1])/all_loss[-5]<tol:
                    break
                

    print("Tolerance reached at " + str(epoch+1) + " number of iterations")
    best_loss = min(all_loss)
    return all_loss, best_loss
