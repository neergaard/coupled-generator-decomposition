import matplotlib.pyplot as plt
import torch
import numpy as np
from CGD import TMMSAA, TMMSAA_trainer, visualize_AA_trajectory

modalitynames = ["EEG", "MEG"]
M = len(modalitynames)
# K = 5
model = 1

# Full AA
X_train = {}
X_train_cat = {}
X_train_l2 = {}
X_train_l2_cat = {}
for m in range(M):
    X_train[modalitynames[m]] = torch.load(
        "data/concatenatedData/X_" + modalitynames[m] + "_FT_frob.pt"
    )
    X_train_cat[modalitynames[m]] = torch.cat(
        (
            X_train[modalitynames[m]][:, 0],
            X_train[modalitynames[m]][:, 1],
            X_train[modalitynames[m]][:, 2],
        ),
        dim=-1,
    )

    X_train_l2[modalitynames[m]] = torch.load(
        "data/concatenatedData/Xf_" + modalitynames[m] + "_FT.pt"
    )
    X_train_l2_cat[modalitynames[m]] = torch.cat(
        (
            X_train_l2[modalitynames[m]][:, 0],
            X_train_l2[modalitynames[m]][:, 1],
            X_train_l2[modalitynames[m]][:, 2],
        ),
        dim=-1,
    )

times = torch.load("data/MEEGtimes.pt")
B, L, N, P = X_train["EEG"].shape
C_idx = torch.hstack(
    (torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool))
)

loss_run = []
best_loss = 1000000000

if model == 0:

    for outer in range(1):
        for inner in range(100):
            print(inner)
            model_full = TMMSAA.TMMSAA(
                dimensions=(B, L, N, P),
                num_modalities=M,
                num_comp=5,
                model="DAA",
                C_idx=C_idx,
            )
            optimizer = torch.optim.Adam(model_full.parameters(), lr=0.1)
            loss_full = TMMSAA_trainer.Optimizationloop(
                model=model_full,
                X=X_train,
                Xtilde=X_train_l2,
                Optimizer=optimizer,
                max_iter=10000,
                tol=1e-16,
            )
            loss_run.append(loss_full[-1])
            if loss_run[-1] < best_loss:
                best_model = model_full
                best_model_loss = loss_full
                best_loss = loss_run[-1]

    C, S = best_model.get_model_params()
    # sort archetypes according to their maximum to achieve consistent colors
    maxidx = np.zeros(C.shape[1])
    for k in range(C.shape[1]):
        maxidx[k] = np.argmax(C[:, k])

    C_sort = C[:, np.argsort(maxidx)]
    S_sort = S[:, :, :, np.argsort(maxidx)]

    plt.figure()
    plt.plot(best_model_loss)

    plt.figure()
    plt.plot(1000 * times[C_idx], C_sort.detach())

    fig, axs = plt.subplots(2, 3)
    bottom = torch.zeros(180)
    for m in range(M):
        for c in range(L):
            for k in range(C.shape[1]):
                axs[m, c].bar(
                    times,
                    torch.mean(S_sort[m, :, c, k, :].detach(), dim=0),
                    width=(max(times) - min(times)) / len(times),
                    bottom=bottom,
                )
                bottom += torch.mean(S_sort[m, :, c, k, :].detach(), dim=0)
    visualize_AA_trajectory.visualize_AA_trajectory(S_sort, type="full")

elif model == 1:
    for outer in range(1):
        for inner in range(100):
            print(inner)
            model_equalarchetypes = TMMSAA.TMMSAA(
                dimensions=(B, N, int(P * 3)),
                num_modalities=M,
                num_comp=5,
                model="DAA",
                C_idx=torch.squeeze(torch.tile(C_idx, (1, 3))),
            )
            optimizer = torch.optim.Adam(model_equalarchetypes.parameters(), lr=0.1)
            loss_equalarchetypes = TMMSAA_trainer.Optimizationloop(
                model=model_equalarchetypes,
                X=X_train_cat,
                Xtilde=X_train_l2_cat,
                Optimizer=optimizer,
                max_iter=10000,
                tol=1e-16,
            )

            loss_run.append(loss_equalarchetypes[-1])
            if loss_run[-1] < best_loss:
                best_model = model_equalarchetypes
                best_model_loss = loss_equalarchetypes
                best_loss = loss_run[-1]

    C, S = best_model.get_model_params()
    C3 = np.mean(np.stack((C[0:160], C[160:320], C[320:]), axis=2), axis=2)
    # sort archetypes according to their maximum to achieve consistent colors
    maxidx = np.zeros(C.shape[1])
    for k in range(C.shape[1]):
        maxidx[k] = np.argmax(C3[:, k])

    C_sort = C[:, np.argsort(maxidx)]
    S_sort = S[:, :, np.argsort(maxidx)]

    plt.figure()
    plt.plot(best_model_loss)

    figC, ax = plt.subplots(3, 1)
    ax[0].plot(1000 * times[C_idx], C_sort[0:160].detach())
    ax[1].plot(1000 * times[C_idx], C_sort[160:320].detach())
    ax[2].plot(1000 * times[C_idx], C_sort[320:].detach())

    fig, axs = plt.subplots(2, 3)
    bottom = torch.zeros(180)
    delims = (0, 180, 360, 540)
    for m in range(M):
        for c in range(L):
            for k in range(C.shape[1]):
                axs[m, c].bar(
                    times,
                    torch.mean(
                        S_sort[m, :, k, delims[c] : delims[c + 1]].detach(), dim=0
                    ),
                    width=(max(times) - min(times)) / len(times),
                    bottom=bottom,
                )
                bottom += torch.mean(
                    S_sort[m, :, k, delims[c] : delims[c + 1]].detach(), dim=0
                )

    visualize_AA_trajectory.visualize_AA_trajectory(S_sort, type="concatenated")


y = 7
