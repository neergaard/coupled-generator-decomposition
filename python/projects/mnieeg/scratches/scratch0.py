q = np.row_stack((phatmags, khm[None], custom[None], np.median(phatmags, 0)[None]))
c = np.concatenate((np.zeros(33), np.array([1]), np.array([2]), np.array([3])))

embedding = sklearn.manifold.MDS()
# embedding.fit(q.reshape(35, -1))
x = embedding.fit_transform(q.reshape(len(q), -1))

plt.scatter(*x.T, c=c)
plt.colorbar()


w = dist.loc[["custom_nonlin", "custom_nonlin_opt"]].to_numpy().reshape(2, 32, 63)
plt.figure(figsize=(10, 5))
parts = plt.violinplot(
    w[0].T, positions=np.arange(1, 33) - 0.2, widths=0.3, showmeans=True
)
for pc in parts["bodies"]:
    # pc.set_facecolor('r')
    pc.set_edgecolor(pc.get_facecolor())
    # pc.set_alpha(1)
parts = plt.violinplot(
    w[1].T, positions=np.arange(1, 33) + 0.2, widths=0.3, showmeans=True
)
for pc in parts["bodies"]:
    # pc.set_facecolor('b')
    pc.set_edgecolor(pc.get_facecolor())
plt.grid(True, alpha=0.5)

plt.figure()
plt.violinplot((w[1] - w[0]).T, showmeans=True)
plt.plot([0, 32], [0, 0], color="gray", alpha=0.5)

q = w[0] - w[1]

np.sum(w[0].mean(-1) > w[1].mean(-1))

plt.figure()
plt.bar(np.arange(32) - 0.2, w[0].mean(-1), width=0.2)
plt.bar(np.arange(32) + 0.2, w[1].mean(-1), width=0.2)

plt.figure()
plt.bar(np.array(montage.ch_names, int) - 0.2, w[0].mean(0), width=0.2)
plt.bar(np.array(montage.ch_names, int) + 0.2, w[1].mean(0), width=0.2)

fig = plt.figure()
ax = fig.add_subplot()
im, _ = mne.viz.plot_topomap(
    w[1].mean(0) - w[0].mean(0),
    info,
    sensors=False,
    contours=False,
    axes=ax,
    names=montage.ch_names,
    show_names=True,
)
cbar = fig.colorbar(im, ax=ax, shrink=1, pad=0.025)
cbar.set_label("mm")  # , rotation=-90)
