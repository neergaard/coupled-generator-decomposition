from projects.facerecognition.evaluation_viz_surf import _crop_imgs_from_dict


x = np.load("/home/jesperdn/nobackup/forward.npy")
u, s, v = np.linalg.svd(x, full_matrices=False)
v = v.T

v /= np.percentile(np.abs(v), 99, 0)

overlay_kwargs = dict(cmap="RdBu_r", show_scalar_bar=False)
zoom_factor = np.sqrt(2)
plotter_kwargs = dict(off_screen=True, window_size=(400, 400))

comps = [0, 1, 5, 6, 7, 20, 21, 25, 26, 27, 40, 41, 45, 46, 47]

imgs = {}
# for i in np.linspace(50,59,10):
for i in comps:
    i = int(i)
    p = pv.Plotter(**plotter_kwargs)
    p = brain.plot(
        dict(lh=v[: len(v) // 2, i], rh=v[len(v) // 2 :, i]),
        name="x",
        overlay_kwargs=overlay_kwargs,
        plotter=p,
    )
    p.view_xy()
    p.camera.zoom(zoom_factor)
    p.enable_parallel_projection()
    imgs[i] = p.screenshot(transparent_background=True)
    p.close()

rows, cols, aspect_ratio = _crop_imgs_from_dict(imgs, "all not equal", 255)

fig, axes = plt.subplots(
    3,
    5,
    figsize=figure_sizing.get_figsize("double", 1 / aspect_ratio, subplots=(3, 5)),
    constrained_layout=True,
)

for k, ax in zip(imgs, axes.flat):
    ax.imshow(imgs[k][rows, cols])
    ax.set_title(k)
    ax.set_axis_off()

fig.savefig("/home/jesperdn/fwd_source_modes.png")


plt.figure(figsize=figure_sizing.get_figsize("single"))
plt.scatter(np.arange(len(s)), s / s.max(), marker=".")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title("Singular Values of Gain Matrix")
plt.grid(alpha=0.25)
plt.savefig("/home/jesperdn/fwd_svd_vals.pdf")


info = create_info()
kwargs = dict(cmap="RdBu_r", show=False)


fig, axes = plt.subplots(
    3,
    5,
    figsize=figure_sizing.get_figsize("double", 1.2, subplots=(3, 5)),
    constrained_layout=True,
)
for i, ax in zip(comps, axes.flat):
    im, cs = mne.viz.plot_topomap(u[:, i], info, axes=ax, contours=0, **kwargs,)
    # ax.clabel(cs, cs.levels, fontsize=6)
    ax.set_title(i)

fig.savefig("/home/jesperdn/fwd_sensor_modes.pdf")


tri = [s['tris'] for s in fwd['src']]
pts = np.concatenate([s['rr'] for s in fwd['src']], axis=0)
pts *= 1e3

tree = scipy.spatial.cKDTree(pts)
dist,idx = tree.query(pts, 200)
i = (dist > 10) & (dist < 15)
m = i.sum(1).min()
i = np.array([np.where(ii)[0][:m] for ii in i])
ix = np.array([idx[j,jj] for j,jj in enumerate(i)])

nrdm = []
for i in range(fwd.shape[1]):
    nrdm.append(np.array([rdm(fwd[:,i], x) for x in fwd[:,ix[i]].T]).mean())
nrdm = np.array(nrdm) 


ii = idx[np.arange(len(idx)), np.abs(dist - 10).argmin(1)]

tri = np.concatenate([t+i for t,i in zip(tri,[0,10242])], axis=0)
a= get_adjacency_matrix(tri)
aa = a @ a
fwd = fwd_ref["sol"]["data"][:,2::3]

nrdm = []
for i in range(fwd.shape[1]):
    nrdm.append(np.array([rdm(fwd[:,i], x) for x in fwd[:,aa[i].indices].T]).mean())
nrdm = np.array(nrdm) 



x = df['RDM', 'Man-Template', "01", 'normal']
y = dfi['Man-Template', 'Dipole', 8, "psf", 'peak_err']
plt.scatter(x,y)

x = df['RDM', 'Custom-Template', "01", 'normal'] + 0.25*nrdm
y = dfi['Custom-Template', 'MUSIC', 8, "psf", 'peak_err']
plt.scatter(x,y,marker='.')

x = nrdm
y = dfi['Custom-Template', 'MUSIC', 8, "psf", 'peak_err']
plt.scatter(x,y,marker='.')


np.save("/mrhome/jesperdn/neighbor_rdm_sub01_ix", nrdm)

plt.scatter(nrdm, df['RDM', 'Custom-Template', "01", 'normal'])