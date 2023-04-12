"""
- lh.sphere.freesurfer.gii is identical to fsaverage's lh.sphere
- lh.central.freesurfer.gii is identical to fsaverage's (lh.white + lh.pial) / 2
- lh.inflated.freesurfer.gii is not the same as fsaverage's lh.inflated; the
triangulation *is* the same though but the CAT surface is smaller and slightly
moved
"""


def stat_fun(data, factor_levels, effect):
    # data is expected to be a list (of length n_conditions) of arrays of shape
    # data[0].shape = (n_observations, n_sources)
    # f_mway_rm expects data.shape = (n_observations, n_conditions, n_sources)

    # A : A1, A2, A3
    # B : B1, B2

    # A1B1, A1B2, A2B1, A2B2, A3B1, A3B2
    # is this factor_levels = [3, 2] or [2, 3]

    return mne.stats.f_mway_rm(
        np.swapaxes(data, 0, 1),
        factor_levels=factor_levels,
        effects=effects,
        return_pvals=False,
    )[0]


# median = np.median(data, 0)
# d = np.linalg.norm(data - median, axis=-1)
# std = np.std(d, 1)
# prob = d > (np.median(d, 1) + 20)[:, None]
# subidx, chidx = np.where(prob)
# # idx = np.where(np.sum(prob,1))[0]
# sub = subidx + 1

# labels = np.arange(1, 62)

# x = pv.PolyData(data[9])
# p = pv.Plotter(notebook=False)
# # p.add_mesh(x)
# p.add_point_labels(x, labels)
# p.show()

# plt.boxplot(d.T)


# x = pv.PolyData(data.reshape(-1))
# y = pv.PolyData(data[8])
# # x['x'] = np.zeros(len(x.points))
# # x['x'][61-1::28] = 1

# p = pv.Plotter(notebook=False)
# p.add_mesh(x)
# p.add_mesh(y, color="r")
# p.show()


def x():
    io = utils.GroupIO()
    io.filenamer.update(stage="forward", session="01", suffix="info")

    info = mne.io.read_info(
        io.filenamer.get_filename(
            subject=io.subjects[0], forward=Config.forward.MODELS[0]
        )
    )
    n_channels = len(mne.pick_types(info, eeg=True))
    n_subjects = len(io.subjects)

    data = np.zeros((n_subjects, n_channels, 3))
    for i, subject in enumerate(io.subjects):
        io.simnibs.update(subject=subject)
        f = io.simnibs.get_path("subject") / "mni_digitized_proj.csv"
        montage = eeg.make_montage(f)
        data[i] = montage.ch_pos
    data = data[:, :-2]

    mu = data.mean(0)
    mu = forward.project_to_mni_surface(mu)

    montage = eeg.make_montage("easycap_m10")
    custom = montage.ch_pos

    a = pv.PolyData(mu)
    b = pv.PolyData(data.reshape(-1, 3))
    c = pv.PolyData(custom)

    p = pv.Plotter(notebook=False)
    p.add_mesh(a)
    p.add_mesh(b)
    p.add_mesh(c)
    p.show()


def compare_cov():
    io = utils.GroupIO()
    io.data.update(stage="preprocessing")

    cov = []
    # with BlockTimer("Loading covariance matrices"):
    for subject, session in io.subjects.items():
        io.data.update(subject=subject, session=session)
        # for session in utils.get_func_sessions(io.data):
        # io.data.update(session=session)
        # Check if session exists for this subject, else move on
        # k = (subject, session)
        this_cov = mne.read_cov(io.data.get_filename(suffix="cov"))
        cov.append(this_cov["data"])
        # i2s[i] = k
        # i += 1
    cov = np.stack(cov)
    ch_names = this_cov.ch_names
    ch_names_ = [
        ch_name if i % 2 == 0 else "" for i, ch_name in enumerate(ch_names, start=1)
    ]
    nfree = this_cov["nfree"]
    projs = this_cov["projs"]

    d = cov.diagonal(axis1=1, axis2=2)

    q1, q3 = np.quantile(d, [0.25, 0.75], 0)
    iqr = q3 - q1
    limit = q3 + 1.5 * iqr

    sub, ch = np.where(d > limit)
    u, c = np.unique(sub, return_counts=True)
    [i2s[i] for i in u[c.argsort()[::-1]]]  # sorted by channel counts

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(d, labels=ch_names_)
    # ax.set_xticklabels(ax.get_xticklabels()[::2])

    avg_cov = cov.mean(0)

    u, s, v = np.linalg.svd(cov.reshape(cov.shape[0], -1), full_matrices=False)
    v = v.T
    first_comp = v[:, 0].reshape(cov.shape[1:]) * s[0]

    n = np.prod(cov.shape[1:])
    s1 = np.sum(np.sign(first_comp) == 1) / n > 0.5
    s2 = np.sum(np.sign(avg_cov) == 1) / n > 0.5

    first_comp = first_comp if s1 and s2 or not s1 and not s2 else -first_comp

    avg_cov = mne.Covariance(first_comp, ch_names, [], projs, nfree)

    mne.write_cov(Config.path.RESOURCES / "group-cov.fif", avg_cov)

    # fliers = np.concatenate([i.get_xydata() for i in box['fliers']])
    # sub = sub[ch.argsort()]
    # for s, f in zip(sub, fliers):
    #    ax.annotate(str(i2s[s]), f)

    su = "01"
    se = "01"
    raw = mne.io.read_raw(
        f"/mnt/projects/PhaTMagS/jesper/analysis/sub-{su}/ses-{se}/stage-preprocessing/sub-{su}_ses-{se}_task-rest_eeg.fif"
    )
    info = raw.info
    cov[su, se].plot(info)

    return cov


import matplotlib.pyplot as plt
import numpy as np


def plot_parcellation():

    kw["xlabel"]
    kw["ylabel"]
    kw["major_tick_params"]
    kw["minor_tick_params"]

    x
    major_ticks
    minor_tick_labels

    fig, ax = plt.subplots()
    ax.plot(x)
    ax.set_xlabel("Network")

    minor_ticks = 0.5 * (major_ticks[:-1] + major_ticks[1:])

    # Major ticks
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticklabels([], minor=False)
    # Minor ticks
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(minor_tick_labels, minor=True)
    # see help(ax.tick_params)
    ax.xaxis.set_tick_params("minor", length=0)

    plt.show()


# X subdivisions of an icosahedron (20-sided convex regular polygon)
# fsaverageX : n vertices = 2+10*2**(2*X)
# fsaverageX : n triangles = 20*2**(2*X)

x = np.arange(100)
x_labels = np.arange(0, 100 + 1, 10)
x_labels_pos = np.arange(5, 95 + 1, 10)
x_labels_pos1 = [str(i) for i in x_labels_pos]
y = np.random.random((len(x), 6))
fig, ax = plt.subplots()
ax.plot(x, y)

# Major ticks
ax.xaxis.set_ticks(x_labels, minor=False)
ax.xaxis.set_ticklabels([], minor=False)
# Minor ticks
ax.xaxis.set_ticks(x_labels_pos, minor=True)
ax.xaxis.set_ticklabels(x_labels_pos1, minor=True)
ax.xaxis.set_tick_params("minor", length=0)

ax.set_xlabel("Network")

plt.show()

fig.show()
