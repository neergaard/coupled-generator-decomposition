import mne
import numpy as np
import pandas as pd

# import pyvista as pv

from projects.mnieeg.evaluation_viz_surf import get_central_fsaverage

from projects.facerecognition import utils
from projects.facerecognition.config import Config
from projects.facerecognition.preprocess import prepare_contrasts

# For plotting on (full resolution) fsaverage instead of fsaverage5

# from simnibs.simulation.eeg_mne_tools import make_source_spaces, make_source_morph
# from simnibs.simulation.eeg import FsAverage

# def make_morph_to_fsaverage(subdivision=5):
#     fsaverage = FsAverage()
#     fs_central = fsaverage.get_central_surface()
#     fs_sphere = fsaverage.get_surface("sphere")
#     fsaverage = make_source_spaces(fs_central, subject_id='fsaverage')

#     fsaverage5 = FsAverage(subdivision)
#     fs5_central = fsaverage5.get_central_surface()
#     fs5_sphere = fsaverage5.get_surface("sphere")
#     fsaverage5 = make_source_spaces(fs5_central, subject_id='fsaverage5')

#     return make_source_morph(fsaverage5, fsaverage, fs5_sphere, fs_sphere)

# fsaverage = FsAverage()
# fs_central = fsaverage.get_central_surface()
# fs_sphere = fsaverage.get_surface("sphere")
# fsaverage = make_source_spaces(fs_central, subject_id='fsaverage')
# fsaverage.save('/mrhome/jesperdn/INN_JESPER/projects/facerecognition/resources/fsaverage_full_central-src.fif')

# morph = make_morph_to_fsaverage(Config.inverse.FSAVERAGE)
# morph.save('/mrhome/jesperdn/INN_JESPER/projects/facerecognition/resources/fsaverage5_to_fsaverage7-morph')


def fit_normal(x):
    return x.mean(0), np.cov(x, rowvar=False)


def fit_normal_using_ransac(x):
    # x : coordinates (n, 3)
    n_iter = 100
    n_inliers = 10
    good_fit_cutoff = 0.01
    good_model_cutoff = 80

    # x0 = (x - x.mean(0)) / x.std()
    n = x.shape[0]
    idx = np.arange(n)

    assert good_model_cutoff < n - n_inliers

    rng = np.random.default_rng(seed=0)

    best_sum_pdf = 0
    goodcount = 0
    for _ in range(n_iter):
        pidx = rng.permutation(idx)

        hypo_inliers = pidx[:n_inliers]
        x_inliers = x[hypo_inliers]

        test = pidx[n_inliers:]
        x_test = x[test]

        mu, cov = fit_normal(x_inliers)
        normal = multivariate_normal(mean=mu, cov=cov)
        good_data_points = normal.pdf(x_test) >= good_fit_cutoff

        if good_data_points.sum() >= good_model_cutoff:
            goodcount += 1
            x_this = np.concatenate((x_inliers, x_test[good_data_points]))
            mu, cov = fit_normal(x_this)
            normal = multivariate_normal(mean=mu, cov=cov)
            sum_pdf = normal.pdf(x_this).sum()
            if sum_pdf > best_sum_pdf:
                inliers = list(hypo_inliers) + list(test[good_data_points])
                best_mu, best_cov = mu, cov
                best_sum_pdf = sum_pdf

    return best_mu, best_cov


# c = np.zeros(len(x))
# c[inliers] = 1
# q,w = np.mgrid[-3:6:.1, -3:6:.1]
# r = multivariate_normal(mean=best_mu, cov=best_cov)
# plt.figure()
# plt.contourf(q,w,r.pdf(np.dstack((q,w))))
# plt.scatter(*x.T,c=c)

# r = multivariate_normal(mean=x.mean(0), cov=np.cov(x,rowvar=False))
# plt.figure()
# plt.contourf(q,w,r.pdf(np.dstack((q,w))))
# plt.scatter(*x.T)


def add_contrasts(stc_dict):
    for contrast in Config.conditions.CONTRASTS:
        stc_dict[contrast.name] = sum(
            stc_dict[c] * w for c, w in zip(contrast.conditions, contrast.weights)
        )


def get_time_window(evoked, time_window, time_delta):
    """Get time window (as slice object) centered around the peak signal in
    `time_window` +/- `time_delta` (including both end points).
    """
    assert len(time_window) == 2

    twin_start, twin_stop = evoked.time_as_index(time_window)
    data = np.abs(evoked.data[:, slice(twin_start, twin_stop)])
    _, tmax = np.where(data == data.max())
    twin_center = twin_start + tmax[0]

    twin_center_s = evoked.times[twin_center]
    twin_start_s = twin_center_s - time_delta
    twin_stop_s = twin_center_s + time_delta
    return twin_start_s, twin_stop_s


def estimate_lambda2(inv, evokeds):
    # Estimate regularization from the average SNR over conditions
    lambda2, snr_dict = {}, {}
    for evoked in evokeds:
        snr = mne.minimum_norm.estimate_snr(evoked, inv)[0]
        time_window_idx = evoked.time_as_index(Config.inverse.TIME_WINDOW)
        snr_estimate = snr[time_window_idx].mean()
        lambda2_ = 1 / snr_estimate ** 2
        snr_dict[evoked.comment] = snr_estimate
        lambda2[evoked.comment] = lambda2_
        print(f"Condition {evoked.comment}")
        print(f"Estimated SNR     : {snr_estimate}")
        print(f"Estimated lambda2 : {lambda2_}")

    # snr = np.stack([mne.minimum_norm.estimate_snr(e, inv)[0] for e in evokeds]).mean(0)
    # time_window_idx = evokeds[0].time_as_index(Config.inverse.TIME_WINDOW)
    # snr_estimate = snr[time_window_idx].mean()
    # lambda2 = 1 / snr_estimate ** 2
    # print(f"Estimated SNR     : {snr_estimate}")
    # print(f"Estimated lambda2 : {lambda2}")
    # lambda2 = 1e-7
    return lambda2, snr_dict


def compute_source_estimate(subject_id):

    kw_morph = dict(stage="forward", suffix="morph", extension="h5")

    io = utils.SubjectIO(subject_id)

    info = mne.io.read_info(io.data.get_filename(stage="forward", suffix="info"))

    io.data.update(stage="preprocess", **Config.preprocess.USE_FOR_CONTRAST)
    noise_cov = mne.read_cov(io.data.get_filename(space="noise", suffix="cov"))
    data_cov = mne.read_cov(io.data.get_filename(space="data", suffix="cov"))

    epochs = mne.read_epochs(io.data.get_filename(suffix="epo"))
    # evokeds = [epochs[c].average() for c in epochs.event_id.keys()]
    evokeds = [epochs[c].average() for c in Config.conditions.CONDITIONS]
    for evoked, condition in zip(evokeds, Config.conditions.CONDITIONS):
        evoked.comment = condition
    evokeds_contrasts = prepare_contrasts(epochs)

    fname = io.data.get_filename(
        stage="preprocess", **Config.preprocess.USE_FOR_CONTRAST, suffix="ave"
    )
    mne.write_evokeds(fname, evokeds + evokeds_contrasts)

    times = {
        e.comment: get_time_window(
            e, Config.inverse.TIME_WINDOW, Config.inverse.TIME_DELTA
        )
        for e in evokeds_contrasts
    }

    io.data.update(stage="inverse")
    io.data.path.ensure_exists()
    io.data.update(**{k: None for k in Config.preprocess.USE_FOR_CONTRAST})  # reset

    # nave = np.round(np.mean([e.nave for e in evokeds])).astype(int)

    morph_ref = mne.read_source_morph(
        io.data.get_filename(forward=Config.forward.REFERENCE, **kw_morph)
    )
    vertnos = np.concatenate(morph_ref.vertices_to)
    hemis = np.concatenate(
        [[h] * len(v) for h, v in zip(("lh", "rh"), morph_ref.vertices_to)]
    )
    rows = pd.MultiIndex.from_arrays([hemis, vertnos], names=["Hemi", "Source"])
    cols = pd.MultiIndex.from_product(
        (
            [io.subject],
            Config.inverse.METHODS,
            Config.forward.MODELS,
            [c.name for c in Config.conditions.CONTRASTS],
        ),
        names=["Subject", "Inverse", "Forward", "Contrast"],
    )
    df = pd.DataFrame(columns=cols, index=rows, dtype=float)

    lambda2_dict, snr_dict = {}, {}
    for forward in Config.forward.MODELS:
        print(f"Forward : {forward}")
        io.data.update(forward=forward)
        fwd = mne.read_forward_solution(
            io.data.get_filename(stage="forward", suffix="fwd")
        )
        morph = mne.read_source_morph(
            io.data.get_filename(stage="forward", suffix="morph", extension="h5")
        )

        # Make inverse operator
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov)  # , depth=2)
        mne.minimum_norm.write_inverse_operator(io.data.get_filename(suffix="inv"), inv)
        lambda2, snr = estimate_lambda2(inv, evokeds)
        lambda2_dict[forward] = lambda2
        snr_dict[forward] = snr
        stc_all_dict = {}
        for inverse, im_type in zip(Config.inverse.METHODS, Config.inverse.METHOD_TYPE):
            print(f"Inverse : {inverse}")
            stc_dict = {}
            # io.data.update(inverse=inverse, suffix="inv")

            if im_type in ("mne", "beamformer"):
                if im_type == "beamformer":
                    # estimate common filters
                    # reg=0.05
                    filters = mne.beamformer.make_lcmv(
                        evokeds[0].info,
                        fwd,
                        data_cov,
                        noise_cov=noise_cov,  # weight_norm='nai'
                    )
                    # apply
                    for evoked in evokeds:
                        stc_dict[evoked.comment] = mne.beamformer.apply_lcmv(
                            evoked, filters
                        )
                elif im_type == "mne":
                    for evoked in evokeds:
                        this_inv = mne.minimum_norm.prepare_inverse_operator(
                            inv, evoked.nave, lambda2[evoked.comment], inverse
                        )
                        stc_dict[evoked.comment] = mne.minimum_norm.apply_inverse(
                            evoked,
                            this_inv,
                            lambda2[evoked.comment],
                            inverse,
                            prepared=True,
                        )
                else:
                    raise ValueError
                # Restrict time window
                for k in stc_dict:
                    stc_dict[k].crop(*Config.inverse.TIME_WINDOW)

                add_contrasts(stc_dict)
                # remove individual conditions
                for evoked in evokeds:
                    del stc_dict[evoked.comment]

            elif im_type == "dipole":
                raise NotImplementedError
                # dipole = mne.dipole.fit_dipole(evoked, noise_cov)

            elif im_type == "music":
                # form the ERP for each contrast and apply RAP-MUSIC
                for contrast in Config.conditions.CONTRASTS:
                    this_evokeds = [
                        e
                        for c in contrast.conditions
                        for e in evokeds
                        if e.comment == c
                    ]

                    evoked = mne.combine_evoked(this_evokeds, contrast.weights)
                    evoked.comment = contrast.name

                    # restrict to time window of interest
                    evoked.crop(*Config.inverse.TIME_WINDOW)

                    dipoles = mne.beamformer.rap_music(
                        evoked, fwd, noise_cov, n_dipoles=2
                    )

                    # Convert Dipole object to STC
                    vertices = [s["vertno"] for s in fwd["src"]]
                    vertices_rr = np.concatenate(
                        [s["rr"][s["vertno"]] for s in fwd["src"]]
                    )
                    n_times = len(dipoles[0].times)
                    n_src = sum(len(h) for h in vertices)
                    tmin = dipoles[0].times[0]
                    tstep = dipoles[0].times[1] - tmin
                    stc = mne.SourceEstimate(
                        np.zeros((n_src, n_times)), vertices, tmin, tstep
                    )
                    for dip in dipoles:
                        ix = np.linalg.norm(vertices_rr - dip.pos[0], axis=-1).argmin()
                        stc.data[ix] = np.abs(dip.amplitude)
                    stc_dict[contrast.name] = stc
            stc_all_dict[inverse] = stc_dict

        # utils.write_pickle(
        #     stc_all_dict, io.data.get_filename(suffix="stc", extension="pickle"),
        # )

        # morph to fsaverage and compute average in time window
        for inv in stc_all_dict:
            for con, this_stc in stc_all_dict[inv].items():
                this_stc_fs = morph.apply(this_stc)
                stc_all_dict[inv][con] = this_stc_fs.copy()
                this_stc_fs.crop(*times[con])
                df[io.subject, inv, forward, con] = this_stc_fs.mean().data.squeeze()

        utils.write_pickle(
            stc_all_dict,
            io.data.get_filename(suffix="stc", extension="pickle", space="fsaverage"),
        )
    # io.data.update(inverse=None)
    io.data.update(forward=None)

    fname = io.data.get_filename(suffix="stc", extension="pickle", space="fsaverage")
    df.to_pickle(fname)

    fname = io.data.get_filename(suffix="lambda2", extension="pickle")
    utils.write_pickle(lambda2_dict, fname)

    fname = io.data.get_filename(suffix="snr", extension="pickle")
    utils.write_pickle(snr_dict, fname)


def fit_dipole_linear(evoked, noise_cov, fwd):
    whitener = mne.cov.compute_whitener(noise_cov)

    evoked.data
    fwd["sol"]["data"]


def inspect_solution(subject_id):

    condition = "faces vs. scrambled"
    zoom_factor = 1.1

    io = utils.SubjectIO(subject_id)
    evokeds = mne.read_evokeds(
        io.data.get_filename(
            stage="preprocess", **Config.preprocess.USE_FOR_CONTRAST, suffix="ave"
        )
    )

    io.data.update(stage="inverse", space="fsaverage", suffix="stc", extension="pickle")

    fsavg = get_central_fsaverage()

    evoked = evokeds[-1]
    assert evoked.comment == condition
    evoked.pick_types(eeg=True)

    times = get_time_window(
        evoked, Config.inverse.TIME_WINDOW, Config.inverse.TIME_DELTA
    )
    times_s = evoked.times[[times.start, times.stop]]
    print(
        f"Showing average activity in {1e3*times_s[0]:.0f} - {1e3*times_s[1]:.0f} ms time window"
    )

    # MNE tries to use the "extra" digitization points (which are in the nasal
    # area) resulting in a wrong head radius estimate so fit to EEG channels
    # instead
    # radius, _, _ = mne.bem.fit_sphere_to_headshape(
    #     evoked.info, dig_kinds=["cardinal", "eeg"]
    # )

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # evoked.plot(spatial_colors=True, axes=ax, show=False, sphere=radius)
    # ymin, ymax = ax.get_ylim()
    # rect = plt.Rectangle(
    #     [times_s[0], ymin],
    #     times_s[1] - times_s[0],
    #     ymax - ymin,
    #     alpha=0.25,
    #     color="gray",
    #     # edgecolor=None,
    #     # zorder=0,
    # )
    # ax.add_patch(rect)
    # ax.grid(alpha=0.25)
    # fig.show()

    stcs = {}
    for inverse in Config.inverse.METHODS:
        io.data.update(inverse=inverse)
        for forward in Config.forward.MODELS:
            stc = utils.read_pickle(io.data.get_filename(forward=forward))[condition]
            stcs[(inverse, forward, "lh")] = stc.lh_data[:, times].mean(1)
            stcs[(inverse, forward, "rh")] = stc.rh_data[:, times].mean(1)

    keys = list(stcs.keys())
    data = np.array(list(stcs.values()))

    # Normalize each inverse solution (rows)
    # vmax = {
    #     inverse: np.abs(data[[inverse in k for k in keys]]).max()
    #     for inverse in Config.inverse.METHODS
    # }
    # for k, v in stcs.items():
    #     v /= vmax[k[0]]

    # Normalize each (inverse, forward) pair seperately
    vmax = {
        (inverse, forward): np.abs(
            data[[inverse in k and forward in k for k in keys]]
        ).max()
        for inverse in Config.inverse.METHODS
        for forward in Config.forward.MODELS
    }
    for k, v in stcs.items():
        v /= vmax[k[0], k[1]]

    # vmin, vmax = np.percentile(np.stack(list(data.values())), [1, 99])
    scalar_bar_kw = dict(
        clim=[-0.9, 0.9], cmap="coolwarm", below_color="magenta", above_color="yellow"
    )

    pixels = 300  # pixel per subplot
    n = len(Config.forward.MODELS)
    m = len(Config.inverse.METHODS)
    p = pv.Plotter(shape=(m, n), window_size=(n * 300, m * 300))
    for row, inverse in enumerate(Config.inverse.METHODS):
        io.data.update(inverse=inverse)
        for col, forward in enumerate(Config.forward.MODELS):
            # if inverse != "sLORETA":
            #    continue
            p.subplot(row, col)
            if col == 0:
                p.add_text(inverse, "left_edge")
            if row == 0:
                p.add_text(forward)
            # stc = utils.read_pickle(io.data.get_filename(forward=forward))[condition]
            # times = slice(*stc.time_as_index(Config.inverse.TIME_WINDOW))
            # times = slice(*stc.time_as_index([0.1, 0.12]))
            # p.add_mesh(fsavg["lh"].copy(), scalars=stc.lh_data[:, times].mean(1))
            # p.add_mesh(fsavg["rh"].copy(), scalars=stc.rh_data[:, times].mean(1))

            p.add_mesh(
                fsavg["lh"].copy(),
                scalars=stcs[inverse, forward, "lh"],
                **scalar_bar_kw,
            )
            p.add_mesh(
                fsavg["rh"].copy(),
                scalars=stcs[inverse, forward, "rh"],
                **scalar_bar_kw,
            )
            p.view_xy(True)
            p.camera.zoom(zoom_factor)
    p.link_views()
    p.show()

    # import meshio
    # import pyvista as pv
    # from projects.mnieeg.evaluation import get_central_fsaverage

    # fsavg = get_central_fsaverage()
    # fsavg5 = fsavg.combine()
    # fsavg5 = pv.PolyData(fsavg5.points, fsavg5.cells)

    # points = fsavg5.points
    # cells = [("triangle", fsavg5.faces.reshape(-1, 4)[:, 1:])]

    # for k, v in stc_dict.items():
    #     filename = "/home/jesperdn/nobackup/new2ft_" + "_".join(k.split()) + ".xdmf"
    #     # filename = '/home/jesperdn/nobackup/presubtracted.xdmf'
    #     with meshio.xdmf.TimeSeriesWriter(filename) as writer:
    #         writer.write_points_cells(points, cells)
    #         for t, data in zip(v.times, v.data.T):
    #             writer.write_data(t, point_data={"source estimate": data})

    # # visualize with MNE (doesn't seem to work...)

    # subjects_dir = '/mrhome/jesperdn/git_repos/simnibs/simnibs/resources/templates/freesurfer'
    # stc.plot(subject='fsaverage5', subjects_dir=subjects_dir)
