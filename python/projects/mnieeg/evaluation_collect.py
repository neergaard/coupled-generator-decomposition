import mne
import mne.channels._standard_montage_utils
import numpy as np
import pandas as pd
import scipy.stats

from projects.base.forward_evaluation import rdm, lnmag
from projects.mnieeg import utils
from projects.mnieeg.config import Config
from projects.facerecognition.utils import fsaverage_as_index

from simnibs.simulation import eeg

mne.set_log_level("warning")


def channel_collect():
    io = utils.GroupIO()
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    n_subjects = len(io.subjects)

    first = next(iter(io.subjects))
    info = mne.io.read_info(
        io.data.get_filename(
            subject=first,
            session=io.subjects[first],
            stage="forward",
            forward=Config.forward.ALL_MODELS[0],
            suffix="info",
        )
    )
    n_channels = len(mne.pick_types(info, eeg=True))

    models = [i for i in Config.forward.ALL_MODELS if "template" not in i]
    final_model_names = [Config.plot.ch_model_name[m] for m in models]

    # models += ['custom_nonlin_opt']
    n_model = len(models)

    axes = ("x", "y", "z")
    n_axes = len(axes)

    print("Loading channel positions")
    data = np.zeros((n_model, n_subjects, n_axes, n_channels))
    for i, (subject, session) in enumerate(io.subjects.items()):
        trans = mne.read_trans(
            io.data.get_filename(subject=subject, session=session, suffix="trans")
        )
        trans = trans["trans"]
        trans[:3, 3] *= 1e3
        io.simnibs.update(subject=subject)
        subject_dir = io.simnibs.get_path("subject")
        for j, model in enumerate(models):
            montage = eeg.make_montage(subject_dir / f"montage_{model}_proj.csv")
            montage.apply_trans(trans)
            data[j, i] = montage.ch_pos.T
    data = np.ascontiguousarray(data)

    index = pd.MultiIndex.from_product(
        [final_model_names, io.subjects.keys(), axes],
        names=("Forward", "Subject", "Axis"),
    )

    data = data.reshape(len(index), len(info["ch_names"]))
    df = pd.DataFrame(data, columns=info["ch_names"], index=index)
    df.to_pickle(Config.path.RESULTS / "channel.pickle")

    # Optimization
    io.data.update(
        prefix="montage", stage="preprocessing", task=None, extension="pickle"
    )
    io.data.update(suffix="ref")
    dfref = pd.concat(
        [
            pd.read_pickle(io.data.get_filename(subject=sub, session=ses))
            for sub, ses in io.subjects.items()
        ],
        axis="columns",
    )

    io.data.update(suffix="opt")
    dfopt = pd.concat(
        [
            pd.read_pickle(io.data.get_filename(subject=sub, session=ses))
            for sub, ses in io.subjects.items()
        ],
        axis="columns",
    )

    dfref.to_pickle(Config.path.RESULTS / "montage_ref.pickle")
    dfopt.to_pickle(Config.path.RESULTS / "montage_opt.pickle")


def forward_collect():
    """Compute RDM and lnMAG *in fsaverage*. Forward solutions are morphed to
    fsaverage *prior* to computing RDM/lnMAG because the source space of the
    forward model(s) based on template warping is different from the rest
    (which is based on the actual MRIs of the subject).
    """
    io = utils.GroupIO()
    io.data.update(stage="forward", space="fsaverage", suffix="fwd")
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    force_fixed = Config.inverse.orientation_prior not in (False, "auto")

    metrics = ("RDM", "lnMAG")
    n_metrics = len(metrics)

    ref = Config.forward.REFERENCE
    models = [i for i in Config.forward.MODELS if i != ref]
    final_model_names = [Config.plot.fwd_model_name[m] for m in models]

    n_models = len(models)

    n_subjects = len(io.subjects)

    first = next(iter(io.subjects))
    n_src = mne.read_forward_solution(
        io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
    )["nsource"]

    ori = ("normal",) if force_fixed else ("x", "y", "z")
    n_ori = len(ori)
    data = np.zeros((n_metrics, n_models, n_subjects, n_src, n_ori))

    print(f"Computing evaluation metrics {metrics}")
    for i, (subject, session) in enumerate(io.subjects.items()):
        io.data.update(subject=subject, session=session, forward=ref)
        fwd_ref = mne.read_forward_solution(io.data.get_filename())
        fwd_ref = mne.convert_forward_solution(fwd_ref, force_fixed=force_fixed)
        fwd_ref = fwd_ref["sol"]["data"]
        for j, model in enumerate(models):
            io.data.update(forward=model)
            fwd = mne.read_forward_solution(io.data.get_filename())
            fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed)
            fwd = fwd["sol"]["data"]
            data[0, j, i] = rdm(fwd, fwd_ref).reshape(n_src, n_ori)
            data[1, j, i] = lnmag(fwd, fwd_ref).reshape(n_src, n_ori)
    data = np.ascontiguousarray(data.transpose((3, 0, 1, 2, 4)))

    print("Saving as DataFrame")
    index = fsaverage_as_index(Config.inverse.FSAVERAGE)
    columns = pd.MultiIndex.from_product(
        [metrics, final_model_names, io.subjects.keys(), ori],
        names=("Metric", "Forward", "Subject", "Orientation"),
    )
    data = data.reshape(len(index), len(columns))
    df = pd.DataFrame(data, columns=columns, index=index)  # .sort_index(axis=1)
    df.to_pickle(Config.path.RESULTS / "forward.pickle")


def forward_collect_distance_matrix():

    io = utils.GroupIO()
    io.data.update(stage="forward", space="fsaverage", suffix="fwd")
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    force_fixed = Config.inverse.FIXED not in (False, "auto")

    metrics = ("RDM", "|lnMAG|")

    ref = Config.forward.REFERENCE
    models = [ref] + [i for i in Config.forward.MODELS if i != ref]
    final_model_names = [Config.plot.fwd_model_name[m] for m in models]

    n_models = len(models)

    n_subjects = len(io.subjects)

    first = next(iter(io.subjects))
    io.subjects[first]
    n_src = mne.read_forward_solution(
        io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
    )["nsource"]
    n_chan = mne.read_forward_solution(
        io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
    )["nchan"]

    ori = ("normal",) if force_fixed else ("x", "y", "z")
    n_ori = len(ori)
    data = np.zeros((n_models, n_subjects, n_chan, n_src, n_ori))

    print("Collecting forward solutions")
    for i, (subject, session) in enumerate(io.subjects.items()):
        io.data.update(subject=subject, session=session)
        for j, model in enumerate(models):
            io.data.update(forward=model)
            fwd = mne.read_forward_solution(io.data.get_filename())
            fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed)
            data[j, i] = fwd["sol"]["data"].reshape(n_chan, n_src, n_ori)
    data = data.reshape(n_models * n_subjects, n_chan, n_src, n_ori).squeeze()

    print("Computing distance matrices")
    abs_lnmag = lambda x, y: np.abs(lnmag(x, y))
    metric_fun = dict(zip(metrics, (rdm, abs_lnmag)))
    dist = {}
    for metric in metrics:
        fun = metric_fun[metric]
        dist[metric] = np.zeros((n_models * n_subjects, n_models * n_subjects))
        for i in range(n_models * n_subjects):
            for j in range(n_models * n_subjects):
                if i == j:
                    dist[metric][i, j] = 0
                elif i < j:
                    dist[metric][i, j] = fun(data[i], data[j]).mean()
                elif i > j:
                    dist[metric][i, j] = dist[metric][j, i]

    print("Saving as DataFrames")
    cols = pd.MultiIndex.from_product(
        (final_model_names, io.subjects), names=("Forward", "Subject")
    )
    rows = pd.MultiIndex.from_product(
        (metrics, final_model_names, io.subjects),
        names=("Metric", "Forward", "Subject"),
    )
    df = pd.DataFrame(np.concatenate((list(dist.values()))), columns=cols, index=rows)
    df.to_pickle(Config.path.RESULTS / "forward_distance_matrix.pickle")


def inverse_collect():
    io = utils.GroupIO()
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    name = "inverse"
    for prefix in ("free", "fixed"):
        io.data.update(
            prefix=prefix,
            stage="source",
            space="fsaverage",
            suffix="res",
            extension="pickle",
        )
        try:
            df = pd.concat(
                [
                    pd.read_pickle(io.data.get_filename(subject=sub, session=ses))
                    for sub, ses in io.subjects.items()
                ],
                axis=1,
                keys=io.subjects.keys(),
                names=["Subject"],
            )
        except FileNotFoundError:
            continue
        this_name = name
        if prefix:
            this_name += "_" + prefix
        this_name += ".pickle"

        df.to_pickle(Config.path.RESULTS / this_name)


def inverse_compute_summary():  # results_dir
    output_dir = Config.path.RESULTS / "figure_inverse"
    if not output_dir.exists():
        output_dir.mkdir()

    ori = dict(Free="inverse.pickle", Fixed="inverse_fixed.pickle")

    df = pd.concat(
        [pd.read_pickle(Config.path.RESULTS / v) for v in ori.values()],
        axis=1,
        keys=ori.keys(),
        names=["Orientation"],
    )
    cols = [c for c in df.columns.names if c != "Subject"]

    df_summary = get_summary_statistics(df, cols)

    # stack sorts columns so reindex to those of df...
    first_subject = df.columns.unique("Subject")[0]
    reindexer = df.loc[:, pd.IndexSlice[:, first_subject]].columns.droplevel("Subject")
    df_dens = (
        df.stack("Subject")
        .reindex(reindexer, axis=1)
        .apply(make_kde, points=make_density_points())
    )
    # Fast
    # df_dens = df.stack("Subject").reindex(reindexer, axis=1).apply(make_histogram, bins=make_bins())

    df_summary.to_pickle(Config.path.RESULTS / "inverse_summary.pickle")
    df_dens.to_pickle(Config.path.RESULTS / "inverse_density.pickle")


def get_summary_statistics(df, levels, axis=1):
    return pd.concat(
        [
            # df.min(axis, level=levels),
            df.mean(axis, level=levels),
            df.median(axis, level=levels),
            # df.max(axis, level=levels),
            df.std(axis, level=levels),
        ],
        axis=1,
        # keys=["min", "mean", "median", "max", "std"],
        keys=["mean", "median", "std"],
        names=["Statistic"],
    )


def make_density_points(n_points=201):
    vmin, vmax = Config.plot.density_point_range
    # try:
    #     vmin, vmax = Config.plot.limits["inverse"]["density"]["point_rage"]
    # except KeyError:
    #     vmin, vmax = np.percentile(df.to_numpy().ravel(), [0.001, 99])
    # print(f"Extracting densities between {vmin:.3f} and {vmax:.3f}")
    return np.linspace(vmin, vmax, n_points)


def make_bins(n_bins=200):
    vmin, vmax = Config.plot.density_point_range
    return np.linspace(vmin, vmax, n_bins + 1)


def make_kde(df, points):
    return pd.Series(scipy.stats.gaussian_kde(df)(points), points)


def make_histogram(df, bins):
    index = bins[:-1] + (bins[1] - bins[0]) / 2
    return pd.Series(np.histogram(df, bins, density=True)[0], index)
