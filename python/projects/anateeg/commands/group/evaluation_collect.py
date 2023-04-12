import mne
import numpy as np
import pandas as pd

from projects.base.forward_evaluation import rdm, lnmag
from projects.facerecognition.utils import fsaverage_as_index
from projects.anateeg import utils
from projects.anateeg.config import Config

from projects.mnieeg.evaluation_collect import (
    get_summary_statistics,
    make_density_points,
    make_kde,
)

# from projects.mnieeg.evaluation_collect import make_bins, make_histogram


mne.set_log_level("warning")


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

    force_fixed = True  # Config.inverse.FIXED not in (False, "auto")

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

    force_fixed = True  # Config.inverse.FIXED not in (False, "auto")

    metrics = ("RDM", "|lnMAG|")

    ref = Config.forward.REFERENCE
    models = [ref] + [i for i in Config.forward.MODELS if i != ref]
    final_model_names = [Config.plot.model_name[m] for m in models]

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


# def forward_collect():
#     """Compute RDM and lnMAG *in fsaverage*. Forward solutions are morphed to
#     fsaverage *prior* to computing RDM/lnMAG because the source space of the
#     forward model(s) based on template warping is different from the rest
#     (which is based on the actual MRIs of the subject).
#     """
#     io = utils.GroupIO()
#     io.data.update(stage="forward", space="fsaverage", suffix="fwd")
#     if not Config.path.RESULTS.exists():
#         Config.path.RESULTS.mkdir()

#     force_fixed = Config.inverse.FIXED not in (False, "auto")

#     metrics = ("RDM", "lnMAG")
#     n_metrics = len(metrics)

#     ref = Config.forward.REFERENCE
#     models = [i for i in Config.forward.MODELS if i != ref]
#     final_model_names = [Config.plot.model_name[m] for m in models]
#     n_models = len(models)

#     n_subjects = len(io.subjects)

#     first = next(iter(io.subjects))
#     n_src = mne.read_forward_solution(
#         io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
#     )["nsource"]

#     ori = ("normal",) if force_fixed else ("x", "y", "z")
#     n_ori = len(ori)
#     data = np.zeros((n_metrics, n_models, n_subjects, n_src, n_ori))

#     print(f"Computing evaluation metrics {metrics}")
#     for i, (subject, session) in enumerate(io.subjects.items()):
#         io.data.update(subject=subject, session=session, forward=ref)
#         fwd_ref = mne.read_forward_solution(io.data.get_filename())
#         fwd_ref = mne.convert_forward_solution(fwd_ref, force_fixed=force_fixed)
#         fwd_ref = fwd_ref["sol"]["data"]
#         for j, model in enumerate(models):
#             io.data.update(forward=model)
#             fwd = mne.read_forward_solution(io.data.get_filename())
#             fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed)
#             fwd = fwd["sol"]["data"]
#             data[0, j, i] = rdm(fwd, fwd_ref).reshape(n_src, n_ori)
#             data[1, j, i] = lnmag(fwd, fwd_ref).reshape(n_src, n_ori)
#     data = np.ascontiguousarray(data.transpose((0, 1, 2, 4, 3)))

#     print("Saving as DataFrame")
#     index = pd.MultiIndex.from_product(
#         [metrics, final_model_names, io.subjects.keys(), ori],
#         names=("Metric", "Forward", "Subject", "Orientation"),
#     )
#     # col_names = list(map(str, range(len(central["lh"]["points"]))))
#     col_names = pd.MultiIndex.from_product(
#         [("lh", "rh"), map(str, range(n_src // 2))], names=["Hemi", "Source"]
#     )
#     data = data.reshape(len(index), len(col_names))
#     df = pd.DataFrame(data, columns=col_names, index=index)
#     df.to_pickle(Config.path.RESULTS / "forward.pickle")


def forward_collect_charm_t1_only():
    io = utils.GroupIO()
    io.data.update(stage="forward", space="fsaverage", suffix="fwd")
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    force_fixed = True  # Config.inverse.FIXED not in (False, "auto")

    metrics = ("RDM", "lnMAG")
    n_metrics = len(metrics)

    ref = Config.forward.REFERENCE
    models = ["charm"]
    final_model_names = [Config.plot.fwd_model_name[m] + " (T1 only)" for m in models]
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
            # io.data.update(forward=model)
            io.data.update(forward=model + "_T1_only")
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
    df.to_pickle(Config.path.RESULTS / "forward_charm_t1_only.pickle")


def forward_collect_ref_match_fieldtrip():
    io = utils.GroupIO()
    io.data.update(stage="forward", space="fsaverage", suffix="fwd")
    if not Config.path.RESULTS.exists():
        Config.path.RESULTS.mkdir()

    force_fixed = True  # Config.inverse.FIXED not in (False, "auto")

    metrics = ("RDM", "lnMAG")
    n_metrics = len(metrics)

    ref = Config.forward.REFERENCE + "_match_fieldtrip"
    models = ["fieldtrip"]
    final_model_names = [Config.plot.fwd_model_name[m] + " (Matched)" for m in models]
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
    df.to_pickle(Config.path.RESULTS / "forward_match_fieldtrip.pickle")


# def forward_collect_distance_matrix():

#     io = utils.GroupIO()
#     io.data.update(stage="forward", space="fsaverage", suffix="fwd")
#     if not Config.path.RESULTS.exists():
#         Config.path.RESULTS.mkdir()

#     force_fixed = Config.inverse.FIXED not in (False, "auto")

#     metrics = ("RDM", "|lnMAG|")

#     ref = Config.forward.REFERENCE
#     models = [ref] + [i for i in Config.forward.MODELS if i != ref]
#     final_model_names = [Config.plot.model_name[m] for m in models]
#     n_models = len(models)

#     n_subjects = len(io.subjects)

#     first = next(iter(io.subjects))
#     io.subjects[first]
#     n_src = mne.read_forward_solution(
#         io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
#     )["nsource"]
#     n_chan = mne.read_forward_solution(
#         io.data.get_filename(subject=first, session=io.subjects[first], forward=ref,)
#     )["nchan"]

#     ori = ("normal",) if force_fixed else ("x", "y", "z")
#     n_ori = len(ori)
#     data = np.zeros((n_models, n_subjects, n_chan, n_src, n_ori))

#     print("Collecting forward solutions")
#     for i, (subject, session) in enumerate(io.subjects.items()):
#         io.data.update(subject=subject, session=session)
#         for j, model in enumerate(models):
#             io.data.update(forward=model)
#             fwd = mne.read_forward_solution(io.data.get_filename())
#             fwd = mne.convert_forward_solution(fwd, force_fixed=force_fixed)
#             data[j, i] = fwd["sol"]["data"].reshape(n_chan, n_src, n_ori)
#     data = data.reshape(n_models * n_subjects, n_chan, n_src, n_ori).squeeze()

#     print("Computing distance matrices")
#     abs_lnmag = lambda x, y: np.abs(lnmag(x, y))
#     metric_fun = dict(zip(metrics, (rdm, abs_lnmag)))
#     dist = {}
#     for metric in metrics:
#         fun = metric_fun[metric]
#         dist[metric] = np.zeros((n_models * n_subjects, n_models * n_subjects))
#         for i in range(n_models * n_subjects):
#             for j in range(n_models * n_subjects):
#                 if i == j:
#                     dist[metric][i, j] = 0
#                 elif i < j:
#                     dist[metric][i, j] = fun(data[i], data[j]).mean()
#                 elif i > j:
#                     dist[metric][i, j] = dist[metric][j, i]

#     print("Saving as DataFrames")
#     cols = pd.MultiIndex.from_product(
#         (final_model_names, io.subjects), names=("Forward", "Subject")
#     )
#     rows = pd.MultiIndex.from_product(
#         (metrics, final_model_names, io.subjects),
#         names=("Metric", "Forward", "Subject"),
#     )
#     df = pd.DataFrame(np.concatenate((list(dist.values()))), columns=cols, index=rows)
#     df.to_pickle(Config.path.RESULTS / "forward_distance_matrix.pickle")


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
        df.to_pickle(Config.path.RESULTS / f"{name}_{prefix}.pickle")


def inverse_compute_summary():
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


if __name__ == "__main__":
    forward_collect()
    forward_collect_ref_match_fieldtrip()
    forward_collect_distance_matrix()
    # inverse_collect()
    # inverse_compute_summary()
