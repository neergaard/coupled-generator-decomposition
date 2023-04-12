import itertools
from pathlib import Path
import re
from string import ascii_uppercase

# import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import mne
import mne.channels._standard_montage_utils
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.sparse

# from pyvistaqt import BackgroundPlotter

# from scipy.stats import f_oneway, normaltest, ttest_rel
import scipy.stats

from projects.base.geometry import get_adjacency_matrix

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.base.colors import get_random_cmap
from projects.facerecognition.evaluation_viz_surf import (
    init_image_grid,
    get_scalarmappable,
    rescale_cbar,
)
from projects.mnieeg import utils
from projects.mnieeg.config import Config

from simnibs.utils.file_finder import get_montage_neighbors

# circular ...
from projects.mnieeg.commands.subject.optimize_montage import load_neighbors

from simnibs.simulation import eeg

# Set this environment variable (otherwise BackgroundPlotter fails beucase it
# cannot find the correct plugin file...)
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
#     Path(os.environ["CONDA_PREFIX"]) / "Library" / "plugins" / "platforms"
# )

style = "presentation" # publication, presentation

if style == "publication":
    mpl_styles.set_publication_styles()
    suffix = ""
elif style == "presentation":
    mpl_styles.set_dark_styles()
    suffix = "_dark"
    plt.rcParams["savefig.format"] = "svg"

pv.set_plot_theme("document")

# mne.set_log_level("warning")


def df_to_numpy(df):
    names = df.index.names
    axis = dict(zip(names, range(len(names))))
    return df.to_numpy().reshape((*df.index.levshape, df.shape[1])), axis


# CHANNEL EVALUATION
# =============================================================================


def create_info():
    # Create dummy Info for visualization
    fname = Config.path.RESOURCES / (Config.forward.MONTAGE_MNE + ".txt")
    montage = mne.channels._standard_montage_utils._read_theta_phi_in_degrees(
        fname, mne.defaults.HEAD_SIZE_DEFAULT, add_fiducials=True
    )
    info = mne.create_info(montage.ch_names, 100, ch_types="eeg")
    info.set_montage(montage)
    return info


def channel_plot():
    """

    - Create topomap of average euclidian distance between reference and other
    positions.
    - Create topomap of average absolute distance between reference and other
    positions along each axis.

    """
    # DataFrame
    #
    # access columns
    #   df['x']
    # access rows/groups
    #   df.loc['my_model', 'my_subject']
    # combined to access a specific column only
    #   df.loc(('my_model', 'my_subject'), 'x')
    output_dir = Config.path.RESULTS / "figure_channel"
    if not output_dir.exists():
        output_dir.mkdir()

    df = pd.read_pickle(Config.path.RESULTS / "channel.pickle")

    ref = Config.forward.REFERENCE
    ref = Config.plot.ch_model_name[ref]
    models = [m for m in df.index.unique("Forward") if m != ref]
    # diff = df.loc[models] - df.loc[ref]
    # diff = diff.sort_index(level="Forward").reorder_levels(df.index.names)

    # Get difference but keep order of labels
    axes = ["Subject", "Axis"]
    diff = (
        df.loc[models]
        .unstack(axes)
        .sub(df.loc[ref].unstack(axes))
        .stack(axes)
        .reindex(columns=df.columns)
    )

    absdiff = np.abs(diff)
    dist = np.sqrt(diff.pow(2).sum(level=["Forward", "Subject"]))

    info = create_info()

    # Normality test
    # k2, p = normaltest(dist, axis=0)

    # Perform an ANOVA for each channel for main effect of forward model
    # f, p = f_oneway(*dist.transpose(1, 0, 2), axis=0)
    # only the custom ones... no effect of registration method
    # f, p = f_oneway(*dist.transpose(1, 0, 2)[:3], axis=0)

    # t-test
    # t, p = ttest_rel(dist[:, 0], dist[:, -1], axis=0)

    # Boxplot
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.boxplot(dist)
    # fig.savefig(output_dir / "", **kw_savefig)

    fig = ch_plot_errors(dist, info)
    fig.savefig(output_dir / f"channel_topomap_errors_reduced{suffix}")

    fig = ch_plot_errors_xyz(absdiff, info)
    fig.savefig(output_dir / f"channel_topomap_errors_xyz{suffix}")

    fig = ch_plot_density(dist)
    fig.savefig(output_dir / f"channel_density_errors{suffix}")


def ch_plot_density(df, n_points=200):
    ref = Config.forward.REFERENCE
    models = [m for m in df.index.unique("Forward") if m != ref]

    points = np.linspace(0, np.ceil(df.to_numpy().max()), n_points)

    w = figure_sizing.fig_width["inch"]["single"]
    fig, ax = plt.subplots(figsize=(w, w / 1.33))
    for m in models:
        kernel = scipy.stats.gaussian_kde(df.loc[m].to_numpy().ravel())
        ax.plot(points, kernel(points))
    ax.grid(alpha=0.25)
    ax.legend(models)
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Probability Density")

    return fig


def choose_to_contour_levels(x, n=3):
    x = np.ceil(x / 10) * 10  # round up to closest 10
    return np.round(np.linspace(0, x, n + 2)[1:-1])


def ch_plot_errors(df, info, kwargs=None):
    """Topomap of errors."""

    if kwargs is None:
        kwargs = dict(vmin=0, cmap="viridis", show=False)

    models = df.index.unique("Forward")

    mu = df.mean(level="Forward")
    std = df.std(level="Forward")

    metrics = ("Mean Error", "Standard Deviation")
    df = pd.concat([mu, std], keys=metrics, names=["Metric"])
    metric_max = df.max(1).max(level=["Metric"])
    cs_levels = {
        k: choose_to_contour_levels(v) for k, v in metric_max.items()
    }  # contour levels

    w = figure_sizing.fig_width["inch"]["single"]
    h = w * len(metrics) / len(models) * 0.9
    fig, axes = plt.subplots(
        len(metrics), len(models), figsize=(w, h), constrained_layout=True
    )
    for i, (row, metric) in enumerate(zip(axes, metrics)):
        for j, (ax, model) in enumerate(zip(row, models)):
            im, cs = mne.viz.plot_topomap(
                df.loc[metric, model],
                info,
                axes=ax,
                contours=cs_levels[metric],
                vmax=metric_max[metric],
                **kwargs,
            )
            ax.clabel(cs, cs.levels, fontsize=6)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(metric)
        cbar = fig.colorbar(im, ax=row, shrink=0.75, pad=0.025)
        cbar.set_label("mm")  # , rotation=-90)

    return fig


def ch_plot_errors_xyz(df, info, kwargs=None):
    """Topomap of (absolute) errors along each axis."""

    if kwargs is None:
        kwargs = dict(vmin=0, cmap="viridis", show=False)

    models = df.index.unique("Forward")
    axis_labels = df.index.unique("Axis")

    df = df.mean(level=["Axis", "Forward"])
    model_max = df.max(1).max(level=["Forward"])  # max per forward model
    cs_levels = {
        k: choose_to_contour_levels(v) for k, v in model_max.items()
    }  # contour levels

    w = figure_sizing.fig_width["inch"]["double"]
    h = w * 3 / len(models) * 1.1
    fig, axes = plt.subplots(3, len(models), figsize=(w, h), constrained_layout=True)
    for j, (col, model) in enumerate(zip(axes.T, models)):
        for i, (ax, ax_label) in enumerate(zip(col, axis_labels)):
            im, cs = mne.viz.plot_topomap(
                df.loc[ax_label, model],
                info,
                axes=ax,
                contours=cs_levels[model],
                vmax=model_max[model],
                **kwargs,
            )
            ax.clabel(cs, cs.levels, fontsize=6)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(ax_label)
        # This is *not* pretty!
        # if j in (1, 3):
        cbar = fig.colorbar(im, ax=col, location="bottom", shrink=0.75, pad=0.025)
        cbar.set_label("mm")
        cbar.ax.locator_params(nbins=5)
        if j in (0, 2):
            cbar.remove()

    add_axes_description(fig)

    return fig


def add_axes_description(fig):
    pos = (0.05, 0.05)  # where to start arrows
    d_inches = 0.3  # how long arrows in inches

    w, h = fig.get_size_inches()
    dx = 1 / w * d_inches
    dy = 1 / h * d_inches

    # ha = horizontalalignment
    # va = verticalalignment
    arrow_kwargs = dict(head_width=dx / 5, head_length=dx / 5, color="black")
    fig.patches += [
        mpl.patches.FancyArrow(*pos, dx, 0, transform=fig.transFigure, **arrow_kwargs),
        mpl.patches.FancyArrow(*pos, 0, dy, transform=fig.transFigure, **arrow_kwargs),
    ]
    text_kwargs = dict(ha="center", va="center")
    fig.text(pos[0] + dx / 2, pos[1] - dy / 4, "x", **text_kwargs)
    fig.text(pos[0] - dx / 4, pos[1] + dy / 2, "y", **text_kwargs)


def channel_optimization_plot():
    output_dir = Config.path.RESULTS / "figure_optimization"
    if not output_dir.exists():
        output_dir.mkdir()

    # Illustration of measurement pairs
    measurement_pairs = ["43", "64", "70", "71"]

    # Neighborhood information
    fname = get_montage_neighbors("easycap_BC_TMS64_X21")
    neighbors, neighbor_ch_names = load_neighbors(fname)
    neighbors = scipy.sparse.triu(neighbors)
    ref_ix = neighbor_ch_names.index("reference")
    assert ref_ix == 0
    neighbors.data[neighbors.row == ref_ix] = 0
    neighbors.eliminate_zeros()
    neighbors = scipy.sparse.coo_matrix(neighbors.todense()[1:, 1:])

    #####
    # info = create_info()
    # ch_pos_2d = mne.channels.layout._find_topomap_coords(info, None)
    # tri = scipy.spatial.Delaunay(ch_pos_2d)
    # # scipy.spatial.delaunay_plot_2d(tri)
    # adj = get_adjacency_matrix(tri.simplices).tocoo()
    # neighbors = scipy.sparse.triu(adj)

    fig = plot_sensors(measurement_pairs, neighbors)
    fig.savefig(output_dir / f"layout_with_meas{suffix}")

    # levels = ["Distance Error", "Angular Error"]
    # constraints = [43, 64, 70, 71]

    dfr = pd.read_pickle(Config.path.RESULTS / "montage_ref.pickle")
    dfo = pd.read_pickle(Config.path.RESULTS / "montage_opt.pickle")

    # Average over repetitions
    cols = dfo.columns.names.difference(["Repetition"])
    dfo = dfo.mean(1, level=cols)

    diff = dfo - dfr

    # All error differences > 15 mm are subject 33!
    # diff[diff > 15].dropna(axis=1, how='all').columns

    names = ["Initial", "Optimized"]

    fig = optimization_density(dfr, dfo, names)
    fig.savefig(output_dir / f"density{suffix}")

    fig = optimization_violin_plot(dfr, dfo, names)
    fig.savefig(output_dir / f"violin_subject{suffix}")

    figsize = figure_sizing.get_figsize("double", 0.8, subplots=(1, 2))
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    vrange = (-20, 10)
    make_optimization_crf_noise(diff, vrange, axes[0])
    make_optimization_crf_subject(diff, vrange, axes[1])
    fig.savefig(output_dir / f"diff_dens{suffix}")

    # Quantiles of difference error
    # quantiles = [0.05, 0.5, 0.95]
    # vranges = [(-15, 15), (-8, 8), (-15, 15)]

    # # quantiles = np.linspace(0.01, 0.99, 10)
    # q = diff.groupby(levels, axis=1).quantile(quantiles)
    # q.columns.set_names("Quantile", level=2, inplace=True)
    # q.columns = q.columns.reorder_levels(
    #     ["Quantile", "Distance Error", "Angular Error"]
    # )
    # sorted_cols, _ = q.columns.sortlevel("Quantile")
    # q = q.reindex(sorted_cols, axis=1)

    # # find all values over...
    # # dfo[dfo>50].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # # Quantiles of difference error
    # for quan, (vmin, vmax) in zip(quantiles, vranges):
    #     fig = make_optimization_subplot_error(q[quan], vmin, vmax)
    #     fig.suptitle(f"{100*quan:2.0f}th percentile of Error Difference")
    #     fig.savefig(output_dir / f"diff_quantile_error_{100*quan:02.0f}th")

    alpha_level = 0.05
    # stat_levels = [(0, 0), (4, 8), (10, 20)]
    de = dfo.columns.unique("Distance Error")
    ae = dfo.columns.unique("Angular Error")
    stat_levels = itertools.product(de, ae)
    tstat, pval = make_permutation_t_test(diff, stat_levels)
    mask = {k: v <= alpha_level for k, v in pval.items()}

    fig = make_optimization_subplot_error(
        diff.mean(1, level=["Distance Error", "Angular Error"]),
        vmin=-10,
        vmax=10,
        mask=mask,
    )
    fig.suptitle(f"Mean Error Difference")
    fig.savefig(output_dir / f"diff_error_mean{suffix}")

    fig = make_optimization_subplot_error(
        diff.std(1, level=["Distance Error", "Angular Error"]), vmin=0, vmax=10
    )
    fig.suptitle(f"Standard Deviation of Error Difference")
    fig.savefig(output_dir / f"diff_error_std{suffix}")

    # Subject level difference error
    n = 3
    vrange = (-15, 15)
    sub_de = de[np.linspace(0, len(de) - 1, n, dtype=int)]
    sub_ae = ae[np.linspace(0, len(ae) - 1, n, dtype=int)]
    for sde, sae in zip(sub_de, sub_ae):
        fig = make_optimization_subplot_subject(
            diff, sde, sae, shape=(4, 8), vrange=vrange
        )

        fig.suptitle(f"Distance Error {sde:.2f} mm; Angular Error {sae:.2f}$^\circ$")
        fname = (
            f"diff_subject_error_{str(sde).replace('.','')}_{str(sae).replace('.','')}"
        )
        fig.savefig(output_dir / f"{fname}{suffix}")


def compute_tri_angles(tri):
    # tris = tri.simplices[np.argsort(tri.simplices.sum(1))]
    mesh = tri.points[tri.simplices]

    pairs = [[(0, 1), (0, 2)], [(1, 0), (1, 2)], [(2, 0), (2, 1)]]

    angles = []
    for p0, p1 in pairs:
        v0 = mesh[:, p0[1]] - mesh[:, p0[0]]
        v1 = mesh[:, p1[1]] - mesh[:, p1[0]]
        angles.append(
            np.rad2deg(
                np.arccos(
                    np.sum(v0 * v1, 1)
                    / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=-1))
                )
            )
        )
    return np.array(angles)


def plot_sensors(measurement_pairs, neighbors):

    info = create_info()
    info["bads"] = measurement_pairs  # for different colors

    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("single", 1))
    fig = mne.viz.utils.plot_sensors(
        info, show_names=measurement_pairs, axes=ax, pointsize=15, show=False,
    )

    ax = fig.axes[0]
    ch_pos = ax.collections[0].get_offsets().data
    for a, b in list(zip(neighbors.row, neighbors.col)):
        ax.add_line(plt.Line2D(*ch_pos[[a, b]].T, linewidth=1, zorder=0))
    return fig


def optimization_violin_plot(dfr, dfo, names):
    offset = 0.2
    width = 0.3

    n_subjects = len(dfo.columns.unique("Subject"))
    kwargs = dict(widths=width, showmeans=True, showextrema=False)
    positions = np.arange(n_subjects)

    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("double"))
    parts = [ax.violinplot(dfr, positions=positions - offset, **kwargs)]
    parts.append(
        ax.violinplot(
            dfo.stack(["Distance Error", "Angular Error"]),
            positions=np.arange(n_subjects) + offset,
            **kwargs,
        )
    )
    for part in parts:
        for p in part["bodies"]:
            p.set_alpha(0.5)

    ax.grid(alpha=0.25)
    ax.set_xlim(positions[0] - 0.99, positions[-1] + 0.99)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Error (mm)")
    ax.set_title("Density Estimate of Error Difference")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend([p["bodies"][0] for p in parts], names, loc="upper left")
    return fig


def optimization_density(dfr, dfo, names, n_points=200):

    data_r = dfr.to_numpy()
    data_o = dfo.to_numpy()
    points = np.linspace(0, max(data_r.max(), data_o.max()), n_points)

    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("single"))
    for d in (data_r, data_o):
        kernel = scipy.stats.gaussian_kde(d.ravel())
        ax.plot(points, kernel(points))
    ax.legend(names)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Probability Density")
    return fig


def make_optimization_crf_noise(df, vrange, ax, n_points=301):
    inset_pad = 0.02
    inset_size = 0.4
    inset_fontsize = 8

    points = np.linspace(*vrange, n_points)

    de = df.columns.unique("Distance Error")
    ae = df.columns.unique("Angular Error")
    nde = len(de)
    nae = len(ae)

    color_de = np.zeros((nde, 3))
    color_de[:, 0] = np.linspace(0, 1, nde)
    color_ae = np.zeros((nae, 3))
    color_ae[:, 1] = np.linspace(0, 1, nae)

    color_legend = color_de[:, None] + color_ae[None]

    for d, cd in zip(de, color_de):
        for a, ca in zip(ae, color_ae):
            this_data = df.loc[:, pd.IndexSlice[:, d, a]]
            # crf = np.linspace(0, 1, this_data.size)

            kernel = scipy.stats.gaussian_kde(this_data.to_numpy().ravel())
            # ax.plot(points, kernel(points))

            ax.plot(
                points,
                kernel(points),
                # np.sort(this_data.to_numpy().ravel()),
                # crf,
                color=tuple(cd + ca),
                linewidth=0.5,
                alpha=0.5,
            )

    # csum = np.abs(np.sort(df.to_numpy().ravel())).cumsum()
    # ax2 = ax.twinx()
    # ax2.plot(np.sort(df.to_numpy().ravel()), csum / csum[-1], linewidth=0.5)
    # ax2.set_ylabel("Cumulative Relative Improvement (mm)")

    ax.grid(alpha=0.25)
    ax.set_xlim(vrange)
    ax.set_xlabel("Error Difference (mm)")
    ax.set_ylabel("Probability")
    ax.set_title("Noise Level")

    # Overall
    # crf = np.linspace(0, 1, diff.size)
    # ax.plot(np.sort(diff.to_numpy().ravel()), crf)

    axin = ax.inset_axes(
        [3 * inset_pad, 1 - inset_size - inset_pad, inset_size, inset_size]
    )
    axin.imshow(color_legend, origin="lower")
    axin.set_xlabel("Distance Error", fontsize=inset_fontsize)
    axin.set_ylabel("Angular Error", fontsize=inset_fontsize)
    # tickpos = np.round(np.linspace(0, nde - 1, 3)).astype(int)
    tickpos = np.arange(nde)
    axin.set_xticks(tickpos, de[tickpos], fontsize=inset_fontsize)
    # axin.xaxis.tick_top()
    # axin.xaxis.set_label_position("top")
    # tickpos = np.round(np.linspace(0, nae - 1, 3)).astype(int)
    tickpos = np.arange(nae)
    # axin.yaxis.tick_right()
    # axin.yaxis.set_label_position("right")
    axin.set_yticks(tickpos, ae[tickpos], fontsize=inset_fontsize)
    # axin.yaxis.tick_right()
    # axin.yaxis.set_label_position("right")


def make_optimization_crf_subject(df, vrange, ax, n_points=301):
    points = np.linspace(*vrange, n_points)

    subjects = df.columns.unique("Subject")
    subjects = subjects[np.argsort(df.mean(1, level="Subject").mean())]

    # cmap = get_random_cmap(len(subjects))
    cmap = plt.get_cmap("viridis")
    for subject, color in zip(subjects, cmap(np.linspace(0, 1, len(subjects)))):
        this_data = df[subject]
        # crf = np.linspace(0, 1, this_data.size)
        kernel = scipy.stats.gaussian_kde(this_data.to_numpy().ravel())
        ax.plot(
            points, kernel(points), color=color,
        )
        # ax.plot(
        #     np.sort(this_data.to_numpy().ravel()), crf, color=color,
        # )
    ax.grid(alpha=0.25)
    ax.set_xlim(vrange)
    ax.set_xlabel("Error Difference (mm)")
    ax.set_title("Subjects")


def make_permutation_t_test(df, levels, axis=0):
    tstat, pval = {}, {}
    if levels:
        for k in levels:
            data = -df.loc[:, pd.IndexSlice[:, k[0], k[1]]].to_numpy()
            if axis == 0:
                data = data.T
            tstat[k], pval[k], _ = mne.stats.permutation_t_test(data, verbose=False)
    return tstat, pval


def make_optimization_subplot_error(df, vmin, vmax, mask=None):

    # n = 10
    cmap = "RdBu_r" if vmin < 0 else "Reds"
    kwargs = dict(
        cmap=cmap,
        contours=0,
        show=False,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            linewidth=0,
            markersize=2.5,
        ),
    )
    imgrid_kwargs = dict(
        cbar_mode="single", cbar_pad=0.1, cbar_size="2.5%", cbar_location="bottom"
    )

    mask = mask or {}

    de = df.columns.unique("Distance Error")
    ae = df.columns.unique("Angular Error")
    sub_de = de
    sub_ae = ae
    nsub_de = len(sub_de)
    nsub_ae = len(sub_ae)
    # sub_de = de[np.linspace(0, len(de) - 1, n, dtype=int)]
    # sub_ae = ae[np.linspace(0, len(ae) - 1, n, dtype=int)]

    # fig, axes = plt.subplots(
    #     nsub_de,
    #     nsub_ae,
    #     figsize=(2 * nsub_de - 2, 2 * nsub_ae),
    #     constrained_layout=True,
    # )
    info = create_info()
    fig, grid = init_image_grid(
        (nsub_de, nsub_ae), 1, "double", imgrid_kw=imgrid_kwargs
    )
    for i, (row_ax, d) in enumerate(zip(grid.axes_row, sub_de)):
        for j, (ax, a) in enumerate(zip(row_ax, sub_ae)):
            m = mask[d, a] if (d, a) in mask else None
            mne.viz.plot_topomap(
                df[d, a], info, axes=ax, vmin=vmin, vmax=vmax, mask=m, **kwargs,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(f"{a}$^\circ$")
            if j == 0:
                ax.set_ylabel(f"{d} mm")
    clim = vmin, vmax
    rescale_cbar(grid, 2, 3, location=imgrid_kwargs["cbar_location"])
    cbar = fig.colorbar(
        get_scalarmappable(cmap, clim), cax=grid.cbar_axes[0], orientation="horizontal"
    )
    cbar.set_label("mm")
    return fig


def make_optimization_subplot_subject(df, de, ae, shape, vrange=None):
    cmap = "RdBu_r"
    kwargs = dict(cmap=cmap, contours=0, show=False)
    imgrid_kwargs = dict(
        cbar_mode="single", cbar_pad=0.1, cbar_size="2.5%", cbar_location="bottom"
    )

    tstat, pval = make_permutation_t_test(df, [(de, ae)], axis=1)
    alpha_level = 0.05
    mask = pval[de, ae] <= alpha_level
    tstat = tstat[de, ae]

    info = create_info()

    if vrange:
        vmin, vmax = vrange
    else:
        vmin, vmax = np.round(
            np.percentile(df.loc[:, pd.IndexSlice[:, de, ae]].to_numpy(), [5, 95])
        )
        vmax = max(abs(vmin), vmax)
        vmin = -vmax

    fig, grid = init_image_grid(shape, 1, "double", imgrid_kw=imgrid_kwargs)
    for ax, sub, m, t in zip(grid.axes_all, df.columns.unique("Subject"), mask, tstat):
        mne.viz.plot_topomap(
            df[sub, de, ae], info, axes=ax, vmin=vmin, vmax=vmax, **kwargs,
        )
        if m:
            color = "blue" if t > 0 else "red"
            ax.scatter(-0.085, 0.085, s=10, marker="*", color=color)

    clim = vmin, vmax
    rescale_cbar(grid, 3, 4, location=imgrid_kwargs["cbar_location"])
    cbar = fig.colorbar(
        get_scalarmappable(cmap, clim), cax=grid.cbar_axes[0], orientation="horizontal"
    )
    cbar.set_label("mm")

    # fig, axes = plt.subplots(*shape, figsize=(9, 4), constrained_layout=True)
    # sm = plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), plt.get_cmap(cmap))
    # cbar = fig.colorbar(sm, ax=axes[1:-1], shrink=0.75, pad=0.02)
    # cbar.set_label("mm")
    return fig


def inverse_plot():

    """
        Factor              Levels
    ------              ------
    forward_models      4
    snr                 3
    inverse_models      3

    Outcomes (y)
    n(res_functions)    = 2
    n(res_metrics)      = 3

    y ~ model * im * snr
    y(psf, peak_err) ~ model * im * C(snr)
    e.g., 4 * 2 * 3 * 3 * 2 = 144

    All resolution metrics metrics

    13 gb = 35 subjects * 144 metrics * 2*160e3 sources
            * 8 bytes/64-bit float * 1e-9 gb/bytes

    13 / 6 = 2.167

    35 subjects
    4 x 3 x 2 (FWD x SNR x INV)
    320,000 sources (restrict to one hemisphere? symmetry...)

    ANALYSIS
    --------
    3-way ANOVA per source
    - main effects
    - interactions?

    (1)
    For each res_fun/res_metric combination (six in total), let

        X = (n_subjects, n_vertices, FWD x SNR x INV) or whatever dimensions

    be the data matrix (on which permutation will be performed).

    (2)
    Define stat_fun as the 3-way ANOVA fun for a particular effect (e.g., main
    effect of FWD) which returns a 1D array (!), i.e., only a single effect can
    be tested at a time.
    - Main effect of FWD[, SNR, INV]
    - Interaction effect of FWD with SNR/INV?

    (3)
    Feed these to e.g. mne.stats.permutation_cluster_test as a list of
    conditions each having (n_observations/n_subjects, n_vertices) where the
    LAST dimension corresponds to the adjacency parameter.

    1-tailed as this is an F-statistic?

    -> For each (res_fun, res_metric) pair and each effect of interest (e.g.,
    main effect of FWD and interaction between FWD and SNR) we get a
    significance map of the F statistic.

    """
    output_dir = Config.path.RESULTS / "figure_inverse"
    if not output_dir.exists():
        output_dir.mkdir()

    # df = pd.read_pickle(r"C:\Users\jdue\Desktop\results\inverse.pickle")
    df = pd.read_pickle(Config.path.RESULTS / "inverse.pickle")

    dfs = get_summary_statistics(
        df, ["Forward", "Inverse", "SNR", "Resolution Function", "Resolution Metric"]
    )

    fun_to_plot = ["psf", "ctf"]
    met_to_plot = ["peak_err", "sd_ext"]
    for fun, met in itertools.product(fun_to_plot, met_to_plot):
        fig = inv_plot_density(df, fun, met)
        fig.savefig(output_dir / f"inverse_density_{fun}_{met}")

    levels_of_interest = ["Forward", "SNR"]
    # Ensure same order as in df
    level_names = [i for i in df.index.names if i in levels_of_interest]
    level_vals = [df.index.unique(i).values.tolist() for i in level_names]
    level_n = [len(v) for v in level_vals]

    include = set((level_names + ["Subject"]))
    data_shape = tuple(
        j for i, j in zip(df.index.names, df.index.levshape) if i in include
    )
    data = df.loc[:, :, "dSPM", :, "psf", "sd_ext"].to_numpy().reshape(*data_shape, -1)

    # forward x inverse x snr where snr changes the fastest, ...
    conditions = list(itertools.product(*level_vals))

    # f_mway_rm only accepts A, B, C, ... effect names
    effect2ascii = {n.lower(): ascii_uppercase[i] for i, n in enumerate(level_names)}
    effect2ascii_fun = lambda x: effect2ascii[x.group(0)]
    pattern = "forward|inverse|snr"
    # ascii_effects = [re.sub(pattern, repl_fun, effect) for effect in Config.stats.EFFECTS_OF_INTEREST]
    ascii_effects = [
        re.sub(pattern, effect2ascii_fun, effect)
        for effect in ["forward", "snr", "forward:snr"]
    ]

    # Forward x SNR effect: larger difference as SNR increases (less
    # regularization and thus more weight is given to the elements of the gain
    # matrix compared to the noise covariance [which is identical across all
    # conditions])

    # Statistical function
    fvalues, pvalues = mne.stats.f_mway_rm(
        data.reshape(data.shape[0], -1, data.shape[-1]),
        factor_levels=level_n,
        effects=ascii_effects,
        correction=True,
    )

    # ...
    p_cutoff = 0.001  # 0.05
    is_significant = pval <= p_cutoff

    # surf = get_central_fsaverage()
    titles = Config.stats.EFFECTS_OF_INTEREST
    inv_plot_statistic(surf, fvalues, pvalues, titles, 0.001)

    fwd_map = {k: i for i, k in enumerate(df.index.unique("Forward"))}
    snr_map = {k: i for i, k in enumerate(df.index.unique("SNR"))}

    # {model} - digitized
    snr = 25
    t_tests = (
        ("digitized", "custom_nonlin", snr),
        ("digitized", "manufacturer_affine_lm", snr),
        ("digitized", "template_nonlin", snr),
    )

    tvalues, pvalues, titles = [], [], []
    for m_ref, m, s in t_tests:
        res = scipy.stats.ttest_1samp(
            np.diff(
                data[:, (fwd_map[m_ref], fwd_map[m]), snr_map[s]], axis=1
            ).squeeze(),
            0,
            axis=0,
        )
        tvalues.append(res.statistic)
        pvalues.append(res.pvalue)
        titles.append(f"{m} - {m_ref} (SNR = {s})")

    inv_plot_statistic(surf, tvalues, pvalues, titles, 0.001)

    mne.stats.permutation_t_test()

    i = 12682  # fval[2].argmax()
    c = 0
    for m in range(4):
        plt.boxplot(data[:, m, :, i], positions=np.arange(c, c + 3))
        c += 3
    # , notch=True, conf_intervals=np.column_stack((sem[:,0], sem[:,0])));
    print(fval[:, i])
    print(pval[:, i])

    i = 100
    print("probability that the null hypothesis is TRUE")
    w = 0
    for m, (j, k) in enumerate(((3 + w, 0 + w), (6 + w, 0 + w), (9 + w, 0 + w))):
        q = data[:, j, i] - data[:, k, i]
        print(f"{q.mean()} ({q.std()/ np.sqrt(len(q)-1)})")
        print(scipy.stats.ttest_1samp(q, 0, axis=0))
        plt.scatter(np.ones(32) * m, data[:, j, i] - data[:, k, i])

    io = utils.GroupIO()
    n_subjects = len(io.subjects)

    fsavg = eeg.FsAverage(int(Config.inverse.SOURCE_SPACE[-1]))
    src = fsavg.get_central_surface()
    adj = scipy.sparse.block_diag(
        [get_adjacency_matrix(src[hemi]["tris"]) for hemi in src]
    )

    # data for clustering: list (of conditions) of (repetitions/subjects, observations/sources)
    # data for ANOVA: (repetitions/subjects, conditions, observations/sources)

    # Must return a 1D array
    def stat_fun(*args, factor_levels, effect):
        return mne.stats.f_mway_rm(
            np.swapaxes(args, 1, 0), factor_levels, effect, return_pvals=False
        )[0]

    # kwargs for permutation_cluster_test
    kw_permutation = dict(
        threshold=50,  # Config.stats.TFCE_THRESHOLD,
        n_permutations=256,
        tail=1,  # f-test, so tail > 0
        adjacency=adj,  # fsaverage
        check_disjoint=True,
        n_jobs=1,
    )

    (*cluster_data,) = data.swapaxes(1, 0)  # unpack first axis to list
    for effect in ascii_effects:
        stat_fun_ = functools.partial(stat_fun, factor_levels=level_n, effect=effect)
        f_vals, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
            cluster_data, stat_fun=stat_fun_, **kw_permutation, verbose=True
        )

    arr = np.zeros_like(f_vals)
    for c, p in zip(clusters, cluster_pv):
        if p <= 0.05:
            arr[c[0]] = f_vals[c[0]]

    p = pv.Plotter(notebook=False)
    p.add_text(effect)
    p.add_mesh(mb["lh"].copy(), scalars=arr[:10242])
    # p.add_mesh(mb['rh'].copy(), scalars=arr[10242:])
    p.view_xy(negative=True)
    p.show()


def inv_plot_statistic(surf, statval, pval, titles, p_thres=0.05):
    """Plot statistic and corresponding p-values on a surface."""

    statval = np.asarray(statval)
    pval = np.asarray(pval)

    if isinstance(titles, str):
        titles = [titles]
    n_titles = len(titles)

    zoom_factor = 1.3

    alpha = 5
    s_lim = np.percentile(statval, [alpha, 100 - alpha])
    s_lim_sign = np.sign(s_lim)
    use_below_color = False
    if all(s_lim_sign == -1):
        s_lim = np.array([np.floor(s_lim[0]), 0])
    elif all(s_lim_sign == 1):
        s_lim = np.array([0, np.ceil(s_lim[1])])
    else:
        s_lim[1] = np.abs(s_lim).max()
        s_lim[0] = -s_lim[1]
        use_below_color = True
    p_lim = [1e-10, p_thres]

    inds = np.zeros((len(surf), 2), dtype=int)
    c = 0
    for i, hemi in enumerate(surf):
        inds[i] = c, (c := c + hemi.n_points)

    window_size = [1500, 1100]
    p = pv.Plotter(shape=(2, n_titles), notebook=False, off_screen=True)
    # kw = dict(below_color="blue", above_color="red")
    for i, (name, array, limits) in enumerate(
        zip(
            ("Statistic", "p"),
            (np.atleast_2d(statval), np.atleast_2d(pval)),
            (s_lim, p_lim),
        )
    ):
        cbar_kw = dict(clim=limits)
        cbar_kw["above_color"] = "red"
        if name != "Statistic" or use_below_color:
            cbar_kw["below_color"] = "blue"

        log_scale = False  # name == "p"
        for j, (array_row, title) in enumerate(zip(array, titles)):
            p.subplot(i, j)
            if j == 0:
                cbar_kw["scalar_bar_args"] = {"title": " ".join((name, title))}
            for ind, hemi in zip(inds, surf):
                # Copy needed!!!
                p.add_mesh(
                    hemi.copy(),
                    scalars=array_row[slice(*ind)],
                    log_scale=log_scale,
                    **cbar_kw,
                )
                # print(cbar_kw)
            # p.view_xy(negative=True)
            p.camera.zoom(zoom_factor)
            if i == 0:
                p.add_text(title, "upper_edge")
            if j == 0:
                p.add_text(name, "left_edge")
                # del cbar_kw['scalar_bar_args']
    p.link_views()
    p.show()
    return p


def aggregate_data_on_lobes(df):
    """

    We compute the mean of the mean over subjects, however, ...

    The histogram for each region will sometimes be Gaussian-like, however,
    often there might be heavy tails or it may even be bimodal (this seems
    particularly true for `MNE`)

    So some differences might get washed out.

    """

    exclude_lobes = {"unknown", "corpuscallosum"}
    hemispheres = ("lh", "rh")
    names2abbr = dict(
        unknown="UNK",
        frontal="FRO",
        corpuscallosum="COR",
        parahippocampalgyrus="PHG",
        cingulate="CIN",
        occipital="OCC",
        temporal="TEM",
        parietal="PAR",
        insula="INS",
    )

    default_hemi = hemispheres[0]
    lobes_dir = Config.path.RESOURCES / Config.inverse.SOURCE_SPACE

    keys = ("labels", "ctab", "names")
    lobes_annot = {
        h: {
            k: v
            for k, v in zip(
                keys,
                nib.freesurfer.read_annot(lobes_dir / f"{h}.lobesStrictPHCG.annot"),
            )
        }
        for h in hemispheres
    }
    # names and ctab are the same across hemispheres
    lobes = dict(
        names=[name.decode() for name in lobes_annot[default_hemi]["names"]],
        ctab=lobes_annot[default_hemi]["ctab"],
        labels=np.concatenate([lobes_annot[hemi]["labels"] for hemi in hemispheres]),
    )
    lobes["in_use"] = [name not in exclude_lobes for name in lobes["names"]]
    lobes["names_abbr"] = [names2abbr[name] for name in lobes["names"]]

    df_lobes = pd.DataFrame(
        index=df.index,
        columns=list(itertools.compress(lobes["names"], lobes["in_use"])),
        dtype=float,
    )
    for i, (name, in_use) in enumerate(zip(lobes["names"], lobes["in_use"])):
        if in_use:
            df_lobes[name] = df.loc[:, lobes["labels"] == i].mean(1)
    return df_lobes


# exclude_lobes = {"unknown", "corpuscallosum"}

# df = dfs.copy()
# df.columns = np.array(lobes["names"])[lobes["labels"]]
# df = df.loc["mean"]
# df = df[[i for i in df.columns.unique() if i not in exclude_lobes]]


def grouped_bar_chart(df):

    # width = 0.25  # the width of the bars

    plot_difference = False

    inter_group_distance = 0.2

    n_kde_points = 256
    cmaps = ["Blues", "Greens", "Purples", "Reds"]

    region_abbrev_mapper = dict(
        frontal="FRO",
        parietal="PAR",
        occipital="OCC",
        temporal="TEM",
        parahippocampalgyrus="PHG",
        insula="INS",
        cingulate="CIN",
    )
    regions = df.columns.unique()
    n_regions = len(regions)
    region_abbrev = [region_abbrev_mapper[region] for region in regions]
    # region_abbrev = [i[:3].upper() for i in regions]
    region_positions = np.arange(n_regions)

    inverse = df.index.unique("Inverse")
    snr = df.index.unique("SNR")
    metrics = ("peak_err", "sd_ext")
    n_inverse = len(inverse)
    n_metrics = len(metrics)
    n_snr = len(snr)

    forward = list(df.index.unique("Forward"))
    n_forward = len(forward)
    if plot_difference:
        forward_ref_index = forward.index(Config.forward.REFERENCE)
        forward_ref = forward.pop(forward_ref_index)
        n_forward = len(forward)
        index = pd.MultiIndex.from_tuples(
            [i for i in df.index if forward_ref not in i], names=df.index.names
        )
        data = (
            df.loc[forward].to_numpy().reshape(len(forward), -1, df.shape[1])
            - df.loc[forward_ref].to_numpy()
        )
        df = pd.DataFrame(
            data.reshape(-1, df.shape[1]), index=index, columns=df.columns
        )
        cmaps.pop(forward_ref_index)
    cmaps = cmaps[:n_forward]

    # We change the order of SNR and forward such that forward models appear
    # next to each other (in the dataframe, SNR is the fastest changing)!
    subgroups = list(itertools.product(snr, forward))
    n_subgroups = len(subgroups)
    width = (1 - inter_group_distance) / n_subgroups
    subgroup_positions = -0.5 * width * (n_subgroups - 1) + width * np.arange(
        n_subgroups
    )

    # color = plt.cm.viridis(np.linspace(0, 1, n_forward)) # color cycle from cmap
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = default_colors[:n_forward] * n_snr

    cmaps *= n_snr
    # height = np.round(df.to_numpy().max(), 1)
    vmin, vmax = np.round(np.percentile(df.to_numpy(), [0.1, 99.9]), 1)
    kde_points = np.linspace(vmin, vmax, n_kde_points)

    # barplot_kw = dict(
    #     region_positions=region_positions,
    #     subgroup_positions=subgroup_positions,
    #     width=width,
    #     color=colors,
    # )
    # barplot_data_kw = {}  # dict(n_forward=n_forward, n_snr=n_snr)
    # if plot_difference:
    #    barplot_data_kw.update(forward=forward, forward_ref=forward_ref)

    positions = region_positions[:, None] + subgroup_positions[None]

    handles = []
    for cmap in cmaps[:n_forward]:
        cm = plt.cm.get_cmap(cmap)
        handles.append(mpl.lines.Line2D([0, 0], [1, 1], color=cm(np.arange(cm.N))[-50]))

    df_dens, df_mean = _compute_densities(df, kde_points, inverse, metrics, regions)
    # Normalize all densities to same scale...
    # density_max = np.percentile(df_dens.to_numpy(), 99)
    # df_dens = np.clip(df_dens, 0, density_max)
    # df_dens /= density_max

    fig, axes = plt.subplots(
        n_inverse + 1,
        n_metrics,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        figsize=(10, 16),
    )
    irow = 0
    for fun in df.index.unique("Resolution Function"):
        for inv in inverse:
            # if fun == "psf" and inv == "dSPM":
            #    break

            if fun == "ctf" and inv != Config.inverse.METHODS[0]:
                continue
            row_axes = axes[irow]

            for icol, (metric, ax) in enumerate(zip(metrics, row_axes)):
                # data = _get_barplot_data(df, inv, fun, metric, **barplot_data_kw)

                all_bars = [
                    _add_uniform_bars_to_axis(ax, pos, vmin, vmax, width)
                    for pos in positions
                ]

                for region, bars in zip(regions, all_bars):
                    # data = _get_data(df, region, inv, fun, metric, **barplot_data_kw)
                    # _add_boxplot_to_axis(ax, data.T, pos, width, color)
                    # _add_barplot_to_axis(ax, data, **barplot_kw)

                    density = df_dens.loc[(inv, fun, metric), region].to_numpy()
                    mean = df_mean.loc[(inv, fun, metric), region].to_numpy()
                    _add_density_gradients_to_bars(ax, bars, cmaps, density, mean)

                    # _add_scatterplot_to_axis(ax, data, pos, colors)
                ax.grid(alpha=0.5)

                if irow == 0:
                    ax.set_title(metric)
                if icol == 0:
                    ax.set_ylabel(f"{fun} {inv}")
                    # ax.set_ylabel("Error (cm)")
                if irow == 0 and icol == 1:
                    ax.legend(handles, forward)
                    # ax.legend(cmaps[:n_forward])
            irow += 1
    ax.set_xticks(region_positions)
    ax.set_xticklabels(region_abbrev)

    return fig


def _compute_densities(df, points, inverse, metrics, regions):
    dfds = {}
    dfms = {}
    for fun in df.index.unique("Resolution Function"):
        for inv in inverse:
            if fun == "ctf" and inv != Config.inverse.METHODS[0]:
                continue
            for metric in metrics:
                dfd, dfm = [], []
                for region in regions:
                    dfd_, dfm_ = _get_df_density_and_mean(
                        df, points, region, inv, fun, metric
                    )
                    dfd.append(dfd_)
                    dfm.append(dfm_)
                dfds[(inv, fun, metric)] = pd.concat(dfd, "columns")
                dfms[(inv, fun, metric)] = pd.concat(dfm, "columns")
    names = ["Inverse", "Resolution Function", "Resolution Metric"]
    return pd.concat(dfds, names=names), pd.concat(dfms, names=names)


def _add_uniform_bars_to_axis(ax, pos, vmin, vmax, width):
    return ax.bar(pos, pos.size * [vmax - vmin], width, vmin)


def _add_density_gradients_to_bars(ax, bars, cmaps, densities, means):

    line_kw = dict(linewidth=1, color="black", zorder=2)

    densities = densities[..., None]

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for bar, density, mean, cmap in zip(bars, densities, means, cmaps):

        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()

        density_max = density.max()
        if density_max > 0:
            density /= density_max

        ax.imshow(
            density,
            cmap=cmap,
            aspect="auto",
            alpha=np.abs(density),
            origin="lower",
            extent=[x, x + w, y, y + h],
            zorder=0,
        )
        ax.add_line(mpl.lines.Line2D([x, x + w], [mean, mean], **line_kw))

        # ax.scatter([x+w/2]*len(this_data), this_data, marker='.', alpha=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# cmaps = []
# for color in colors:
#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         "mymap", list(zip([0, 1], [color, color]))
#     )
#     cmaps_with_alpha = cmap(np.arange(cmap.N))
#     cmaps_with_alpha[:, -1] = np.linspace(0, 1, cmap.N)
#     cmaps.append(mpl.colors.ListedColormap(cmaps_with_alpha))


def _get_df_density_and_mean(df, points, region, inv, fun, metric):
    idx = pd.IndexSlice
    n_points = len(points)
    n_snr = df.index.levshape[df.index.names.index("SNR")]

    forward = df.index.unique("Forward")
    n_forward = len(forward)
    data = (
        df.loc[idx[:, inv, :, fun, metric], region]
        .to_numpy()
        .reshape(n_forward, n_snr, -1)
    )
    data = data.swapaxes(0, 1).reshape(n_snr * n_forward, -1)
    densities = np.zeros((len(data), n_points))
    for i, this_data in enumerate(data):
        try:
            kernel = scipy.stats.gaussian_kde(this_data)
            densities[i] = kernel(points)
        except np.linalg.LinAlgError:
            pass  # already zeros

    index = pd.MultiIndex.from_product((df.index.unique("SNR"), forward))
    return (
        pd.DataFrame(
            densities,
            index=index,
            columns=pd.MultiIndex.from_product([(region,), np.arange(n_points)]),
            dtype=float,
        ),
        pd.DataFrame(data.mean(1), index=index, columns=(region,)),
    )


# def _get_data(df, region, inv, fun, metric, forward=None, forward_ref=None):
#     idx = pd.IndexSlice
#     n_snr = df.index.levshape[df.index.names.index("SNR")]
#     if forward is None and forward_ref is None:
#         n_forward = df.index.levshape[df.index.names.index("Forward")]
#         data = (
#             df.loc[idx[:, inv, :, fun, metric], region]
#             .to_numpy()
#             .reshape(n_forward, n_snr, -1)
#         )
#     else:
#         n_forward = len(forward)
#         data = (
#             df.loc[idx[forward, inv, :, fun, metric], region]
#             .to_numpy()
#             .reshape(n_forward, n_snr, -1)
#             - df.loc[idx[forward_ref, inv, :, fun, metric], region].to_numpy()
#         )
#     data = data.swapaxes(0, 1).reshape(n_snr * n_forward, -1)

#     kernel = scipy.stats.gaussian_kde(this_data)
#     density = kernel(points)
#     return pd.DataFrame(
#         data, index=pd.MultiIndex.from_product([df.index.unique("SNR"), forward]), cols=
#     )


def _get_barplot_data(
    df, inv, fun, metric, n_forward, n_snr, n_regions, forward=None, forward_ref=None
):
    idx = pd.IndexSlice
    if forward is None and forward_ref is None:
        data = (
            df.loc[idx[:, inv, :, fun, metric]]
            .to_numpy()
            .reshape(n_forward, n_snr, n_regions)
        )
    else:
        data = (
            df.loc[idx[forward, inv, :, fun, metric]]
            .to_numpy()
            .reshape(n_forward, n_snr, n_regions)
            - df.loc[idx[forward_ref, inv, :, fun, metric]].to_numpy()
        )
    return data.swapaxes(0, 1).reshape(n_snr * n_forward, n_regions)


def _add_barplot_to_axis(ax, data, region_positions, subgroup_positions, width, color):
    for subgroup_pos, d, c in zip(subgroup_positions, data, color):
        ax.bar(region_positions + subgroup_pos, d, width, color=c)


def _add_boxplot_to_axis(ax, data, positions, width, color):
    box = ax.boxplot(data, positions=positions, widths=width, patch_artist=True)
    for k, v in box.items():
        if k == "boxes":
            for patch, c in zip(v, color):
                patch.set_facecolor(c)
                patch.set_edgecolor(c)


def _add_scatterplot_to_axis(ax, data, positions, colors):
    x = np.broadcast_to(positions[:, None], data.shape).ravel()
    y = data.ravel()
    c = np.broadcast_to(colors[:, None], data.shape).ravel()
    ax.scatter(x, y, c=c, alpha=0.1, marker=".")


def _add_scatterplot_to_axis(
    ax, data, region_positions, subgroup_positions, width, color
):
    for subgroup_pos, d, c in zip(subgroup_positions, data, color):
        box = ax.scatter(region_positions + subgroup_pos, d, widths=width, color=c)
        for v in box.values():
            v["color"] = c


def get_cbar_limits(array, low, high, decimals=1, symmetrize=False):
    limits = np.round(np.percentile(array, [low, high]), decimals)
    if symmetrize:
        limits = np.array([-1, 1]) * np.abs(limits).max()
    return limits
