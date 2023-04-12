from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.interpolate import interpn

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])

# Make DataFrame
# data = np.stack(
#     [np.stack([v for v in rdm.values()]), np.stack([v for v in mag.values()])]
# )
# data = data.reshape(6, -1).T
# metrics = ("RDM", "lnMAG")
# models = ("SimNIBS (Low Res.)", "SimNIBS (High Res.)", "FieldTrip Dipoli")
# orientation = ("x", "y", "z")
# source = np.arange(len(rdm["sim"]))
# index = pd.MultiIndex.from_product(
#     (source, orientation), names=["Source", "Orientation"]
# )
# column = pd.MultiIndex.from_product((metrics, models), names=["Metric", "Model"])
# df = pd.DataFrame(np.ascontiguousarray(data), index, column)

# df_distance = pd.DataFrame(dist, columns=["Distance"])


# df.to_pickle("/mrhome/jesperdn/INN_JESPER/projects/simval/head/bem_ref/metrics.pickle")
# df_distance.to_pickle(
#     "/mrhome/jesperdn/INN_JESPER/projects/simval/head/bem_ref/distance_to_inner_skull.pickle"
# )


def _symmetrize(arr):
    b = np.abs(arr).max()
    return (-b, b)


def get_plot_limits(df, percentiles=[0.5, 99.5]):
    # Limits for plotting
    limits = {}
    for metric in df.columns.unique("Metric"):
        p = np.percentile(df[metric].to_numpy(), percentiles, axis=0)
        pmin = p[0].min()
        pmax = p[1].max()
        limits[metric] = (min(0, pmin), max(0, pmax))
        if metric == "lnMAG":
            limits[metric] = _symmetrize(limits[metric])
    return limits


def compute_density(df_mean, distance, nbins=30):

    density = {}
    for model in df_mean.columns.unique("Model"):
        data = df_mean[model]
        H, xe, ye = np.histogram2d(distance, data, bins=nbins, density=True)
        gx = 0.5 * (xe[:-1] + xe[1:])
        gy = 0.5 * (ye[:-1] + ye[1:])
        sample_points = np.vstack([distance, data]).T
        z = interpn(
            (gx, gy),
            H,
            sample_points,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        z[np.isnan(z)] = 0
        z[z < 0] = 0

        density[model] = np.sqrt(z)  # square root of density

    density_max = np.stack([v for v in density.values()]).max()
    norm = Normalize(vmin=0, vmax=density_max)

    return density, norm


def plot_crf(df, limits):
    # Cumulative relative frequency
    n = len(df.index)
    n_metrics = len(df.columns.unique("Metric"))
    labels = [m.replace('>', '\\textgreater{}') for m in df.columns.unique("Model")]

    ix = np.arange(n)
    w = figure_sizing.fig_width['inch']['double']
    h = w/2 * 0.9
    figsize = (w, h)
    fig, axes = plt.subplots(1, n_metrics, sharey=True, figsize=figsize, constrained_layout=True)
    for metric, ax in zip(df.columns.unique("Metric"), axes):
        sval = np.sort(df[metric].to_numpy(), 0)
        ax.plot(sval, ix / n)
        ax.set_xlim(limits[metric])
        ax.set_xlabel(metric)
        ax.grid(alpha=0.25)
        # ax.xaxis.set_major_formatter("{x:.2f}")
        # ax.set_title('Cumulative Distribution Function')
        if metric == "RDM":
            ax.set_ylabel("Cumulative Relative Frequency")
            ax.legend(labels, loc="lower right")
    return fig


def plot_distance_density(df, df_distance):
    # Inner skull distance vs. density
    n_metrics = len(df.columns.unique("Metric"))
    n_models = len(df.columns.unique("Model"))

    distance = df_distance.to_numpy().squeeze()

    w = figure_sizing.fig_width['inch']['double']
    h = w * 2/3
    figsize = (w, h)
    fig, axes = plt.subplots(
        n_metrics, n_models, sharex=True, sharey="row", figsize=figsize, constrained_layout=True
    )
    for i, metric in enumerate(df.columns.unique("Metric")):
        df_mean = df[metric].groupby("Source").mean()  # mean over orientation
        # df_mean = df[metric].loc[pd.IndexSlice[:,'z'], :]
        dens, normalizer = compute_density(df_mean, distance)

        for j, model in enumerate(df.columns.unique("Model")):
            ax = axes[i, j]
            s = np.argsort(dens[model])
            ax.scatter(
                distance[s],
                df_mean[model][s],
                c=dens[model][s],
                marker=".",
                norm=normalizer,
                cmap="summer",
            )
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(model.replace('>', '\\textgreater{}'))
            elif i == j == 1:
                ax.set_xlabel("Distance to inner skull (mm)")
            if j == 0:
                ax.set_ylabel(metric)
    return fig

df = pd.read_pickle(
    "/mrhome/jesperdn/INN_JESPER/projects/simval/head/bem_ref/metrics.pickle"
)
df_distance = pd.read_pickle(
    "/mrhome/jesperdn/INN_JESPER/projects/simval/head/bem_ref/distance_to_inner_skull.pickle"
)
plotdir = Path("/mrhome/jesperdn/INN_JESPER/projects/simval/head/bem_ref/results")


limits = get_plot_limits(df)

fig = plot_crf(df, limits)
fig.savefig(plotdir / "bem_ref_head_crf")

fig = plot_distance_density(df, df_distance)
fig.savefig(plotdir / "bem_ref_distance_vs_metric.png")


# Make DataFrame
# data = np.stack(
#     [np.stack([v for v in rdm.values()]), np.stack([v for v in mag.values()])]
# )
# data = data.reshape(2 * 6, -1).T
# metrics = ("RDM", "lnMAG")
# models = (
#     "FEM 0.5 > FEM 0.5r",
#     "FEM 1.0 > FEM 1.0r",
#     "FEM 0.5 > FEM 1.0",
#     "BEM > FEM 0.5",
#     "BEM > FEM 1.0",
#     "BEM > FEM 1.0r",
# )
# orientation = ("x", "y", "z")
# source = np.arange(len(depth))
# index = pd.MultiIndex.from_product(
#     (source, orientation), names=["Source", "Orientation"]
# )
# column = pd.MultiIndex.from_product((metrics, models), names=["Metric", "Model"])
# df = pd.DataFrame(np.ascontiguousarray(data), index, column)

# df_distance = pd.DataFrame(depth, columns=["Distance"])

# df.to_pickle("/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/metrics.pickle")
# df_distance.to_pickle(
#     "/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/distance_to_inner_skull.pickle"
# )

plotdir = Path('/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/results')

df = pd.read_pickle("/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/metrics.pickle")
df_distance = pd.read_pickle(
    "/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/distance_to_inner_skull.pickle"
)

limits = get_plot_limits(df)

fig = plot_crf(df, limits)
fig.savefig(plotdir / "fem_ref_head_crf")

models = df.columns.unique("Model")

idx = pd.IndexSlice

fig = plot_distance_density(df.loc[:, idx[:, models[:3]]], df_distance)
fig.savefig(plotdir / "fem_ref_distance_vs_metric_fem.png")
fig = plot_distance_density(df.loc[:, idx[:, models[3:]]], df_distance)
fig.savefig(plotdir / "fem_ref_distance_vs_metric_bem.png")
