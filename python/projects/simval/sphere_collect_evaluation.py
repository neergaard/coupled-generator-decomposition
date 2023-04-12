from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.base.forward_evaluation import lnmag, rdm

from projects.simval.config import Config
from projects.simval.sphere_utils import idx_to_density

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])

def collect_evaluation():

    metrics = ("RDM", "lnMAG")

    densities, dirs = idx_to_density()
    radii = np.load(Config.path.SPHERE / "src_radii.npy")
    sol = np.load(Config.path.SPHERE / "analytical_solutions" / "solution_3_layer.npy")

    index = pd.MultiIndex.from_product(
        (radii, np.arange(Config.sphere.n_rays), np.arange(3)),
        names=("radius", "ray", "orientation"),
    )
    columns = pd.MultiIndex.from_product(
        (metrics, densities), names=("Metric", "Density")
    )
    df = pd.DataFrame(index=index, columns=columns)
    for d, dens in zip(dirs, densities):
        sim = np.load(d / "fem_sphere_3_layer" / "gain.npy")
        df["RDM", dens] = rdm(sim, sol).ravel()
        df["lnMAG", dens] = lnmag(sim, sol).ravel()

    return df

# def _symmetrize(ax, axis=1):
#     if axis == 0:
#         b = np.abs(ax.get_xlim()).max()
#         ax.set_xlim(-b, b)
#     if axis == 1:
#         b = np.abs(ax.get_ylim()).max()
#         ax.set_ylim(-b, b)


def _symmetrize(arr):
    b = np.abs(arr).max()
    return (-b, b)


def plot_evaluation(df):
    w = figure_sizing.fig_width["inch"]["double"]
    h = w / 2 * 0.9

    percentiles = (5, 95)
    radii = df.index.unique("radius")

    res_dir = Config.path.SPHERE / "results"

    limits = {
        m: np.percentile(df[m].to_numpy(), [0.5, 99.5])
        for m in df.columns.unique("Metric")
    }
    limits["RDM"][0] = 0
    limits["lnMAG"] = _symmetrize(limits["lnMAG"])

    # Radius
    fig, axes = plt.subplots(1, 2, figsize=(w, h), constrained_layout=True)
    for m, ax in zip(df.columns.unique("Metric"), axes):
        mu = df[m].groupby("radius").mean().to_numpy()
        plow, phigh = np.percentile(
            df[m].to_numpy().reshape((*df.index.levshape, -1)), percentiles, axis=(1, 2)
        )
        for j, k in zip(plow.T, phigh.T):
            ax.fill_between(radii, j, k, alpha=0.25)
        ax.plot(radii, mu, label=df.columns.unique("Density").tolist())

        ax.set_ylim(limits[m])
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_title(m)
        if m == "RDM":
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        # ax.vlines(77, *ax.get_ylim(), color="gray", linewidth=1, linestyle="--")
    fig.savefig(res_dir / "sphere_eccentricity")

    # CDF
    n = len(df.index)
    ix = np.arange(n)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(w, h), constrained_layout=True)
    for metric, ax in zip(df.columns.unique("Metric"), axes):
        sval = np.sort(df[metric].to_numpy(), 0)
        ax.plot(sval, ix / n)
        ax.set_xlim(limits[metric])
        ax.set_xlabel(metric)
        # ax.set_title('Cumulative Distribution Function')
        if metric == "RDM":
            ax.set_ylabel("Cumulative Relative Frequency")
        if metric == "lnMAG":
            ax.legend(df.columns.unique("Density"), loc="upper left")
        ax.grid(alpha=0.25)

    fig.savefig(res_dir / "sphere_crf")

if __name__ == "__main__":
    df = collect_evaluation()
    plot_evaluation(df)
