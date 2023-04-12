from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.preprocessing import MaxAbsScaler

from simnibs.simulation import eeg_viz

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.facerecognition import utils
from projects.facerecognition.config import Config


style = "presentation" # publication, presentation

if style == "publication":
    mpl_styles.set_publication_styles()
    suffix = ""
elif style == "presentation":
    mpl_styles.set_dark_styles()
    suffix = "_dark"
    plt.rcParams["savefig.format"] = "svg"

pv.set_plot_theme("document")

"""
kw = dict(stage="inverse", space="fsaverage", suffix="stc", extension="pickle")

io = utils.GroupIO()
io.data.update(**kw)

df = pd.concat(
    [pd.read_pickle(io.data.get_filename(subject=sub)) for sub in io.subjects],
    axis="columns",
)

scaler = MaxAbsScaler()
scaler.fit(df)
df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

levels = list(df.columns.names)
levels.remove("Subject")
df_mean = df.mean(axis="columns", level=levels)
scaler = MaxAbsScaler()
scaler.fit(df_mean)
df_mean_scaled = pd.DataFrame(
    scaler.transform(df_mean), index=df_mean.index, columns=df_mean.columns
)


morph = mne.read_source_morph("/mrhome/jesperdn/INN_JESPER/projects/facerecognition/resources/fsaverage5_to_fsaverage7-morph-morph.h5")

vertnos = np.concatenate(morph.vertices_to)
hemis = np.concatenate(
    [[h] * len(v) for h, v in zip(("lh", "rh"), morph.vertices_to)]
)
rows = pd.MultiIndex.from_arrays([hemis, vertnos], names=["Hemi", "Source"])
df_mean_scaled_7 = pd.DataFrame(morph.morph_mat @ df_mean_scaled.to_numpy(), index=rows, columns=df_mean_scaled.columns)
df_mean_scaled_7.to_pickle("/mrhome/jesperdn/df_mean_scaled_7_facerecog.pickle")

df_mean_scaled = pd.read_pickle("/mrhome/jesperdn/df_mean_scaled_7_facerecog.pickle")
df_mean_scaled.columns = df_mean_scaled.columns.set_levels(["SimNIBS-CHARM", "FT-SPM", "MNE-FS", "MNI-Template"], "Forward")

brain = eeg_viz.FsAveragePlotter() # Config.inverse.FSAVERAGE
brain.add_curvature()

for i in range(3, 9):
    p = inv_plot_surface_interactive(df_scaled[f"{i:02d}"], surf)
    p.show()
    p.close()

p = inv_plot_surface_interactive(brain, df_mean_scaled)
p.show()


fig = inv_plot_surface(brain, df_mean_scaled)
fig.savefig(f"/mrhome/jesperdn/inverse_facerecognition_anatomy{suffix}.png")


fig = plot_fmri_activations_group()
fig.savefig(f"/mrhome/jesperdn/inverse_facerecognition_fmri_group{suffix}.png")

"""

def inv_plot_surface(brain, df):

    window_size = (500, 500)

    contrast = "faces vs. scrambled"
    fwds = df.columns.unique("Forward")
    invs = df.columns.unique("Inverse")

    # show top 1% "activations" / activations above 99th percentile
    threshold = df.quantile(0.99, axis=0)#.mean(level=["Inverse"])

    overlay_kwargs = dict(show_scalar_bar=False)
    imgs = {fwd: {} for fwd in fwds}
    for fwd, inv in itertools.product(fwds, invs):

        if inv == "MUSIC":
            x = df[inv, fwd, contrast]/df[inv, fwd, contrast].max()
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p = brain.plot(
            df[inv, fwd, contrast],
            # threshold=0.75,
            threshold=threshold[inv, fwd, contrast],
            name="data",
            plotter=p,
            overlay_kwargs=overlay_kwargs,
        )
        p.enable_parallel_projection()

        imgs[fwd][inv] = p.screenshot(transparent_background=True)

    imgrid_kwargs = dict(
        axes_pad=0.1,
    )
    return make_image_grid(imgs, imgrid_kwargs=imgrid_kwargs)


def compute_mask(imgs, how, val):
    how = how or "all not equal"
    if how == "all not equal":
        mask = imgs != val
    elif how == "all less than":
        mask = imgs < val
    elif how == "all greater than":
        mask = imgs > val
    else:
        raise ValueError
    return np.any(mask[..., :3], axis=(0, -1))


def get_bbox_slices(mask):
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def get_aspect_ratio(rows, cols):
    """ width to heigh ratio"""
    return (cols.stop - cols.start) / (rows.stop - rows.start)


def _crop_imgs_from_dict(imgs, mask_how, mask_val):
    # Ignore possible alpha channel
    # nonbackground = np.any(
    #     np.stack(
    #         [vv[..., :3] != ignore_val for v in imgs.values() for vv in v.values()]
    #     ),
    #     axis=(0, -1),
    # )
    if isinstance(imgs, dict):
        first = next(iter(imgs))
        if isinstance(imgs[first], dict):
            all_imgs = np.stack(
                [vv[..., :3] for v in imgs.values() for vv in v.values()]
            )
        elif isinstance(imgs[first], list):
            all_imgs = np.stack([vv[..., :3] for v in imgs.values() for vv in v])
        elif isinstance(imgs[first], np.ndarray):
            all_imgs = np.stack(imgs.values())
        else:
            raise ValueError("`imgs` must be dict of dicts or dict of lists.")
    else:
        all_imgs = np.stack(imgs)
    nonbackground = compute_mask(all_imgs, mask_how, mask_val)
    rows, cols = get_bbox_slices(nonbackground)
    return rows, cols, get_aspect_ratio(rows, cols)


def init_image_grid(shape, aspect_ratio, width="double", imgrid_kw=None):
    imgrid_kwargs = {}
    if imgrid_kw:
        imgrid_kwargs.update(imgrid_kw)
        # share_all=True,
        # axes_pad=0.05,
        # cbar_mode="single",
        # cbar_location="right",
        # cbar_pad=0.1,

    nrows, ncols = shape
    pad = 0.02  # default padding in ImageGrid in inches
    w = figure_sizing.fig_width["inch"][width]
    w_eff = w + (ncols - 1) * pad - (nrows - 1) * pad
    figsize = w, w_eff * nrows / ncols / aspect_ratio  # width, height
    # figsize = figure_sizing.get_figsize(width, aspect_ratio, subplots=(nrows,ncols))
    fig = plt.figure(figsize=figsize)
    return fig, ImageGrid(fig, 111, nrows_ncols=shape, **imgrid_kwargs)


def make_image_grid(
    imgs,
    mask_how=None,
    mask_val=255,
    imgrid_kwargs=None,
    cmap=None,
    clim=None,
    cbar_rows=None,
    return_cbar=False,
    width="double",
):
    """Create image grid with first level in rows and second level in columns.

    imgs : dict of dicts | dict of lists
    """
    rows, cols, aspect_ratio = _crop_imgs_from_dict(imgs, mask_how, mask_val)
    first = next(iter(imgs))
    dict_of_dicts = isinstance(imgs[first], dict)

    shape = len(imgs), len(imgs[first])
    fig, grid = init_image_grid(shape, aspect_ratio, width, imgrid_kw=imgrid_kwargs)
    for i, (row_ax, first) in enumerate(zip(grid.axes_row, imgs)):
        for j, (ax, second) in enumerate(zip(row_ax, imgs[first])):
            if dict_of_dicts:
                ax.imshow(imgs[first][second][rows, cols])
            else:
                ax.imshow(second[rows, cols])
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0 and dict_of_dicts:
                ax.set_title(second)
            if j == 0:
                ax.set_ylabel(first)
    if cmap:
        assert clim is not None
        assert (
            isinstance(imgrid_kwargs, dict) and imgrid_kwargs["cbar_mode"] == "single"
        )
        if cbar_rows:
            if isinstance(cbar_rows, int):
                rescale_cbar(grid, cbar_rows)
            else:
                rescale_cbar(grid, *cbar_rows)
        cbar = fig.colorbar(get_scalarmappable(cmap, clim), cax=grid.cbar_axes[0])
        if return_cbar:
            return fig, cbar
    fig.tight_layout()
    return fig


def make_image_grid_from_flat(
    imgs,
    shape,
    mask_how=None,
    mask_val=255,
    imgrid_kwargs=None,
    cmap=None,
    clim=None,
    cbar_rows=None,
    return_cbar=False,
):
    """
    imgs : dict of ndarrays

    """
    rows, cols, aspect_ratio = _crop_imgs_from_dict(imgs, mask_how, mask_val)
    is_dict = isinstance(imgs, dict)

    fig, grid = init_image_grid(shape, aspect_ratio, imgrid_kw=imgrid_kwargs)
    for ax, label in zip(grid.axes_all, imgs):
        if is_dict:
            ax.imshow(imgs[label][rows, cols])
            ax.set_title(label)
        else:
            ax.imshow(label[rows, cols])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if cmap:
        assert clim is not None
        assert (
            isinstance(imgrid_kwargs, dict) and imgrid_kwargs["cbar_mode"] == "single"
        )
        if cbar_rows:
            if isinstance(cbar_rows, int):
                rescale_cbar(grid, cbar_rows)
            else:
                rescale_cbar(grid, *cbar_rows)
        # set orientation='horizontal' if top or bottom...
        cbar = fig.colorbar(get_scalarmappable(cmap, clim), cax=grid.cbar_axes[0])
        if return_cbar:
            return fig, cbar
    fig.tight_layout()
    return fig


def get_scalarmappable(cmap, clim):
    return plt.cm.ScalarMappable(plt.Normalize(*clim), plt.get_cmap(cmap))


def rescale_cbar(grid, first, last=None, location="right"):
    last = last or first
    n = len(grid.axes_row) if location in ("left", "right") else len(grid.axes_column)
    first = 2 * first
    last = 2 * (last - n) + 1
    # nx is index into grid.get_divider().get_horizontal()
    # ny is index into grid.get_divider().get_vertical()
    # It is always the last column so nx is fixed
    if location == "right":
        locator = grid._divider.new_locator(nx=-2, ny=first, ny1=last)
    elif location == "left":
        locator = grid._divider.new_locator(nx=0, ny=first, ny1=last)
    elif location == "top":
        locator = grid._divider.new_locator(nx=first, ny=-2, nx1=last)
    elif location == "bottom":
        locator = grid._divider.new_locator(nx=first, ny=0, nx1=last)
    else:
        raise ValueError
    grid.cbar_axes[0].set_axes_locator(locator)


def inv_plot_surface_interactive(brain, df):
    """

    px : pixels per subplot

    """
    scalar_bar_args = dict(vertical=False, n_labels=3, label_font_size=10)
    px = 300
    contrast = "faces vs. scrambled"

    fwds = list(df.columns.unique("Forward"))
    n_fwds = len(fwds)
    invs = df.columns.unique("Inverse")
    n_invs = len(invs)
    zoom_factor = 1.3

    shape = (n_fwds, n_invs)
    window_size = tuple(px * i for i in shape[::-1])
    p = pv.Plotter(shape=shape, window_size=window_size, notebook=False, border=False)
    for (i, fwd), (j, inv) in itertools.product(enumerate(fwds), enumerate(invs)):
        p.subplot(i, j)
        p = brain.plot(df[inv, fwd, contrast], threshold=0.75, name="data", plotter=p)
        # p.view_xy(negative=True)
        # p.camera.zoom(zoom_factor)
        if j == 0:
            p.add_text(fwd, "left_edge", font_size=12)
        if i == 0:
            p.add_text(inv, "upper_right", font_size=8)
    p.link_views()

    return p


def plot_fmri_activations_group():
    kw_plot = dict(threshold=3, use_abs=True)
    kw_overlay = dict(scalar_bar_args=dict(vertical=True, title_font_size=30, label_font_size=24))
    window_size = (1000,1000)
    df_group = pd.read_pickle(Config.path.RESULTS / "fmri_fsaverage_group.pickle")
    brain = eeg_viz.FsAveragePlotter()
    brain.add_curvature()
    p = pv.Plotter(off_screen=True, window_size=window_size)
    brain.plot(df_group["group"], name="t-statistic", plotter=p, overlay_kwargs=kw_overlay, **kw_plot)
    p.enable_parallel_projection()
    # p.show()
    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("single"))
    img = p.screenshot(transparent_background=True)
    ax.imshow(img[img[...,3].any(1)][:, img[...,3].any(0)])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig



def plot_fmri_activations_subject():

    kw_plot = dict(threshold=3, use_abs=True)

    df = pd.read_pickle(Config.path.RESULTS / "fmri_fsaverage_subject.pickle")

    brain = eeg_viz.FsAveragePlotter()
    brain.add_curvature()

    overlay_kwargs = dict(show_scalar_bar=False)

    imgs = {}
    for subject in df.columns:
        # brain.add_overlay(df[subject], f"fmri activation {subject}")
        p = brain.plot(
            df[subject], name="t-statistic", **kw_plot, overlay_kwargs=overlay_kwargs
        )
        p.enable_parallel_projection()
        imgs[subject] = p.screenshot(transparent_background=True)

    return make_image_grid_from_flat(imgs, shape=(4, 4))
