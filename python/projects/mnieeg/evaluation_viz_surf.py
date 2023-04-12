import json
import itertools
from pathlib import Path
import re
from string import ascii_uppercase

# import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import mne
# import mne.channels._standard_montage_utils
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv

# from pyvistaqt import BackgroundPlotter


# from scipy.stats import f_oneway, normaltest, ttest_rel
import scipy.stats

from simnibs.simulation import eeg, eeg_viz
from simnibs.utils.csv_reader import read_csv_positions

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.mnieeg import utils
from projects.mnieeg.config import Config

from projects.facerecognition.evaluation_viz_surf import make_image_grid
from projects.facerecognition.evaluation_viz_surf import make_image_grid_from_flat

from projects.mnieeg.phatmags.mrifiducials import MRIFiducials
from projects.mnieeg.evaluation_collect import get_summary_statistics

# Set this environment variable (otherwise BackgroundPlotter fails beucase it
# cannot find the correct plugin file...)
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
#     Path(os.environ["CONDA_PREFIX"]) / "Library" / "plugins" / "platforms"
# )

style = "publication" # publication, presentation

if style == "publication":
    mpl_styles.set_publication_styles()
    suffix = ""
elif style == "presentation":
    mpl_styles.set_dark_styles()
    suffix = "_dark"
    plt.rcParams["savefig.format"] = "svg"

pv.set_plot_theme("document")

# FORWARD EVALUATION
# =============================================================================

# from pathlib import Path

# output_dir = (
#     Path(r"C:\Users\jdue\Desktop\desktop_backup\anateeg_results") / "figure_forward"
# )
# if not output_dir.exists():
#     output_dir.mkdir()

# df_fname = (
#     Path(r"C:\Users\jdue\Desktop\desktop_backup\anateeg_results") / "forward.pickle"
# )
# df = pd.read_pickle(df_fname)


def ras_to_neuromag(nasion, lpa, rpa):

    right = rpa - lpa
    right_unit = right / np.linalg.norm(right)

    origin = lpa + np.dot(nasion - lpa, right_unit) * right_unit

    anterior = nasion - origin
    anterior_unit = anterior / np.linalg.norm(anterior)

    superior_unit = np.cross(right_unit, anterior_unit)

    x, y, z = -origin
    origin_trans = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float
    )
    trans_l = np.vstack((right_unit, anterior_unit, superior_unit, [0, 0, 0]))
    trans_r = np.reshape([0, 0, 0, 1], (4, 1))
    rot_trans = np.hstack((trans_l, trans_r))

    return np.dot(rot_trans, origin_trans)


def make_angle_arc(origin, normal, polar):
    angle_arc = pv.merge(
        [
            pv.CircularArcFromNormal(origin, 100, normal, polar, angle=a)
            for a in (180, -180)
        ]
    )
    angle_arc.rename_array("Distance", "Angle")
    angle_arc["Angle"] *= 180 / np.pi
    return angle_arc


def plot_angle_measurement(subject_id):

    skin_color = "#FFE0BD"
    angle_arc_r = 15

    io = utils.SubjectIO(subject_id)
    skin = pv.read(io.simnibs.root / f"sub-{io.subject}" / "skin_outer_annot.vtk")
    with open(io.data.path.root / "prepare_phatmags.json") as f:
        orig_fids = getattr(MRIFiducials, json.load(f)[io.subject]["original_subject"])
    landmarks = eeg.Montage(landmarks=orig_fids)
    landmarks.project_to_surface(
        dict(points=np.array(skin.points), tris=skin.faces.reshape(-1, 4)[:, 1:]),
        np.where(skin["outer_points"])[0],
    )

    t, c, _, l, _, _ = read_csv_positions(
        io.simnibs.root / f"sub-{io.subject}" / "montage_digitized_proj.csv"
    )
    t = np.array(t)
    l = np.array(l)
    elec = pv.PolyData(c[t == "Electrode"])

    t, c, _, l, _, _ = read_csv_positions(
        io.simnibs.root / f"sub-{io.subject}" / "montage_custom_nonlin_proj.csv"
    )
    t = np.array(t)
    l = np.array(l)
    custom = pv.PolyData(c[t == "Electrode"])

    trans = ras_to_neuromag(
        landmarks.landmarks["nasion"],
        landmarks.landmarks["lpa"],
        landmarks.landmarks["rpa"],
    )

    skin.points = eeg.apply_trans(trans, skin.points)
    elec.points = eeg.apply_trans(trans, elec.points)
    custom.points = eeg.apply_trans(trans, custom.points)
    landmarks.landmarks = {
        k: eeg.apply_trans(trans, v) for k, v in landmarks.landmarks.items()
    }
    elec_label = l[t == "Electrode"]
    elec_dict = dict(zip(elec_label, elec.points))

    xvec = landmarks.landmarks["rpa"] - landmarks.landmarks["lpa"]
    xvec /= np.linalg.norm(xvec)
    origin = (
        landmarks.landmarks["lpa"]
        + np.dot(landmarks.landmarks["nasion"] - landmarks.landmarks["lpa"], xvec)
        * xvec
    )
    yvec = landmarks.landmarks["nasion"] - origin
    yvec /= np.linalg.norm(yvec)
    zvec = np.cross(xvec, yvec)
    zvec /= np.linalg.norm(zvec)

    xdir, ydir, zdir = np.identity(3)

    # measurement plane, direction to extend
    stuff = {  # ("lpa", "70"): ("yz", ),
        ("rpa", "71"): ([1,2], xdir, ydir, 0, 3, 5, 1), # yz = [1, 2]
        ("nasion", "43"): ([0,2], ydir, xdir, 1, 7, 7, 0), # xz = [0,2]
        ("inion", "64"): ([0,2], ydir, xdir, 1, -1, -2, 0), # xz = [0,2]
    }

    arc = {}
    arc_label = {}
    distance = {}
    distance_label = {}
    angle_arc = {}
    for k, v in stuff.items():
        lm, el = k
        pax, vec0, vec1, add_to_dir, add_to_mm, add_to_arc, rel_ax = v

        lm_pos = landmarks.landmarks[lm]
        el_pos = elec_dict[el].copy()
        el_pos[add_to_dir] = lm_pos[add_to_dir]

        lm_to_el = el_pos - lm_pos
        d = np.linalg.norm(lm_to_el)
        lm_to_el /= d
        arc[k] = pv.CircularArc(lm_pos + vec1 * d, el_pos, lm_pos)
        arc[k].points[:, add_to_dir] += add_to_mm

        distance[k] = pv.Line(lm_pos, el_pos)
        distance[k].points[:, add_to_dir] += add_to_mm

        arc_label[k] = np.rad2deg(np.arctan2(*lm_to_el[pax[::-1]]))
        distance_label[k] = np.linalg.norm(np.diff(distance[k].points, axis=0))

        angle_arc[k] = make_angle_arc(lm_pos + add_to_arc * vec0, vec0, vec1)
        expand_axes = [i for i in range(3) if i != add_to_dir]
        expand_coo = angle_arc[k].points[:, expand_axes]
        angle_arc[k].points[:, expand_axes] += angle_arc_r * (
            expand_coo - expand_coo.mean(0)
        )

    q = 0.1
    viewvecs = [
        xdir - q * zdir,
        ydir + q * zdir + q * xdir,
        -ydir + q * zdir + q * xdir,
    ]

    xy_plane = pv.Plane(origin, zvec, i_size=200, j_size=200)
    yz_plane = pv.Plane(origin, xvec, i_size=220, j_size=220)

    kwargs_label = dict(font_size=40, show_points=False, shape_opacity=0.75)
    labels = [("Digitized", "red"), ("Initial", "blue"), ("Landmarks", 'green')]

    imgs = {}
    for k, vv in zip(stuff, viewvecs):
        p = pv.Plotter(window_size=(1000, 1000))
        p.add_mesh(skin, color=skin_color)
        p.add_mesh(elec, color="red", point_size=30, render_points_as_spheres=True)
        p.add_mesh(custom, color="blue", point_size=30, render_points_as_spheres=True)
        p.add_mesh(
            landmarks.get_landmark_pos(),
            color="green",
            point_size=30,
            render_points_as_spheres=True,
        )
        p.add_mesh(xy_plane, color="yellow", opacity=0.5)
        p.add_mesh(yz_plane, color="red", opacity=0.5)

        # arc
        p.add_mesh(arc[k], color="black", line_width=4)
        point2label = arc[k].points[arc[k].n_points // 2]
        label = f"{arc_label[k]:.0f} deg"
        p.add_point_labels([point2label], [label], **kwargs_label)

        # line/distance
        p.add_mesh(distance[k], color="magenta", line_width=4)
        point2label = 0.25*distance[k].points[0] + 0.75*distance[k].points[1]
        label = f"{distance_label[k]:.0f} mm"
        p.add_point_labels([point2label], [label], **kwargs_label)
        # for a, aa in zip(arc.values(), angle_arc.values()):
        # p.add_mesh(a, color="black", line_width=3)
        # p.add_mesh(aa, cmap="coolwarm", line_width=5)

        if k == ("inion", "64"):
            p.add_legend(labels, bcolor="white", size=(0.2, 0.2),
                loc="lower right", face="circle")

        # p.add_text("My angle", tuple(arc.points[50]))
        p.view_vector(vv)
        p.camera.zoom(2.5)
        p.add_axes()
        # p.enable_parallel_projection()
        imgs["{}, {}".format(*k)] = p.screenshot(transparent_background=True)
        # p.show()

    return make_image_grid_from_flat(imgs, (1, 3), "all not equal", 255)


def plot_sensors_optimized(subject_id):

    skin_color = "#FFE0BD"

    io = utils.SubjectIO(subject_id)
    io.data.update(
        stage="preprocessing",
        task=None,
        prefix="montage",
        suffix="optcoords",
        extension="pickle",
    )
    df = pd.read_pickle(io.data.get_filename())
    opt_coords = pv.PolyData(df[io.subject, 0, 0, 0].to_numpy().reshape(-1, 3))

    io.data.update(suffix="refcoords")
    df = pd.read_pickle(io.data.get_filename())
    ref_coords = pv.PolyData(df[io.subject].to_numpy().reshape(-1, 3))

    sub_dir = io.simnibs.get_path("subject")
    montage = eeg.make_montage(sub_dir / "montage_custom_nonlin_proj.csv")
    ini_coords = pv.PolyData(montage.ch_pos)

    skin = pv.read(sub_dir / "skin_outer_annot.vtk")

    kwargs = dict(point_size=15, render_points_as_spheres=True)
    view_vecs = [
        (0.57, 0.57, 0.57),
        # (-0.57, 0.57, 0.57),
        # (0.57, -0.57, 0.57),
        (-0.57, -0.57, 0.57),
    ]
    labels = [("Digitized", "red"), ("Initial", "blue"), ("Optimized", 'green')]
    imgs = []
    for i, vv in enumerate(view_vecs):
        p = pv.Plotter(window_size=(1000, 1000))
        p.add_mesh(skin, color=skin_color)
        p.add_mesh(ref_coords, color="red", label="Digitized", **kwargs)
        p.add_mesh(ini_coords, color="blue", label="Initial", **kwargs)
        p.add_mesh(opt_coords, color="green", label="Optimized", **kwargs)
        p.view_vector(vv)
        p.camera.zoom(1.41)
        if i == 0:
            p.add_legend(labels, bcolor="white", size=(0.2, 0.2),
                loc="lower center", face="circle")
        # p.add_axes()
        # p.enable_parallel_projection()
        imgs.append(p.screenshot(transparent_background=True))
        # p.show()
    return make_image_grid_from_flat(imgs, (1, len(view_vecs)), "all not equal", 255)

def make_angle_wheel():
    # make angular plot axes
    angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    labels = [
        tuple(f"$\mathbf{{{i}}}^\circ$" for i in (0, 45, 90, 135, 180, -135, -90, -45)),
        tuple(f"$\mathbf{{{i}}}^\circ$" for i in (180, 135, 90, 45, 0, -45, -90, -135))
    ]
    angle_wheel = {}

    for k,l in zip(("regular", "reverse"), labels):
        fig = plt.figure(constrained_layout=True, figsize=(4,4))
        ax = fig.add_subplot(111, projection='polar')
        ax.grid(True)
        ax.set_yticks([0.33, 0.67], [])
        ax.set_xticks(angles, l, fontsize=25)
        # for i,tl in enumerate(ax.get_xticklabels()):
        #     if i in (0, 1, 7):
        #         tl.set_horizontalalignment("left")
        #     elif i in (3,4,5):
        #         tl.set_horizontalalignment("right")

        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, np.uint8).reshape((height, width, 4))

        angle_wheel[k] = image

    lm2wheel = dict(nasion="reverse", rpa="regular", inion="regular", lpa="reverse")
    return angle_wheel, lm2wheel

def plot_angle_measurement_with_angle_wheel(subject_id):
    inset_size = 0.4

    angle_wheel, lm2wheel = make_angle_wheel()
    fig = plot_angle_measurement(subject_id)
    for ax in fig.axes[:3]:
        axin = ax.inset_axes([1-inset_size, 1-inset_size, inset_size, inset_size])
        lm = ax.get_title().split(",")[0]
        axin.imshow(angle_wheel[lm2wheel[lm]])
        axin.set_axis_off()
    return fig


# subject_id = 1
# fig = plot_angle_measurement(subject_id)
# fig.savefig(Config.path.RESULTS / "figure_optimization" / f"angle_meas_{subject_id:02d}.png")

# subject_id = 1
# fig = plot_angle_measurement_with_angle_wheel(subject_id)
# fig.savefig(Config.path.RESULTS / "figure_optimization" / f"angle_meas_{subject_id:02d}.png")

# subject_id = 5
# fig = plot_sensors_optimized(subject_id)
# fig.savefig(
#     Config.path.RESULTS / "figure_optimization" / f"sens_opt_{subject_id:02d}.png"
# )


def get_cbar_limits(array, low, high, decimals=1, symmetrize=False):
    limits = np.round(np.percentile(array, [low, high]), decimals)
    if symmetrize:
        limits = np.array([-1, 1]) * np.abs(limits).max()
    return limits


# def get_central_fsaverage():
#     src = eeg.FsAverage(Config.inverse.FSAVERAGE)
#     return surf_to_multiblock(src.get_central_surface())


# def surf_to_multiblock(surf):
#     mb = pv.MultiBlock()
#     for hemi in surf:
#         mb[hemi] = pv.make_tri_mesh(surf[hemi]["points"], surf[hemi]["tris"])
#     return mb


def forward_plot():
    output_dir = Config.path.RESULTS / "figure_forward"
    if not output_dir.exists():
        output_dir.mkdir()

    df_fname = Config.path.RESULTS / "forward.pickle"
    df = pd.read_pickle(df_fname)

    dfs = get_summary_statistics(df, ["Metric", "Forward", "Orientation"])

    fig = fwd_plot_density(df)
    fig.savefig(output_dir / f"forward_density{suffix}")

    fig = fwd_plot_crf(df)
    fig.savefig(output_dir / f"forward_crf{suffix}")

    brain = eeg_viz.FsAveragePlotter(Config.inverse.FSAVERAGE, "central")
    for metric in dfs.columns.unique("Metric"):
        fig = fwd_plot_stat_on_surface(brain, dfs, metric, cbar_rows=1)
        fig.savefig(output_dir / f"forward_source_{metric}{suffix}.png")

    # Distance matrices
    df_fname = Config.path.RESULTS / "forward_distance_matrix.pickle"
    df = pd.read_pickle(df_fname)

    fig = fwd_plot_distance_matrix(df)
    fig.savefig(output_dir / f"forward_distance_matrices{suffix}")


def fwd_plot_density(df, n_points=200):
    models = df.columns.unique("Forward")

    points = {}
    for m in df.columns.unique("Metric"):
        limits = get_cbar_limits(df[m].to_numpy().ravel(), 1, 99, 2, m == "lnMAG")
        if m == "RDM":
            limits[0] = 0
        points[m] = np.linspace(*limits, n_points)

    # points = {
    #     "RDM": np.linspace(0, 1, n_points),
    #     "lnMAG": np.linspace(-0.5, 0.5, n_points),
    # }
    w = figure_sizing.fig_width["inch"]["double"]
    h = w / 2
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(w, h), constrained_layout=True)
    for i, (ax, metric) in enumerate(zip(axes, df.columns.unique("Metric"))):
        for model in models:
            kernel = scipy.stats.gaussian_kde(df[metric, model].to_numpy().ravel())
            ax.plot(points[metric], kernel(points[metric]))
            # q = df.loc[metric, model].to_numpy().ravel()
            # print(model)
            # print(
            #     f"{q.min():0.3f}   {q.mean():0.3f}   {np.median(q):0.3f}   {q.max():0.3f}"
            # )
            # print(f"{q.std():0.3f}")
            # ax.hist(q, 100, density=True, histtype='step')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(metric)
        if i == 0:
            ax.legend(models, facecolor='none')
            ax.set_ylabel("Probability Density")
    return fig


def fwd_plot_crf(df):
    models = df.columns.unique("Forward")

    n = len(df.index) * len(df.columns.unique("Subject"))
    n_metrics = len(df.columns.unique("Metric"))
    ix = np.arange(n)

    limits = {
        m: get_cbar_limits(df[m].to_numpy().ravel(), 1, 99, 2, m == "lnMAG")
        for m in df.columns.unique("Metric")
    }
    limits["RDM"][0] = 0

    w = figure_sizing.fig_width["inch"]["double"]
    h = w / 2
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(w, h), constrained_layout=True)
    for i, (ax, metric) in enumerate(zip(axes, df.columns.unique("Metric"))):
        for model in models:
            sval = np.sort(df[metric, model].to_numpy().ravel(), 0)
            ax.plot(sval, ix / n)
        ax.set_xlim(limits[metric])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(metric)
        if i == 0:
            ax.legend(models, facecolor="none")
            ax.set_ylabel("Cumulative Relative Frequency")
    return fig


def fwd_plot_distance_matrix(df):
    metrics = df.index.unique("Metric")
    models = df.index.unique("Forward")
    n_models = len(models)
    n_subjects = len(df.index.unique("Subject"))

    kw = dict(vmin=df.to_numpy().min(), vmax=df.to_numpy().max(), cmap="inferno")
    tick_pos = np.arange(n_subjects // 2, n_models * n_subjects + 1, n_subjects)

    w = figure_sizing.fig_width["inch"]["double"]
    h = w * 0.8
    fig = plt.figure(figsize=(w, h))
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(1, len(metrics)),
        axes_pad=0.05,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.1,
    )
    for ax, metric in zip(grid, metrics):
        im = ax.imshow(df.loc[metric], **kw)
        ax.set_title(f"Average {metric}")
        ax.tick_params(left=False, bottom=False)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(models, rotation=45)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(models, rotation=45)
    _ = ax.cax.colorbar(im)  # cbar
    # cbar = grid.cbar_axes[0].colorbar(im)
    # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])

    return fig


def fwd_add_stats_to_multiblock(df, surf, metric):
    combs = list(
        itertools.product(df.index.unique("Forward"), df.index.unique("Statistic"),)
    )
    mb = pv.MultiBlock()
    for hemi in surf:
        mb[hemi] = pv.make_tri_mesh(surf[hemi]["points"], surf[hemi]["tris"])
    for hemi in surf:
        for model, stat in combs:
            k = ", ".join((stat, model))
            mb[hemi][k] = df[hemi].loc[stat, metric, model].squeeze()
    return mb


def get_views():
    view_vector = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    view_up = [(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
    assert len(view_vector) == len(view_up)
    return view_vector, view_up


def fwd_plot_stat_on_surface(
    brain, df, metric, stat="mean", cbar_pad=0.1, cbar_size="2.5%", cbar_rows=None,
):

    ori = "normal"
    zoom_factor = 1.3

    models = df.columns.unique("Forward")
    n_models = len(models)

    view_vector, view_up = get_views()
    # view_name = ("z-", "z+", "y-", "y+", "x-", "x+")

    symmetrize = dict(RDM=False, lnMAG=True)
    try:
        clim = Config.plot.limits["forward"][stat][metric]
    except KeyError:
        clim = get_cbar_limits(
            df[stat, metric].to_numpy().ravel(),
            1,
            99,
            decimals=1,
            symmetrize=symmetrize[metric],
        )
        if not symmetrize[metric]:
            clim[0] = 0
    use_cmap = dict(RDM="Reds", lnMAG="RdBu_r")[metric]

    plotter_kwargs = dict(off_screen=True, window_size=(400, 400))
    overlay_kwargs = dict(cmap=use_cmap, clim=clim, show_scalar_bar=False)

    # Make individual plots
    imgs = {}
    for model in models:
        imgs[model] = []
        for vv, vu in zip(view_vector, view_up):
            p = pv.Plotter(**plotter_kwargs)
            p = brain.plot(
                df[stat, metric, model, ori],
                name="x",
                overlay_kwargs=overlay_kwargs,
                plotter=p,
            )
            p.view_vector(vv)
            p.set_viewup(vu)
            p.camera.zoom(zoom_factor)
            p.enable_parallel_projection()

            imgs[model].append(p.screenshot(transparent_background=True))
            p.close()

    cbar_kwargs = dict(cmap=use_cmap, clim=clim, cbar_rows=cbar_rows)
    imgrid_kwargs = dict(cbar_mode="single", cbar_size=cbar_size, cbar_pad=cbar_pad)
    return make_image_grid(imgs, "all not equal", 255, imgrid_kwargs, **cbar_kwargs)


def fwd_plot_stat_on_surface_vtjks():

    p.export_vtkjs(r"C:\Users\jdue\Desktop\results\figure_forward\forward_source_rdm")


def fwd_plot_interactive(df, surf):

    fun = "psf"
    met = "sd_ext"
    snr = 3
    inverse = "MNE"

    p = pv.Plotter(shape=(2, 2), notebook=False)
    kw = dict(clim=[3, 7], below_color="blue", above_color="red")
    for i, model in zip(
        itertools.product(range(2), range(2)), df.index.unique("Forward")
    ):
        k = " ".join((inverse, str(snr), model))
        p.subplot(*i)
        p.add_text(k)
        for hemi in surf:
            # Copy needed!!!
            scalars = (
                df.loc[pd.IndexSlice[:, model, inverse, snr, fun, met], hemi]
                .to_numpy()
                .mean(0)
            )
            p.add_mesh(
                surf[hemi].copy(), scalars=scalars, **kw, show_scalar_bar=i == (1, 1)
            )
            p.view_xy(negative=True)
    p.link_views()
    p.show()
    # p.show(notebook=False)


def inverse_plot(results_dir, fsaverage):
    output_dir = results_dir / "figure_inverse"
    if not output_dir.exists():
        output_dir.mkdir()

    # stats = ("mean",)

    df_dens = pd.read_pickle(results_dir / "inverse_density.pickle")
    df_summary = pd.read_pickle(results_dir / "inverse_summary.pickle")

    prod_dens = itertools.product(
        df_dens.columns.unique("Orientation"),
        df_dens.columns.unique("Resolution Function"),
        df_dens.columns.unique("Resolution Metric"),
    )
    fwd1 = df_summary.columns.unique("Forward")[0]
    snr1 = df_summary.columns.unique("SNR")[0]
    prod_surf = df_summary.loc[
        :, pd.IndexSlice["mean", :, fwd1, :, snr1]
    ].columns.droplevel(["Forward", "SNR"])

    # df_dens = df_dens.drop(["MNE", "LCMV"], axis=1, level="Inverse")

    limits = Config.plot.limits["inverse"]["density"]
    for ori, fun, met in prod_dens:
        name_fun = Config.plot.names["res_fun"][fun]
        name_met = Config.plot.names["res_met"][met]
        fig = inverse_plot_density(df_dens, ori, fun, met, name_fun, name_met, limits)
        fig.savefig(output_dir / f"inverse_density_{ori}_{fun}_{met}{suffix}")

    brain = eeg_viz.FsAveragePlotter(fsaverage, "central")
    for stat, ori, inv, fun, met in prod_surf:
        fig = inverse_plot_stat_on_surface(
            brain, df_summary, stat, ori, inv, fun, met, Config, cbar_rows=(1, 2)
        )
        fig.savefig(output_dir / f"inverse_source_{ori}_{stat}_{inv}_{fun}_{met}{suffix}.png")


"""
fun = "psf"
met = "peak_err"

df_dens = inv_make_density(df, fun, met, density_type="hist")
df_dens_fixed = inv_make_density(dff, fun, met, density_type="hist")

fig = inv_plot_density(df_dens)
fig.suptitle(f"{fun} : {met}")
fig.savefig(Config.path.RESULTS / "inv.png")

fig = inv_plot_density(df_dens_fixed)
fig.suptitle(f"{fun} : {met}")
fig.savefig(Config.path.RESULTS / "inv_fixed.png")
"""


def inverse_plot_density(df, ori, fun, met, name_fun=None, name_met=None, limits=None):

    fmt = mpl.ticker.StrMethodFormatter("{x:.1f}")

    dfp = prune_columns(df.loc[:, pd.IndexSlice[ori, :, :, :, fun, met]])

    models = dfp.columns.unique("Forward")
    snrs = dfp.columns.unique("SNR")
    invs = dfp.columns.unique("Inverse")
    n_invs = len(invs)
    n_snrs = len(snrs)

    # try:
    #     lim_range = Config.plot.limits["inverse"]["density"][met]
    # except KeyError:
    #     lim_range = None
    # try:
    #     lim_prob = Config.plot.limits["inverse"]["density"]["probdens"]
    # except KeyError:
    #     lim_prob = None

    lim_range = limits[met] if limits is not None else limits
    lim_prob = limits["probdens"] if limits is not None else limits

    figsize = figure_sizing.get_figsize("double", subplots=(n_invs, n_snrs))
    fig, axes = plt.subplots(
        n_invs,
        n_snrs,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        figsize=figsize,
    )
    for i, (row, inv) in enumerate(zip(np.atleast_2d(axes), invs)):
        for j, (ax, snr) in enumerate(zip(row, snrs)):
            for model in models:
                ax.plot(dfp.index, dfp[model, inv, snr])
            # ax.set_yscale('log')
            if lim_range:
                ax.set_xlim(*lim_range)
            if lim_prob:
                ax.set_ylim(*lim_prob)
                # ax.set_ylim(*[1e-4,2]) # log scale
            if i == 0:
                ax.set_title(f"SNR = {snr}")
                if j == n_snrs - 1:
                    ax.legend(models, facecolor="none")
            if j == 0:
                ax.set_ylabel(inv)
                ax.yaxis.set_major_formatter(fmt)
            if i == n_invs - 1:
                ax.set_xlabel("cm")
            ax.grid(True, alpha=0.5)
    # Suptitle
    names = dict(met=name_met or met, fun=name_fun or fun, ori=ori,)
    fig.suptitle("{met} of {fun} ({ori})".format(**names))
    return fig


def prune_columns(df):
    """Remove column levels with only one unique value."""
    mask = list(filter(lambda n: len(df.columns.unique(n)) == 1, df.columns.names))
    return df.droplevel(mask, axis=1)


def inverse_plot_stat_on_surface(
    brain,
    df,
    stat,
    ori,
    inv,
    fun,
    met,
    config,
    plot_difference=False,
    cbar_pad=0.1,
    cbar_size="2.5%",
    cbar_rows=None,
):
    invert_view = (False, True)

    dfp = prune_columns(df.loc[:, pd.IndexSlice[stat, ori, :, inv, :, fun, met]])

    # Setup
    fwds = list(dfp.columns.unique("Forward"))
    snrs = list(dfp.columns.unique("SNR"))
    cmap = "viridis"  # YlGnBu
    clim = config.plot.limits["inverse"]["surf"][stat][met]
    if plot_difference:
        fwd_ref = config.plot.fwd_model_name[config.forward.REFERENCE]
        fwds.remove(fwd_ref)
        cmap = "RdBu_r"
        # clim[0] = -clim[1]
    overlay_kwargs = dict(cmap=cmap, clim=clim, show_scalar_bar=False)
    plotter_kwargs = dict(off_screen=True, window_size=(500, 500))

    # Make individual plots
    imgs = {}
    for fwd in fwds:
        imgs[fwd] = {}
        for snr in snrs:
            scalars = dfp[fwd, snr]
            if plot_difference:
                scalars -= dfp[fwd_ref, snr]
            vs = []
            for iv in invert_view:
                p = pv.Plotter(**plotter_kwargs)
                p = brain.plot(
                    scalars, name="cm", overlay_kwargs=overlay_kwargs, plotter=p,
                )
                p.view_xy(iv)
                p.enable_parallel_projection()
                p.parallel_scale /= np.sqrt(2)
                vs.append(p.screenshot(transparent_background=True))
                p.close()
            # crop all white columns
            vs_crop = np.concatenate(
                [x[:, np.any(x[..., :3] != 255, axis=(0, 2))] for x in vs], axis=1
            )
            imgs[fwd][f"SNR = {snr}"] = vs_crop

    cbar_kwargs = dict(
        cmap=overlay_kwargs["cmap"],
        clim=overlay_kwargs["clim"],
        cbar_rows=cbar_rows,
        return_cbar=True,
    )
    imgrid_kwargs = dict(
        axes_pad=0.1, cbar_mode="single", cbar_size=cbar_size, cbar_pad=cbar_pad
    )
    fig, cbar = make_image_grid(
        imgs, "all not equal", 255, imgrid_kwargs, **cbar_kwargs
    )
    cbar.ax.set_title("cm")
    # Suptitle
    names = dict(
        stat=config.plot.names["stat"][stat],
        met=config.plot.names["res_met"][met],
        fun=config.plot.names["res_fun"][fun],
        inv=inv,
        ori=ori,
    )
    fig.suptitle("{inv} ({ori}) / {fun} / {stat} {met}".format(**names))
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


"""
stats = ('mean', 'median', 'std')
inv = 'MNE'
stat = 'mean'
fig = inv_plot_stat_on_surface(brain, dfs, inv, "psf", "peak_err", stat, cbar_rows=(1,2))


p = inv_plot_stat_on_surface_interactive(brain, dfs, inv, "psf", "peak_err", stat)
p.export_html("/mrhome/jesperdn/test")
p.show()


p = inv_plot_stat_on_surface_interactive(brain, dffs, inv, "psf", "peak_err")
p.show()

"""


def inv_plot_stat_on_surface_interactive(
    brain,
    df,
    inverse,
    function,
    metric,
    stat="mean",
    plot_difference=False,
    cbar_pad=0.1,
    cbar_size="2.5%",
    cbar_rows=None,
):
    """

    px : pixels per subplot

    """
    zoom_factor = 1  # np.sqrt(2)

    fwds, snrs, fwd_ref, overlay_kwargs = _inv_plot_stat_on_surface_setup(
        df, stat, function, metric, plot_difference
    )
    overlay_kwargs["show_scalar_bar"] = True
    overlay_kwargs["scalar_bar_args"] = dict(
        vertical=False, n_labels=3, label_font_size=10
    )

    n_fwds = len(fwds)
    n_snrs = len(snrs)

    shape = (n_snrs, n_fwds)
    window_size = tuple(400 * i for i in shape[::-1])
    p = pv.Plotter(shape=shape, window_size=window_size, notebook=False, border=False)
    for i, snr in enumerate(snrs):
        for j, fwd in enumerate(fwds):
            scalars = df[stat, fwd, inverse, snr, function, metric]
            if plot_difference:
                scalars -= df[stat, fwd_ref, inverse, snr, function, metric]
            p.subplot(i, j)
            p = brain.plot(
                scalars, name=metric, overlay_kwargs=overlay_kwargs, plotter=p,
            )
            p.camera.zoom(zoom_factor)
            if j == 0:
                p.add_text(str(snr), "left_edge", font_size=12)
            if i == 0:
                p.add_text(fwd, "upper_edge", font_size=8)
    p.link_views()
    return p


def _inv_plot_stat_on_surface_setup(df, stat, function, metric, plot_difference):

    cmap = "RdBu_r" if plot_difference else "viridis"

    fwds = df.columns.unique("Forward")
    snrs = df.columns.unique("SNR")
    fwds = list(fwds)
    fwd_ref = Config.plot.fwd_model_name[Config.forward.REFERENCE]
    model_ref_idx = fwds.index(fwd_ref)
    if plot_difference:
        fwds.remove(fwd_ref)

    # Setup colorbar
    names = {"Forward", "Inverse", "SNR"}
    df_all_data = df.loc[:, pd.IndexSlice[stat, :, :, :, function, metric]]
    column_subset = df_all_data.columns.remove_unused_levels()
    all_data = df_all_data.to_numpy()
    if plot_difference:
        all_data = all_data.reshape(
            *tuple(
                shape
                for name, shape in zip(column_subset.names, column_subset.levshape)
                if name in names
            ),
            all_data.shape[0],
        )
        all_data = (
            all_data[[i for i in range(all_data.shape[0]) if i != model_ref_idx]]
            - all_data[model_ref_idx]
        )
    clim = get_cbar_limits(all_data[all_data > 0], 5, 95, 1, plot_difference)
    if metric in ("peak_err", "cog_err") and not plot_difference:
        clim[0] = 0

    overlay_kwargs = dict(cmap=cmap, clim=clim, show_scalar_bar=False)
    return fwds, snrs, fwd_ref, overlay_kwargs
