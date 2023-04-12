import copy
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.sparse
import scipy.optimize

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pandas as pd
import pyvista as pv

from simnibs.mesh_tools import mesh_io
from simnibs.utils.file_finder import get_montage_neighbors
from simnibs.simulation import eeg
from projects.mnieeg.config import Config
# from projects.mnieeg.forward import get_outer_surface_points

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing

style = "presentation" # publication, presentation

if style == "publication":
    mpl_styles.set_publication_styles()
    suffix = ""
elif style == "presentation":
    mpl_styles.set_dark_styles()
    suffix = "_dark"
    plt.rcParams["savefig.format"] = "svg"

pv.set_plot_theme("document")

# import os
# os.chdir("/mrhome/jesperdn/project_tools/python/projects/mnieeg/commands/subject")
from projects.mnieeg.commands.subject.optimize_montage import load_neighbors, compute_sphere_parameters, dist_constraint, compute_channel_coords, neighbor_dist, cost_neighbor

from importlib.machinery import SourceFileLoader

def compute_ras_to_neuromag_transform(landmarks):
    # similar to mne.transforms.py

    nas = landmarks["Nz"]
    lpa = landmarks["LPA"]
    rpa = landmarks["RPA"]

    right = rpa - lpa
    right_unit = right / np.linalg.norm(right)

    origin = lpa + np.dot(nas - lpa, right_unit) * right_unit

    anterior = nas - origin
    anterior_unit = anterior / np.linalg.norm(anterior)

    superior_unit = np.cross(right_unit, anterior_unit)

    x, y, z = -origin
    origin_trans = np.array([[1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1]], dtype=float)
    trans_l = np.vstack((right_unit, anterior_unit, superior_unit, [0, 0, 0]))
    trans_r = np.reshape([0, 0, 0, 1], (4, 1))
    rot_trans = np.hstack((trans_l, trans_r))

    return rot_trans @ origin_trans

def get_measurements_from_csv(f):
    with open(f, "r") as of:
        headers = of.readline().rstrip("\n").split(",")
        dist_dict = {}
        angle_dict = {}
        for line in of:
            lm, el, dist, angle = line.rstrip("\n").split(",")
            if dist:
                dist_dict[(lm, el)] = float(dist)
            if angle:
                angle_dict[(lm, el)] = float(angle)
    return dict(dist=dist_dict, angle=angle_dict)


def angle_constraint(
    params,
    init_theta,
    init_phi,
    center,
    radius,
    angle_ch_idx,
    target_angle,
    angle_planes,
    lm_pos_angle,
):
    # vector pointing from landmark to channel
    v = (
        compute_channel_coords(
            params, init_theta, init_phi, center, radius, angle_ch_idx
        )
        - lm_pos_angle
    )
    # angle (-pi, pi)
    # (ras: x points to the right, y points to the front, z points to the top)
    # xz : angle between (1, 0) and v (y coordinate ignored)
    # yz : angle between (1, 0) and v (x coordinate ignored)
    i = np.arange(angle_planes.shape[0])
    angles = np.arctan2(v[i, angle_planes[:,1]], v[i, angle_planes[:,0]])
    return np.abs(np.angle(np.conj(np.exp(1j * angles)) * np.exp(1j * target_angle)))

def _prepare_angle_planes(measurements):
    # xz : (2,0)
    # yz : (2,1)
    plane = dict(nz=(0,2), iz=(0,2), lpa=(1,2), rpa=(1,2))
    return np.array([plane[lm.lower()] for lm,_ in measurements["angle"]])


def _prepare_measurement_arrays(measurements, ch_names):

    measurement_arrays = {}
    channel_idx = {}
    for m,v in measurements.items():
        x = v.values()
        if m == "angle":
            x = map(np.deg2rad, x)
        measurement_arrays[m] = np.array(list(x))
        channel_idx[m] = [ch_names.index(ch) for _, ch in v.keys()]

    return measurement_arrays, channel_idx


def _get_landmark_positions(measurements, landmarks):
    return {m: np.array([landmarks[lm] for lm, _ in v.keys()]) for m, v in measurements.items()}


def optimize_montage(montage, measurements, landmarks_mri, montage_neighbors, tol_dist=2.5, tol_angle=5):

    # Compute some stuff...
    tol_angle = np.deg2rad(tol_angle)

    ch_names = montage.ch_names.tolist()
    measurement_arrays, channel_idx = _prepare_measurement_arrays(measurements, ch_names)
    landmark_pos = _get_landmark_positions(measurements, landmarks_mri.landmarks)
    angle_planes = _prepare_angle_planes(measurements)

    # reparametrize cartesian channel coordinates to spherical coordinates on a
    # sphere fitted to each channel and its neighbors.
    center, radius, init_theta, init_phi = compute_sphere_parameters(
        montage.ch_pos, montage_neighbors
    )

    # Setup constraints
    dist_constraint_partial = partial(
        dist_constraint,
        init_theta=init_theta,
        init_phi=init_phi,
        center=center,
        radius=radius,
        dist_ch_idx=channel_idx["dist"],
        lm_pos_dist=landmark_pos["dist"],
        target_dist=measurement_arrays["dist"],
    )
    angle_constraint_partial = partial(
        angle_constraint,
        init_theta=init_theta,
        init_phi=init_phi,
        center=center,
        radius=radius,
        angle_ch_idx=channel_idx["angle"],
        target_angle=measurement_arrays["angle"],
        angle_planes=angle_planes,
        lm_pos_angle=landmark_pos["angle"],
    )
    constraints = [
        scipy.optimize.NonlinearConstraint(
            dist_constraint_partial, -tol_dist, tol_dist
        ),
        scipy.optimize.NonlinearConstraint(
            angle_constraint_partial, -tol_angle, tol_angle
        ),
    ]

    ch_pos0 = montage.ch_pos
    ch_dist0 = neighbor_dist(ch_pos0, montage_neighbors)

    # weight neighbor penalties such that all electrodes contribute equally
    # (otherwise those with the most neighbors dominate the loss)
    # n_neighbors = np.asarray(neighbors.sum(0)).squeeze()
    # weights = np.repeat(1 / n_neighbors, n_neighbors)
    weights = 1  # / ch_dist0

    # I.e., we start at (init_theta, init_phi) which is the original channel
    # positions
    x0 = np.zeros((montage.n_channels, 2)).ravel()

    res = scipy.optimize.minimize(
        cost_neighbor,
        x0,
        method="trust-constr",
        constraints=constraints,
        args=(weights, montage_neighbors, ch_dist0, init_theta, init_phi, center, radius),
        options=dict(disp=True, maxiter=100),
    )
    ch_pos_final = compute_channel_coords(res.x, init_theta, init_phi, center, radius)

    montage_opt = copy.copy(montage)
    montage_opt.ch_pos = ch_pos_final

    return montage_opt


# p = pv.Plotter()
# p.add_mesh(skin, scalars=None)
# p.add_mesh(montage.ch_pos, c="red")
# p.add_mesh(ch_pos_final, c="green")
# p.add_mesh(, c="blue")

root = Path("/mrhome/jesperdn/nobackup/khm")
m2m = root / "m2m_khm"

montage_name = Config.forward.MONTAGE_SIMNIBS
tol_dist = 2.5
tol_angle = 5


# skin
# m = mesh_io.read_msh(m2m / "khm.msh")
# m = m.crop_mesh(1005)
# surf = {"points": m.nodes.node_coord, "tris": m.elm.node_number_list[:, :3] - 1}
# subset = get_outer_surface_points(m)

# skin = pv.make_tri_mesh(surf["points"], surf["tris"])
# skin["outer_points"] = np.zeros(skin.n_points, dtype=int)
# skin["outer_points"][subset] = 1
# skin.save(root / "skin_outer_annot.vtk")

skin = pv.read(root / "skin_outer_annot.vtk")
surf = {"points": skin.points, "tris": skin.faces.reshape(-1,4)[:, 1:]}
subset = np.where(skin["outer_points"])[0]

montage = eeg.make_montage(montage_name)
# montage.add_landmarks()
mni2mri = nib.load(m2m / "toMNI" / "MNI2Conform_nonl.nii.gz")
montage.apply_deform(mni2mri)
montage.project_to_surface(surf, subset)
montage.write(root / f"{montage_name}_nonlin.csv")

# marked on MRI
landmarks_mri = eeg.Montage()
landmarks_mri.add_landmarks(root / 'landmarks_mri.csv')

mri_head_t = compute_ras_to_neuromag_transform(landmarks_mri.landmarks)

fname = get_montage_neighbors("easycap_BC_TMS64_X21")
montage_neighbors, ch_names = load_neighbors(fname)
# we do not use the reference electrode here...
# neighbors = scipy.sparse.coo_matrix(neighbors.todense()[1:, 1:])
# ch_names = ch_names[1:]
n_channels = len(ch_names)

skin = pv.read(root / "skin_outer_annot.vtk")

lm_mapper = dict(LPA="LPA", RPA="RPA", Nasion="Nz", Inion="Iz")

# SETUP
sessions = [2,3,4,6,7]
use_inion = True
use_extra_measurements = True

for session in sessions:

    montage = eeg.make_montage(root / f"{montage_name}_nonlin.csv")

    landmarks_mri = eeg.Montage()
    landmarks_mri.add_landmarks(root / 'landmarks_mri.csv')

    session_name = f"validation{session}"
    if use_inion:
        session_name += "_inion"
    print(f"Session : {session_name}")
    montage_dig = eeg.make_montage(root / session_name / f"{session_name}.csv")

    # measurements
    # imports the module from the given path
    f = str(root / f"session_Validation{session}" / "measurements.py")
    print(f"Meansurements : {f}")
    measurements = SourceFileLoader("measurements", f).load_module()
    measurements = measurements.measurements

    if not use_extra_measurements:
        delete = [k for k in measurements["dist"] if k not in measurements["angle"]]
        for d in delete:
            del measurements["dist"][d]
        for k in measurements:
            for kk in measurements[k]:
                measurements[k][kk]
    else:
        print("Using all measurements")

    measurements = {k:{(lm_mapper[kk[0]], kk[1]):vv for kk, vv in v.items()} for k,v in measurements.items()}

    #### CONVERT EVERYTHING TO HEAD COORDINATES!!!

    montage.apply_trans(mri_head_t)
    landmarks_mri.apply_trans(mri_head_t)

    montage_opt = optimize_montage(montage, measurements, landmarks_mri, montage_neighbors, tol_dist, tol_angle)

    # convert back to MRI space
    montage.apply_trans(np.linalg.inv(mri_head_t))
    montage_opt.apply_trans(np.linalg.inv(mri_head_t))

    montage_opt.project_to_surface(surf, subset)

    err_ini = np.linalg.norm(montage_dig.ch_pos - montage.ch_pos, axis=1)
    err_opt = np.linalg.norm(montage_dig.ch_pos - montage_opt.ch_pos, axis=1)

    print(f"mean error (initial)   : {err_ini.mean():.2f} mm")
    print(f"mean error (optimized) : {err_opt.mean():.2f} mm")
    print()

    f = f"{session_name}{'_extra' if use_extra_measurements else ''}_opt.csv"
    print(f"Writing {f}")
    print()
    montage_opt.write(root / session_name / f)


plt.figure();
plt.hist(err_ini, alpha=0.5);
plt.hist(err_opt, alpha=0.5);
plt.legend(["ini", "opt"]);

plt.figure();
plt.hist(err_ini-err_opt);




# skin.clear_point_data()

kwargs = dict(point_size=10, render_points_as_spheres=True)
p = pv.Plotter(notebook=False)
p.add_mesh(skin)
p.add_mesh(pv.PolyData(montage_dig.ch_pos), color="green", label="Digitized", **kwargs)
p.add_mesh(pv.PolyData(montage.ch_pos), color="blue", label="Initial", **kwargs)
p.add_mesh(pv.PolyData(montage_opt.ch_pos), color="pink", label="Optimized", **kwargs)
p.add_legend()
p.show()


index = pd.Index(ch_names, name="Channel")
index_coo = pd.MultiIndex.from_product(
    (ch_names, ["x", "y", "z"]), names=["Channel", "Coordinates"]
)
columns = pd.MultiIndex.from_product(
    (
        sessions,
        [False, True],
        [False, True],
    ),
    names=["Session", "Use Inion", "Use Extra"],
)
columns_ini = pd.MultiIndex.from_product((sessions, [False, True]), names=["Session", "Use Inion"])

df_dist_opt = pd.DataFrame(index=index, columns=columns)
df_dist_ini = pd.DataFrame(index=index, columns=columns_ini)

# df_coo_opt = pd.DataFrame(index=index_coo, columns=columns)
# df_coo_ini = pd.DataFrame(index=index_coo, columns=columns_ini)



m_ini = eeg.make_montage(root / f"{montage_name}_nonlin.csv")
for session in sessions:
    session_name = f"validation{session}"

    for use_inion in (False, True):
        if use_inion:
            session_name += "_inion"
        m_dig = eeg.make_montage(root / session_name / f"{session_name}.csv")
        df_dist_ini[session, use_inion] = np.linalg.norm(m_ini.ch_pos-m_dig.ch_pos, axis=1)
        # df_coo_ini[session] = m

        for use_extra in (False, True):
            f = f"{session_name}{'_extra' if use_extra else ''}_opt.csv"
            m_opt = eeg.make_montage(root / session_name / f)
            df_dist_opt[session, use_inion, use_extra] = np.linalg.norm(m_opt.ch_pos-m_dig.ch_pos, axis=1)
            # df_coo_opt[session, use_inion, use_extra] = m_opt.ch_pos

df_dist_opt.to_pickle(root / "dist_optimized.pickle")
df_dist_ini.to_pickle(root / "dist_initial.pickle")

dfi = pd.read_pickle(root / "dist_initial.pickle")
dfo = pd.read_pickle(root / "dist_optimized.pickle")
df_dist_diff = dfo-dfi


n_points = 200
# data_opt = df_dist_opt.to_numpy()
# data_ini = df_dist_ini.to_numpy()
points = np.linspace(df_dist_diff.to_numpy().min(), df_dist_diff.to_numpy().max(), n_points)

fig, axes = plt.subplots(2,2, sharex=True, sharey=True, constrained_layout=True, figsize=figure_sizing.get_figsize("double"))
for i,use_inion in enumerate(df_dist_diff.columns.unique("Use Inion")):
    for j,use_extra in enumerate(df_dist_diff.columns.unique("Use Extra")):
        for session in df_dist_diff.columns.unique("Session"):
            kernel = scipy.stats.gaussian_kde(df_dist_diff[session, use_inion, use_extra].to_numpy())
            axes[i,j].plot(points, kernel(points))
            axes[i,j].grid(alpha=0.25)
        if j == 0:
            axes[i,j].set_ylabel(use_inion)
        if i == 1:
            axes[i,j].set_xlabel(use_extra)
fig.suptitle("Probability Density of Error Difference (mm) over Sessions")
fig.supxlabel("Use Extra Measurements")
fig.supylabel("Use Inion")

fig.savefig(root / "channel_opt_subj_error_dark")



# hardcoded
#
#   coords = np.linspace(0, 40, 100)
#
# in /matplotlib/cbook/__init__.py to avoid cutting of violinplots!

fig = make_violinplots(dfi, dfo)
fig.savefig(root / "channel_opt_subj_violin_dark")


def make_violinplots(dfi, dfo):

    offsets = np.linspace(-0.35, 0.35, 6)
    width = 0.08

    n_sessions = len(dfo.columns.unique("Session"))
    kwargs = dict(widths=width, showmeans=True, showextrema=False)
    positions = np.arange(n_sessions)

    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("double"))
    parts = {
        "00": ax.violinplot(dfi.loc[:, pd.IndexSlice[:, False]], positions=positions+offsets[0], **kwargs),
        "01": ax.violinplot(dfi.loc[:, pd.IndexSlice[:, True]], positions=positions+offsets[1], **kwargs),
        "100": ax.violinplot(dfo.loc[:, pd.IndexSlice[:, False, False]], positions=positions+offsets[2], **kwargs),
        "110": ax.violinplot(dfo.loc[:, pd.IndexSlice[:, True, False]], positions=positions+offsets[3], **kwargs),
        "101": ax.violinplot(dfo.loc[:, pd.IndexSlice[:, False, True]], positions=positions+offsets[4], **kwargs),
        "111": ax.violinplot(dfo.loc[:, pd.IndexSlice[:, True, True]], positions=positions+offsets[5], **kwargs)
    }

    inicolor = parts["00"]['bodies'][0].get_facecolor()
    optcolor = parts["01"]['bodies'][0].get_facecolor()
    hatchcolor = "black"
    use_inion_hatch = "///"
    use_extra_hatch = "\\\\\\"
    both_hatch = "XXX"

    for p in parts.values():
        p["cmeans"].set_color("white")

    alpha = 0.5

    for p in parts["00"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0) # remove edge line (but not hatches!)
    for p in parts["01"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0)
        p.set_facecolor(inicolor)
        p.set_hatch(use_inion_hatch)
        p.set_edgecolor(hatchcolor)
    for p in parts["100"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0)
        p.set_facecolor(optcolor)
        p.set_edgecolor(hatchcolor)
    for p in parts["110"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0)
        p.set_facecolor(optcolor)
        p.set_hatch(use_inion_hatch)
        p.set_edgecolor(hatchcolor)
    for p in parts["101"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0)
        p.set_facecolor(optcolor)
        p.set_hatch(use_extra_hatch)
        p.set_edgecolor(hatchcolor)
    for p in parts["111"]["bodies"]:
        p.set_alpha(alpha)
        p.set_linewidth(0)
        p.set_facecolor(optcolor)
        p.set_hatch(both_hatch)
        p.set_edgecolor(hatchcolor)


    legend_elements = [
        Patch(color=inicolor, label='Initial'),
        Patch(color=optcolor, label='Optimized'),
        Patch(facecolor="none", edgecolor="gray", hatch=use_inion_hatch, label='Use Inion'),
        Patch(facecolor="none", edgecolor="gray", hatch=use_extra_hatch, label='Use Extra'),
    ]

    ax.grid(alpha=0.25)
    ax.set_xlabel("Session")
    ax.set_ylabel("Error (mm)")
    ax.set_title("Density Estimate of Errors")
    ax.legend(handles=legend_elements, loc="upper left", facecolor="none")

    return fig

"""
The inion is generally more difficult to locate than the nasion, LPA, and RPA.
Therefore, it is not clear whether it should be used in the coregistration.
Fitting seven parameters using nine data points may be susceptible to
inaccuracies in the nine data points. On the other hand, using three additional
data points only makes sense, if they are sufficiently accurate.

- Determining the rotation around the x-axis may be difficult using only the
  three landmarks as LPA and RPA do not really contain any information about
  this. Hence, any noise in nasion will influence this considerably. Using inion
  may stabilize this even though its position may be more inaccurate than the
  others.
- The rotation around the y-axis is mostly determined by LPA and RPA and should
  not be particularly sensitive to whether or not the inion is used.
- The rotation around the z-axis should be quite stable even using only three
  points.

The error gets smaller for initial *and* optimized montages when using the
inion to coregister the electrodes to the MR (i.e., the head model).

Since including the inion in the coregistration causes the digitized positions
to better match the template positions, we may assume that it actually enhances
the coregistration

The measurements are probably made wrt. the same landmark positions as those
which was digitized. Hence,

The effect of the optimization procedure, however, seems to be insensitive to
whether the inion is used or not (except for session 2 but the effect is
relatively small).

average projection distance (in mm) for digitized electrode positions on skin
surface.

we see that using inion consistently decreases projection distance (i.e., it
results in a slightly better fit to the head shape) suggesting that these
positions may be more appropriate as ground truth.

Given that there is a flooring effect here (the projection cannot be smaller
than zero), the fact that the mean decreases

        & Use inion? \\
Session & No & Yes \\
2       & 2.78 & 2.66 \\
3       & 3.79 & 3.02 \\
4       & 3.26 & 2.93 \\
6       & 3.76 & 3.03 \\
7       & 3.28 & 2.51 \\

"""