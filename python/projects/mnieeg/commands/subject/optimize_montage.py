import copy
from functools import partial
import json
import sys

# import mne
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.optimize
import scipy.sparse
import scipy.spatial
import scipy.stats


from simnibs.simulation import eeg
from simnibs.utils.file_finder import get_montage_neighbors

from projects.mnieeg import utils
from projects.mnieeg.config import Config
from projects.mnieeg.phatmags.mrifiducials import MRIFiducials


# from projects.base.geometry import get_adjacency_matrix
# from projects.mnieeg.evaluation_viz_topo import create_info


def load_neighbors(filename, as_sparse=True):
    with open(filename, "r") as f:
        neighbors = json.load(f)

    if not as_sparse:
        return neighbors

    n = len(neighbors)
    ch_names = list(neighbors.keys())
    ch_neighbors = list(neighbors.values())
    ch_names2index = {k: i for i, k in enumerate(ch_names)}

    row_ind = np.repeat(np.arange(n), [len(i) for i in ch_neighbors])
    col_ind = np.array([ch_names2index[j] for i in ch_neighbors for j in i])
    data = np.ones_like(row_ind)
    neighbors = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n, n))

    return neighbors, ch_names


def fit_hypersphere_to_points(points):
    """
    conformal geometric algebra (CGA) representation of points.

    Represent points in (n+2)-D space such that p(euclidean) = [x, y, z] becomes

        p(conform) = [x, y, z, 1, 0.5*np.sum(p**2)]

    such that dot products between points in conformed space corresponds to
    distances in euclidean space

    The equation for a sphere in 3D is

        (x - x0)2 + (y - y0)2 + (z - z0)2 = r2

    where (x0, y0, z0) and r is the center and radius of the sphere,
    respectively.



    Reparametrizing as a polynomial gives

        A (x2 + y2 + z2) + B x + C y + D z = 1

    We need to determine the parameters A, B, C, D given the design matrix

        [x2 + y2 + z2, x, y, z]

    and dependent variable

        1

    for each point. Then convert back to our parameters of interest

        center = [a, b, c] = [-B/(2 A), -C/(2 A), -D/(2 A)]
        R = sqrt(4 A + B2 + C2 + D2)/(2 A)

    REFERENCES
    ----------
    L Dorst. Total Least Squares Fitting of k-Spheres in n-D Euclidean Space
        Using an (n+2)-D Isometric Representation. Journal of Mathematical
        Imaging and Vision, 2014, p. 1-21.

    """

    # See 3.3 Implementation
    n, m = points.shape
    assert n >= m + 1, f"Need at least {m+1} points to fit {m+1} parameters"

    # Build the metric matrix
    M = np.eye(m + 2)
    M[-2, -2] = M[-1, -1] = 0
    M[-2, -1] = M[-1, -2] = -1

    # D is the matrix of points represented in (n+2)-D space (conformed points)
    D = np.column_stack((points, np.ones(n), 0.5 * np.sum(points ** 2, 1))).T
    P = D @ D.T @ M / n

    w, v = np.linalg.eig(P)
    i = w == w[w > 0].min()
    loss = w[i]
    vi = v[:, i].squeeze()
    vi /= vi[-2]

    center = vi[:3]
    radius = np.sqrt(np.sum(center ** 2) - 2 * vi[-1])

    """
    A = np.column_stack((np.sum(points ** 2, 1), points))
    b = np.ones(points.shape[0])
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # x = [a, b, ...]
    x0_i2 = 0.5 / x[0]
    center = -x[1:] * x0_i2
    radius = np.sqrt(4 * x[0] + np.sum(x[1:] ** 2)) * x0_i2
    assert radius > 0, "Fitting failed (radius is negative)"

    mb = pv.MultiBlock()
    mb["points"] = pv.PolyData(points)
    mb["sphere1"] = pv.Sphere(radius, center)

    print(np.sum((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    print(np.linalg.norm(points - center, axis=1))
    print(radius)
    print(center)

    """
    return center, radius, loss


# dist = {
#     ('nasion', '43'): 3,
#     ('inion', '54'): 2,
#     ('inion', '61'): 1
# }
# loc = {
#     ('inion', '54'): ('above', 'left'),
#     ('inion', '61'): ('below', 'left')
# }
# angles = {
#     #('inion', '51'): dict(xz=113.3007951),
#     ('inion', '64'): dict(xz=-152.42356129),
#     ('lpa', '70'): dict(yz=-35.40811625),
#     ('rpa', '71'): dict(yz=-34.83624616)
# }


def cart_to_sph(points):
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(points[:, 2] / r)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return r, theta, phi


def sph_to_cart(r, theta, phi):
    return np.column_stack(
        (
            r * np.cos(phi) * np.sin(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(theta),
        )
    )


def compute_channel_coords(params, theta, phi, center, radius, subset=None):
    if subset:
        theta_add, phi_add = params.reshape(-1, 2)[subset].T
        return (
            sph_to_cart(
                radius[subset], theta[subset] + theta_add, phi[subset] + phi_add
            )
            + center[subset]
        )
    else:
        theta_add, phi_add = params.reshape(-1, 2).T
        return sph_to_cart(radius, theta + theta_add, phi + phi_add) + center


def compute_sphere_parameters(pos, neighbors):

    # We want to fit the sphere to a channel *and* its neighbors
    neighbors_w_diag = neighbors.copy()
    neighbors_w_diag.setdiag(1)

    # electrodes 70 and 71 only have two neighbors so we need to add these to
    # be able to fit a sphere (four parameters)
    neighbors_w_diag = neighbors_w_diag.tolil()
    # neighbors_w_diag[37, 62] = 1
    # neighbors_w_diag[62, 37] = 1
    # neighbors_w_diag[47, 61] = 1
    # neighbors_w_diag[61, 47] = 1

    # Ensure all points have at least five neighbors such that there are six
    # points to fit the sphere
    tree = scipy.spatial.cKDTree(pos)
    d, n = tree.query(pos, k=6)

    for i in range(len(pos)):
        j = 0
        while neighbors_w_diag[i].sum() < 6:
            k = n[i][j]
            neighbors_w_diag[i, k] = 1
            neighbors_w_diag[k, i] = 1
            j += 1

    neighbors_w_diag = neighbors_w_diag.tocoo()
    assert neighbors_w_diag.sum(0).min() >= 4

    center, radius = [], []
    # mb = pv.MultiBlock()
    # mb["points"] = pos
    for i in range(neighbors_w_diag.shape[0]):
        c, r, loss = fit_hypersphere_to_points(pos[neighbors_w_diag.getrow(i).indices])
        center.append(c)
        radius.append(r)
        # print(loss)
        # mb[str(ch_names[i + 1])] = pv.Sphere(r, c)
    center = np.array(center)
    radius = np.array(radius)

    # To compute spherical coordinates, we project each channel on its sphere
    # centered on the origin.

    # Move channels to sphere centered at the origin
    ch_on_sphere = pos - center
    # Each sphere was fitted to a channel and its neighbors so the channel will
    # most likely not lie exactly on the sphere, thus, we project each channel
    # onto its best-fitting sphere
    r_ch_sphere = np.linalg.norm(ch_on_sphere, axis=1)
    ch_on_sphere *= (radius / r_ch_sphere)[:, None]
    radius_, init_theta, init_phi = cart_to_sph(ch_on_sphere)
    assert np.allclose(radius, radius_)

    return center, radius, init_theta, init_phi


def _angle_dict_to_array(angle_dict):
    planes = ("xz", "yz")
    arr_shape = (len(angle_dict), len(planes))
    angle_arr = np.zeros(arr_shape)  # cols: xy, yz
    angle_arr_valid = np.zeros(arr_shape, dtype=bool)
    for i, lmch in enumerate(angle_dict):
        if angle_dict[lmch] is not None:
            for j, plane in enumerate(planes):
                if plane in angle_dict[lmch]:
                    value = angle_dict[lmch][plane]
                    angle_arr[i, j] = value
                    angle_arr_valid[i, j] = True
    return angle_arr[angle_arr_valid], angle_arr_valid


def _prepare_target_arrays(dist_dict, angle_dict, ch_names):

    target_dist = np.array([d for d in dist_dict.values()])
    target_angle, target_angle_valid = _angle_dict_to_array(angle_dict)
    target_angle = np.deg2rad(target_angle)

    dist_ch_idx = [ch_names.index(ch) for _, ch in dist_dict]
    angle_ch_idx = [ch_names.index(ch) for _, ch in angle_dict]

    return target_dist, target_angle, target_angle_valid, dist_ch_idx, angle_ch_idx


def _angle_dict_harmonize(angle_dict, dist_dict):
    return {
        lmch: angle_dict[lmch] if lmch in angle_dict else None for lmch in dist_dict
    }


def _angle_dict_to_rad(angle):
    return {
        lmch: {plane: np.deg2rad(angle[lmch][plane])}
        for lmch in angle
        for plane in angle[lmch]
    }


def dist_constraint(
    params, init_theta, init_phi, center, radius, dist_ch_idx, lm_pos_dist, target_dist
):
    ch_pos = compute_channel_coords(
        params, init_theta, init_phi, center, radius, dist_ch_idx
    )
    return np.linalg.norm(ch_pos - lm_pos_dist, axis=1) - target_dist


def angle_constraint(
    params,
    init_theta,
    init_phi,
    center,
    radius,
    angle_ch_idx,
    target_angle,
    target_angle_valid,
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
    angles = np.column_stack(
        (
            np.arctan2(v[:, 2], v[:, 0]),  # along zx in xy-plane
            np.arctan2(v[:, 2], v[:, 1]),
        )
    )  # along y in yz-plane
    angles = angles[target_angle_valid]

    return np.abs(np.angle(np.conj(np.exp(1j * angles)) * np.exp(1j * target_angle)))


def neighbor_dist(pos, neighbors):
    """Distance between neighboring points"""
    return np.linalg.norm(pos[neighbors.row] - pos[neighbors.col], axis=1)
    # return scipy.spatial.distance.pdist(x)


def cost_neighbor(
    x, weights, neighbors, x0_dist, init_theta, init_phi, sphere_center, sphere_r
):
    cart = compute_channel_coords(x, init_theta, init_phi, sphere_center, sphere_r)
    return np.sum(weights * (neighbor_dist(cart, neighbors) - x0_dist) ** 2)


# def grad_cost_neighbor(
#     x, weights, neighbors, x0_dist, init_theta, init_phi, sphere_center, sphere_r
# ):

#     return


# def cost_dist(x):
#     coords = compute_channel_coords(x, init_theta, init_phi, center, r)[ch_idx]
#     return np.linalg.norm(coords - lm_pos, axis=1) - dist


# def cost_angle(x):
#     # loc_q
#     # loc_q_valid
#     cart = compute_channel_coords(x, init_theta, init_phi, center, r)[ch_idx]
#     v = cart - lm_pos_angle

#     # angle in each
#     angles = np.column_stack(
#         (
#             np.arctan2(v[:, 2], v[:, 0]),  # along y / xy-plane
#             np.arctan2(v[:, 2], v[:, 1]),
#         )
#     )  # along x / yz-plane
#     angles = angles[loc_q_valid]

#     # the angle which is formed with the *first* axis (right, forward)
#     # in (-pi, pi)

#     return np.abs(np.angle(np.conj(np.exp(1j * angles)) * np.exp(1j * loc_q)))


# mne.viz.plot_topomap(diff,info,contours=0);


# # Differentiation stuff...
# reduceat_indices = np.concatenate(([0], np.where(np.diff(neighbors.row))[0]))


# def d_loss_fun(
#     x, weights, neighbors, x0_dist, init_theta, init_phi, sphere_center, sphere_r
# ):

#     2 * (neighbor_dist(cart, neighbors) - x0_dist) * d_neighbor_dist()

#     np.sum()


# def d_neighbor_dist(x, neighbors):

#     diff = x[neighbors.row] - x[neighbors.col]
#     # 2* because every pair occurs twice but is only counted once when reducing
#     return np.add.reduceat(
#         (2 * np.sum(diff ** 2, 1)) ** (-3 / 2), reduceat_indices
#     ) + np.add.reduceat((2 * 2 * np.sum(diff, 1)), reduceat_indices)


# #     def loss_fun(x, neighbors, nd0):
# #         """Sum of squared residuals between current distances and the initial
# #         ones."""
# #         return np.sum((neighbor_dist(x, neighbors) - nd0) ** 2)


def get_pseudo_measurements(montage, landmarks, lm_ch, skin, which_angles):
    """Get the geodesic distances between (landmark, channel) pairs."""

    ch_names = montage.ch_names.tolist()

    dist, angles = {}, {}
    ai = 0
    for lm, chs in lm_ch.items():
        pos = landmarks[lm]
        lm_v = skin.find_closest_point(pos)
        for ch in chs:
            i = ch_names.index(ch)

            # dist.append(
            #    np.linalg.norm(skin.points[lm_v]- skin.points[skin.find_closest_point(montage.ch_pos[i])])
            # )
            k = (lm, ch)
            v = montage.ch_pos[i] - pos
            if lm in {"lpa", "rpa"}:
                dist[k] = np.linalg.norm(v)
            else:
                dist[k] = skin.geodesic_distance(
                    lm_v, skin.find_closest_point(montage.ch_pos[i])
                )

            if which_angles[ai] == "xz":
                angles[(lm, ch)] = dict(xz=np.rad2deg(np.arctan2(v[2], v[0])))
            elif which_angles[ai] == "yz":
                angles[(lm, ch)] = dict(yz=np.rad2deg(np.arctan2(v[2], v[1])))

            ai += 1

    return dist, angles


# def _validate_angle(a):
#     if a > np.pi:
#         ...
#     elif a < -np.pi:
#         ...
#     else:
#         return a


def add_measurement_noise_dist(d, scale, rng):
    return {k: v + rng.normal(scale=scale) for k, v in d.items()}
    # return {k: v + noise * np.sign(np.random.random() - 0.5) for k, v in d.items()}


def add_measurement_noise_angle(d, scale, rng):
    return {
        k: {kk: vv + rng.normal(scale=scale) for kk, vv in v.items()}
        for k, v in d.items()
    }
    # return {
    #     k: {kk: vv + noise * np.sign(np.random.random() - 0.5) for kk, vv in v.items()}
    #     for k, v in d.items()
    # }



def run_optimization(
    montage, dist_dict, angle_dict, landmarks, neighbors, tol_dist=2.0, tol_angle=2.0
):
    """

    distance between channels

    euclidean distance (not geodesic or something like that)

    loss function

    sum of squared differences between the distances between channels
    at current iteration vs. the initial distances between channels, i.e., we
    want the relative positions between channels to be more or less maintained.

    constraints

    the distance between the (landmark, channel) pairs should be equal to the
    measured distance within some tolerance.

    PARAMETERS
    ----------
    montage :
        Montage to optimize.
    montage_ref :
        The reference montage from which the pseudo measurements are obtained.
    landmarks :
        Coordinates of landmarks.
    lm_ch :
        Dictionary with landmarks as keys and channel names as values defining
        the (landmark, channel) pairs whose distances were "measured", e.g.,
        {"nasion": ["43", "20"], "inion": ["64", "51"]}.
    tol :
        Tolerance (in mm). lb/ub = d -/+ tol.

    RETURNS
    -------
    Montage with optimized channel positions.

    """

    # Compute some stuff...
    tol_angle = np.deg2rad(tol_angle)

    ch_names = montage.ch_names.tolist()
    (
        target_dist,
        target_angle,
        target_angle_valid,
        dist_ch_idx,
        angle_ch_idx,
    ) = _prepare_target_arrays(dist_dict, angle_dict, ch_names)
    lm_pos_dist = np.array([landmarks[lm] for lm, _ in dist_dict])
    lm_pos_angle = np.array([landmarks[lm] for lm, _ in angle_dict])

    # reparametrize cartesian channel coordinates to spherical coordinates on a
    # sphere fitted to each channel and its neighbors.
    center, radius, init_theta, init_phi = compute_sphere_parameters(
        montage.ch_pos, neighbors
    )

    # Setup constraints
    dist_constraint_partial = partial(
        dist_constraint,
        init_theta=init_theta,
        init_phi=init_phi,
        center=center,
        radius=radius,
        dist_ch_idx=dist_ch_idx,
        lm_pos_dist=lm_pos_dist,
        target_dist=target_dist,
    )
    angle_constraint_partial = partial(
        angle_constraint,
        init_theta=init_theta,
        init_phi=init_phi,
        center=center,
        radius=radius,
        angle_ch_idx=angle_ch_idx,
        target_angle=target_angle,
        target_angle_valid=target_angle_valid,
        lm_pos_angle=lm_pos_angle,
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
    ch_dist0 = neighbor_dist(ch_pos0, neighbors)

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
        args=(weights, neighbors, ch_dist0, init_theta, init_phi, center, radius),
        options=dict(disp=True, maxiter=100),
    )
    ch_pos_final = compute_channel_coords(res.x, init_theta, init_phi, center, radius)

    montage_opt = copy.copy(montage)
    montage_opt.ch_pos = ch_pos_final

    return montage_opt


# x0 = np.zeros((montage.n_channels, 2)).ravel()

# m = []
# for i, j in enumerate(10 * [10]):

#     res = scipy.optimize.minimize(
#         cost_neighbor,
#         x0,
#         method="trust-constr",
#         constraints=constraints,
#         args=(weights, neighbors, ch_dist0, init_theta, init_phi, center, radius),
#         options=dict(disp=True, maxiter=j),
#     )
#     x0 = res.x
#     ch_pos_final = compute_channel_coords(res.x, init_theta, init_phi, center, radius)
#     montage_opt = copy.copy(montage)
#     montage_opt.ch_pos = ch_pos_final
#     montage_opt.project_to_surface(surf, subset)
#     m.append(montage_opt)

# import matplotlib.pyplot as plt

# cmap = plt.get_cmap("YlGn")
# kwargs = dict(point_size=10, render_points_as_spheres=True)

# p = pv.Plotter()
# p.add_mesh(pv.make_tri_mesh(surf["points"], surf["tris"]), **kwargs)
# p.add_mesh(pv.PolyData(montage.ch_pos), color="black", **kwargs)
# p.add_mesh(pv.PolyData(montage_ref.ch_pos), color="red", **kwargs)
# p.add_mesh(pv.PolyData(montage_opt.ch_pos), color="blue", **kwargs)
# for i,c in zip(m,cmap(np.linspace(0,1,10))):
#     p.add_mesh(pv.PolyData(i.ch_pos),color=c[:3], **kwargs)

# p.show()


def percentiles_to_scale(p0, p1, x0, x1):
    assert p1 >= p0 and x1 >= x0
    return (x1 - x0) / (scipy.stats.norm.ppf(p1) - scipy.stats.norm.ppf(p0))


def optimize_montage(subject_id):
    # lm_ch = dict(nasion=["43"], inion=["64", "51"]) # lpa=['70'], rpa=['71']
    # lm_ch = dict(nasion=["43"], inion=["64", '51'], lpa=['70', '68', '55'], rpa=['71', '60', '47'])
    lm_ch = dict(nasion=["43"], inion=["64"], lpa=["70"], rpa=["71"])
    which_angles = ("xz", "xz", "yz", "yz")
    tol_dist = 2.5
    tol_angle = 5
    # np.random.seed(int(subject_id))

    n_repeats = 10

    n_steps = 6
    dist_max = 10
    angle_max = 20
    percentiles = 0.025, 0.975  # 95 % change that the noise is not more extreme than...
    # noise = dict(dist=np.linspace(0, 5, n_steps), angle=np.linspace(0, 15, n_steps))
    noise_lims = dict(
        dist=np.arange(0, dist_max + 1, 2), angle=np.arange(0, angle_max + 1, 4)
    )
    noise_scales = {
        k: [percentiles_to_scale(*percentiles, -vv, vv) for vv in v]
        for k, v in noise_lims.items()
    }

    rng = np.random.default_rng()
    rng.normal(scale=10)

    model = "custom_nonlin"

    fname = get_montage_neighbors("easycap_BC_TMS64_X21")
    neighbors, ch_names = load_neighbors(fname)
    # we do not use the reference electrode here...
    neighbors = scipy.sparse.coo_matrix(neighbors.todense()[1:, 1:])
    ch_names = ch_names[1:]
    n_channels = len(ch_names)

    # info = create_info()
    # ch_names = info["ch_names"]
    # n_channels = len(ch_names)

    # ch_pos_2d = mne.channels.layout._find_topomap_coords(info, None)
    # tri = scipy.spatial.Delaunay(ch_pos_2d)
    # # scipy.spatial.delaunay_plot_2d(tri)
    # adj = get_adjacency_matrix(tri.simplices).tocoo()
    # neighbors = adj
    # neighbors = scipy.sparse.triu(adj)

    io = utils.SubjectIO(subject_id)

    dist_opt = np.zeros((n_steps, n_steps, n_repeats, n_channels))
    dist_ref = np.zeros(n_channels)

    opt_pos = np.zeros((n_steps, n_steps, n_repeats, n_channels, 3))
    ref_pos = np.zeros((n_channels, 3))

    print(f"Optimizing montage {model}")

    sub_dir = io.simnibs.get_path("subject")

    # simulate some measurements
    with open(Config.path.DATA / "prepare_phatmags.json", "r") as f:
        original_subject = json.load(f)[io.subject]["original_subject"]
    landmarks = getattr(MRIFiducials, original_subject)
    skin = pv.read(sub_dir / "skin_outer_annot.vtk")
    montage_ref = eeg.make_montage(
        sub_dir / f"montage_{Config.forward.REFERENCE}_proj.csv"
    )
    assert all(i == j for i, j in zip(ch_names, montage_ref.ch_names))

    surf = {"points": skin.points, "tris": skin.faces.reshape(-1, 4)[:, 1:]}
    subset = skin["outer_points"].nonzero()[0]

    # median/mean/std of abs difference between euclidean and geodesic
    # dists:
    #   inion, 64     : 1.2 / 1.4 / 0.9 mm
    #   nasion, 43    : 3.4 / 3.7 / 1.7 mm
    # for nasion all geodesic > euclidean
    # for inion most geodesic > euclidean (this has to do with the snapping
    # to closest node!)
    dist_dict, angle_dict = get_pseudo_measurements(
        montage_ref, landmarks, lm_ch, skin, which_angles
    )
    # subtract ~3.5 mm from geodesic distance between nasion and X

    montage = eeg.make_montage(sub_dir / f"montage_{model}_proj.csv")
    assert all(i == j for i, j in zip(ch_names, montage.ch_names))
    dist_ref = np.linalg.norm(montage.ch_pos - montage_ref.ch_pos, axis=1)

    ref_pos = montage_ref.ch_pos

    for di, nd in enumerate(noise_scales["dist"]):
        for ai, na in enumerate(noise_scales["angle"]):
            for r in range(n_repeats):
                # print(f"Using distance error {nd:.02f} and angular error {na:.02f}")
                noisy_dist_dict = add_measurement_noise_dist(dist_dict, nd, rng)
                noisy_angle_dict = add_measurement_noise_angle(angle_dict, na, rng)

                montage_opt = run_optimization(
                    montage,
                    noisy_dist_dict,
                    noisy_angle_dict,
                    landmarks,
                    neighbors,
                    tol_dist,
                    tol_angle,
                )
                montage_opt.project_to_surface(surf, subset)
                dist_opt[di, ai, r] = np.linalg.norm(
                    montage_opt.ch_pos - montage_ref.ch_pos, axis=1
                )
                opt_pos[di, ai, r] = montage_opt.ch_pos

                if dist_opt[di, ai, r].max() > 50:
                    print(f"max error is {dist_opt[di,ai].max()}")
                    print(f"for parameters dist={nd} and angle={na}")

                    print(f"DIST DICT : {dist_dict}")
                    print(f"ANGLE DICT: {angle_dict}")

                    print(f"NOISY DIST DICT : {noisy_dist_dict}")
                    print(f"NOISY ANGLE DICT: {noisy_angle_dict}")

    # p = pv.Plotter(notebook=False)
    # p.add_mesh(skin)
    # # p.add_mesh(pv.Sphere(radius[-1], center[-1]), opacity=0.75)
    # # p.add_mesh(pv.Sphere(radius[-2], center[-2]), opacity=0.75)
    # p.add_mesh(montage.ch_pos, color="magenta", render_points_as_spheres=True)
    # # p.add_point_labels(montage.ch_pos, labels=neighbor_ch_names)
    # # p.show()

    # p.add_mesh(ref_pos, color="red", render_points_as_spheres=True)
    # for po in pos:
    #     p.add_mesh(po, color="blue", render_points_as_spheres=True)
    # p.add_mesh(
    #     np.array(list(landmarks.values())),
    #     color="green",
    #     point_size=10,
    #     render_points_as_spheres=True,
    # )
    # p.show()

    # lm_pos_dist = np.array([landmarks[lm] for lm, _ in dist_dict])
    index = pd.Index(ch_names, name="Channel")
    # index = pd.Index(map(int, neighbor_ch_names), name="Channel")
    columns = pd.MultiIndex.from_product(
        (
            [io.subject],
            np.round(noise_lims["dist"], 2),
            np.round(noise_lims["angle"], 2),
            range(n_repeats),
        ),
        names=["Subject", "Distance Error", "Angular Error", "Repetition"],
    )
    df_opt = pd.DataFrame(
        np.ascontiguousarray(dist_opt.reshape(-1, n_channels).T),
        index=index,
        columns=columns,
    )
    columns_ref = pd.Index([io.subject], name="Subject")
    df_ref = pd.DataFrame(
        np.ascontiguousarray(dist_ref.T), index=index, columns=columns_ref
    )

    io.data.update(
        prefix="montage", stage="preprocessing", task=None, extension="pickle"
    )
    df_opt.to_pickle(io.data.get_filename(suffix="opt"))
    df_ref.to_pickle(io.data.get_filename(suffix="ref"))

    index_coo = pd.MultiIndex.from_product(
        (ch_names, ["x", "y", "z"]), names=["Channel", "Coordinates"]
    )
    df_opt_pos = pd.DataFrame(
        np.ascontiguousarray(opt_pos.reshape(-1, 3 * n_channels).T),
        index=index_coo,
        columns=columns,
    )
    df_ref_pos = pd.DataFrame(ref_pos.ravel(), index=index_coo, columns=columns_ref)

    io.data.update(
        prefix="montage", stage="preprocessing", task=None, extension="pickle"
    )
    df_opt_pos.to_pickle(io.data.get_filename(suffix="optcoords"))
    df_ref_pos.to_pickle(io.data.get_filename(suffix="refcoords"))


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    optimize_montage(subject_id)

    # x = np.linalg.norm(montage.ch_pos - montage_ref.ch_pos, axis=1)
    # y = np.linalg.norm(montage_opt.ch_pos - montage_ref.ch_pos, axis=1)
    # z = y - x
    # improved = z <= 0
    # print(f'N got better     : {improved.sum():10d}')
    # print(f'Mean(got better) : {z[improved].mean():10.3f}')
    # print(f'Mean(got worse)  : {z[~improved].mean():10.3f}')
    # print(f'Max(got better)  : {z[improved].min():10.3f}')
    # print(f'Max(got worse)   : {z[~improved].max():10.3f}')

    # print("Percentiles ([5, 25, 50, 75, 95])")
    # print(np.percentile(z, [5, 25, 50, 75, 95]))

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(*montage.ch_pos.T, marker="x")
    # ax.scatter(*montage_ref.ch_pos.T, marker="+")
    # ax.scatter(*montage_opt.ch_pos.T, marker="o")
    # ax.scatter(*lm_pos_dist.T, marker="d")

    # montage_opt.write(sub_dir / f"montage_{model}_opt.csv")

    # montage_opt.project_to_surface(surf, subset)
    # montage_opt.write(sub_dir / f"montage_{model}_opt_proj.csv")

    # refs.append(montage_ref.ch_pos)
    # before.append(montage.ch_pos)
    # after.append(montage_opt.ch_pos)

# refs = np.array(refs)
# before = np.array(before)
# after = np.array(after)

# np.save('/home/jesperdn/nobackup/data.npy', np.stack((refs, before, after)))
# refs, before, after = np.load('/home/jesperdn/nobackup/data.npy')


# x = np.linalg.norm(before - refs, axis=-1)
# y = np.linalg.norm(after - refs, axis=-1)
# z = y - x

# plt.figure(figsize=(12, 6))
# plt.violinplot(y.T)
# plt.grid(alpha=0.5)
# plt.ylim([0, 35])

# plt.figure(figsize=(20, 6))
# parts = plt.violinplot(z.T, showmedians=True)
# parts['cbars'].set_linewidth(1)
# parts['cmaxes'].set_linewidth(1)
# parts['cmins'].set_linewidth(1)
# #parts['cmedians'].set_color('orange')
# plt.grid(alpha=0.25)

# for cb in parts['cbars']:


# for pc in parts["bodies"]:

#     pc.set_facecolor('r')
#     pc.set_edgecolor(pc.get_facecolor())
#     pc.set_alpha(1)


# plt.figure()
# plt.hist(z.ravel(), "auto")
# plt.grid(alpha=0.5)

# a = np.array([neighbor_dist(r, neighbors) for r in refs])
# b = np.array([neighbor_dist(b, neighbors) for b in before])

# dist_norm = np.linalg.norm(a-b, axis=1) / a.shape[0]

# def plot_error_topomap(x, y):
#     kwargs = dict(vmin=0, cmap=None, contours=0, show=False)
#     x = np.atleast_2d(x)
#     y = np.atleast_2d(y)
#     vmax = x.mean(0).max()

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     im, _ = mne.viz.plot_topomap(x.mean(0), info, axes=axes[0], vmax=vmax, **kwargs,)
#     im, _ = mne.viz.plot_topomap(y.mean(0), info, axes=axes[1], vmax=vmax, **kwargs,)
#     cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.025)
#     cbar.set_label("mm")  # , rotation=-90)

#     fig, ax = plt.subplots(1, 1, figsize=(6, 5))
#     im, _ = mne.viz.plot_topomap(z.mean(0), info, axes=ax, cmap=None, contours=0, show=False)
#     cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.025)
#     cbar.set_label("mm")  # , rotation=-90)


#     z = np.atleast_2d(z)

#     fig, axes = plt.subplots(3, 5, figsize=(10, 7))
#     for ax,data in zip(axes.ravel(),z):
#     im, _ = mne.viz.plot_topomap(
#         z.mean(0), info, axes=ax, cmap=None, contours=0, show=False
#     )
#     cbar = fig.colorbar(im, ax=ax, shrink=1, pad=0.025)
#     cbar.set_label("mm")  # , rotation=-90)


# def plot_error_topomap_diff(z):
#     z = np.atleast_2d(z)

#     vmin,vmax = -10,10

#     subjects = list(io.subjects.keys())
#     fig, axes = plt.subplots(6, 6, figsize=(16,12))
#     for ax,data,sub in zip(axes.ravel(),z, subjects):
#         im, _ = mne.viz.plot_topomap(
#             data, info, axes=ax, vmin=vmin, vmax=vmax, cmap=None, contours=0, show=False
#         )
#         ax.set_title(f'{sub}')
#         if sub == '33':
#             break
#     for ax in axes:
#         cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.025)
#         #cbar.set_label("mm")  # , rotation=-90)


# def make_violin_plot(x, y):
#     w = np.stack((x, y))
#     plt.figure(figsize=(15, 5))
#     parts = plt.violinplot(
#         w[0].T, positions=np.arange(32) - 0.2, widths=0.3, showmeans=True
#     )
#     # for pc in parts["bodies"]:
#     # pc.set_facecolor('r')
#     # pc.set_edgecolor(pc.get_facecolor())
#     # pc.set_alpha(1)
#     parts = plt.violinplot(
#         w[1].T, positions=np.arange(32) + 0.2, widths=0.3, showmeans=True
#     )
#     # for pc in parts["bodies"]:
#     # pc.set_facecolor('b')
#     # pc.set_edgecolor(pc.get_facecolor())
#     plt.grid(True, alpha=0.25)


# plt.legend(['before', 'after'])

# def optimize_channel_positions(montage, meas, tol=0, fix_cz=False):
#     """

#     montage :
#         The nonlinearly deformed montage including fiducials.
#     meas :

#     tol :
#         Tolerance (in mm). lb/ub = d -/+ tol.

#     """
#     # Get some indices...
#     lm_names = list(montage.landmarks.keys())
#     lm_pos = np.array(list(montage.landmarks.values()))

#     ch_ind = [montage.ch_names.index(i) for i in meas["channels"]]
#     lm_ind = [lm_names.index(i) for i in meas["landmarks"]]

#     def neighbor_dist(x, neighbors):
#         """Distance between neighboring points"""
#         # return np.linalg.norm(x[neighbors.row] - x[neighbors.col], axis=1)
#         return scipy.spatial.distance.pdist(x)

#     def loss_fun(x, neighbors, nd0):
#         """Sum of squared residuals between current distances and the initial
#         ones."""
#         return np.sum((neighbor_dist(x, neighbors) - nd0) ** 2)

#     def constrained_dist(x):
#         """Constraint on distances between channels and landmarks"""
#         return np.linalg.norm(x[ch_ind] - lm_pos[lm_ind], axis=1)

#     constraints = [
#         scipy.optimize.NonlinearConstraint(
#             constrained_dist, meas["dist"] - tol, meas["dist"] + tol
#         ),
#     ]

#     if not fix_cz:
#         ind_cz = ch_names.index("Cz")

#         def constrained_cz(x):
#             """Constraint on Cz position."""
#             return np.linalg.norm(x[ind_cz] - x0[ind_cz])

#         constraints.append(scipy.optimize.NonlinearConstraint(constrained_cz, 0, 0))

#     x0 = montage.ch_pos
#     nd0 = neighbor_dist(x0)
#     res = scipy.optimize.minimize(
#         loss_fun,
#         x0,
#         method="trust-constr",
#         constraints=constraints,
#         args=(neighbors, nd0),
#     )
#     # dist = np.linalg.norm(res.x - montage.ch_pos, axis=1)
#     # (should be projected on to the scalp at some point maybe..?)

#     X = np.random.random((10, 3))  # point + its neighbors...
#     _, _, Vh = np.linalg.svd(X - X.mean(0), full_matrices=False)
#     basis = Vh.T[:2]  # basis for plane
#     # Vh.T[-1] # normal for plane

#     x0

#     Rx0 = R @ x0
#     # R = (3,3)
#     # x0 = (n,3)
#     # basis = (n,2,3)
#     # coefs = (n,2)
#     ch_pos0  # starting ch_pos
#     x0 = np.zeros((n, 2))

#     basis = []
#     for i in range(n_chs):
#         neighbors.row
#         neighbors.col
#         X = np.random.random((10, 3))  # point + its neighbors...
#         _, _, Vh = np.linalg.svd(X - X.mean(0), full_matrices=False)
#         basis.append(Vh.T[:2])  # basis for plane

#     x = np.sum(coefs * basis, axis=1) + ch_pos0

#     montage_opt = copy.copy(montage)
#     montage_opt.ch_pos = res.x

#     return montage_opt


# # The measured distances for (channel, landmark) pairs
# meas = {
#     "channel": ["1", "2", "30", "61"],
#     "landmark": ["Nz", "LPA", "RPA", "Iz"],
#     "dist": [1, 2, 3, 1],
# }


# neighbors, ch_names = load_neighbors(
#     r"C:\Users\jdue\Desktop\easycap_m10_neighbors.json"
# )


# neighbors = {
#     1: [2, 3, 4, 5, 6, 7],
#     2: [1, 3, 7, 8, 9, 19],
#     3: [1, 2, 4, 9, 10, 11],
#     4: [1, 3, 5, 11, 12, 13],
#     5: [1, 4, 6, 13, 14, 15],
#     6: [1, 5, 7, 15, 16, 17],
#     7: [1, 2, 6, 17, 18, 19],
#     8: [2, 9, 19, 20, 21, 34],
#     9: [2, 3, 8, 10, 21, 22],
#     10: [3, 9, 11, 22, 23],
#     11: [3, 4, 10, 12, 23, 24],
#     12: [4, 11, 13, 25],
#     13: [4, 5, 12, 14, 26, 27],
#     14: [5, 13, 15, 27, 28],
#     15: [5, 6, 14, 16, 28, 29],
#     16: [6, 15, 17, 30],
#     17: [6, 7, 16, 18, 31, 32],
#     18: [7, 17, 19, 32, 33],
#     19: [2, 7, 8, 18, 33, 34],
#     20: [8, 21, 34, 35, 36, 50],
#     21: [8, 9, 22, 37, 36, 20] + [35],
#     22: [9, 21, 37, 38, 23, 10],
#     23: [10, 22, 38, 39, 24, 11],
#     24: [11, 23, 39, 40, 25],
#     25: [12, 24, 40, 41, 26],
#     26: [13, 25, 41, 42, 27],
#     27: [14, 13, 26, 42, 43, 28],
#     28: [15, 14, 27, 43, 44, 29],
#     29: [15, 28, 44, 45, 30],
#     30: [46, 31, 16, 29, 45],
#     31: [17, 30, 32, 46, 47],
#     32: [17, 18, 31, 33, 47, 48],
#     33: [18, 19, 32, 34, 48, 49],
#     34: [8, 19, 20, 33, 49, 50] + [35],
#     35: [20, 36, 50] + [21, 34],
#     36: [20, 21, 35, 37],
#     37: [21, 22, 36, 38, 51],
#     38: [22, 23, 37, 39, 51, 52],
#     39: [23, 24, 38, 40, 52, 53],
#     40: [24, 25, 39, 41, 53, 54],
#     41: [25, 26, 40, 42, 54] + [55],
#     42: [26, 27, 41, 43, 55] + [54, 56],
#     43: [27, 28, 42, 44, 56] + [55, 57],
#     44: [28, 29, 43, 45, 57] + [56, 58],
#     45: [29, 30, 44, 46, 58] + [57],
#     46: [30, 31, 45, 47, 58, 59],
#     47: [31, 32, 46, 48, 59, 60],
#     48: [32, 33, 47, 49, 60, 61],
#     49: [33, 34, 48, 50, 61],
#     50: [20, 34, 35, 49],
#     51: [37, 38, 52],
#     52: [38, 39, 51, 53],
#     53: [39, 40, 52, 54],
#     54: [40, 41, 53, 55] + [42],
#     55: [42, 54, 56] + [41, 43],
#     56: [43, 55, 57] + [42, 44],
#     57: [44, 56, 58] + [43, 45],
#     58: [45, 46, 57, 59] + [44],
#     59: [46, 47, 58, 60],
#     60: [47, 48, 59, 61],
#     61: [48, 49, 60],
# }
# neighbors = {str(k): [str(i) for i in sorted(v)] for k, v in neighbors.items()}
# with open(r"C:\Users\jdue\Desktop\easycap_m10_neighbors.json", "w") as f:
#     json.dump(neighbors, f)
