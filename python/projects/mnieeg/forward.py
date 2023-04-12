import json
import os
from pathlib import Path
import shutil
import warnings

import mne
from mne.io.constants import FIFF
from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees

# from mne.io._digitization import _coord_frame_const
from mne.io.constants import FIFF


import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial import distance
import scipy.stats

# import scipy.io

from projects.mnieeg import utils
from projects.mnieeg.config import Config

from simnibs.mesh_tools import mesh_io
from simnibs.segmentation import brain_surface
from simnibs.simulation import cond, eeg, eeg_mne_tools
from simnibs.utils import file_finder

LANDMARKS_BIDS_TO_MNE = {"NAS": "nasion", "LPA": "lpa", "RPA": "rpa"}


# # The measured distances for (channel, landmark) pairs
# meas = {
#     "channel": ["1", "2", "30", "61"],
#     "landmark": ["Nz", "LPA", "RPA", "Iz"],
#     "dist": [1, 2, 3, 1],
# }


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

import scipy.optimize
import scipy.sparse


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
    nn = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n, n))

    return nn, ch_names


# neighbors, ch_names = load_neighbors(
#     r"C:\Users\jdue\Desktop\easycap_m10_neighbors.json"
# )


# def optimize_positions(montage, meas, tol=0, fix_cz=False):
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


def project_to_mni_surface(points):
    # MNI152 surface
    surf = pv.read(Path(__file__).parent / "mni152_surface.vtk")
    surf = {"points": surf.points, "tris": surf.faces.reshape(-1, 4)[:, 1:]}
    pttris = eeg.get_nearest_triangles_on_surface(points, surf, 2)
    _, _, projs, _ = eeg.project_points_to_surface(points, surf, pttris)
    return projs


def prepare_head_points(io):
    """
    - Extract skin surface from CHARM head mesh
    - Select head points on this surface to use when warping the MNI152
      template with Brainstorm

    """
    headpoints_per_ch = 4

    # Get affine transformation from MRI to MNI space by matching fiducials
    info = mne.io.read_info(
        io.data.get_filename(stage="preprocessing", forward=None, suffix="eeg")
    )
    mri_head_t = mne.read_trans(
        io.data.get_filename(stage=None, forward=None, suffix="trans")
    )
    head_mri_t = mne.transforms.invert_transform(mri_head_t)
    montage = eeg.Montage()
    montage.add_landmarks()
    montage = eeg_mne_tools.simnibs_montage_to_mne_montage(montage, "mri")
    # trans is actually head-to-mni so we need to transform it to mri-to-mni
    trans = mne_coregister_fiducials(info, montage.dig, scale=True)
    trans["trans"] = mri_head_t["trans"] @ trans["trans"]
    trans["from"] = mne.transforms._to_const("unknown")
    mri2mni = trans
    mni2mri = mne.transforms.invert_transform(mri2mni)

    ch_pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    ch_pos = mne.transforms.apply_trans(head_mri_t, ch_pos)

    skin = pv.read(io.simnibs.get_path("subject") / "skin_outer_annot.vtk")

    # query nearest neighbors of each channel
    tree = cKDTree(
        1e-3 * skin.points[skin["outer_points"].astype(bool)]
    )  # convert to m to ch_pos from Info
    _, idx = tree.query(ch_pos, headpoints_per_ch)
    head_pts = tree.data[idx.ravel()]
    head_pts *= 1e3  # * mne.transforms.apply_trans(mri2mni, head_pts)

    subject_dir = io.simnibs_template.get_path("subject")
    if not subject_dir.exists():
        subject_dir.mkdir(parents=True)
    np.savetxt(subject_dir / "headpoints.csv", head_pts)
    mne.write_trans(subject_dir / "mni2mri-trans.fif", mni2mri)
    mne.write_trans(subject_dir / "mri2mni-trans.fif", mri2mni)

    return head_pts, mri2mni, mni2mri


# def prepare_head_points(subject_id):
#     """
#     - Extract skin surface from CHARM head mesh
#     - Select head points on this surface to use when warping the MNI152
#       template with Brainstorm

#     """
#     headpoints_per_ch = 4

#     io = utils.SubjectIO(subject_id)
#     io.filenamer.update(session=io.session, stage="preprocessing", suffix="eeg")
#     info = mne.io.read_info(io.filenamer.get_filename())
#     # No need to transform as these are already in MRI coordinates
#     ch_pos = np.array([ch["loc"][:3] for ch in info["chs"]])

#     tag = 1005  # skin

#     mesh = pv.read(io.simnibs.match("sub*.msh"))
#     skin = mesh.extract_cells(mesh["gmsh:geometrical"] == tag)
#     skin.clear_data()
#     skin.save(io.simnibs.get_path("subject") / "skin.vtk")

#     # query nearest neighbors of each channel
#     tree = cKDTree(1e-3 * skin.points)  # convert to m to ch_pos from Info
#     _, idx = tree.query(ch_pos, headpoints_per_ch)
#     arr = tree.data[idx.ravel()]

#     # Get affine transformation from MRI to MNI space by matching fiducials
#     montage = eeg.Montage()
#     montage.add_landmarks()
#     montage = eeg_mne_tools.simnibs_montage_to_mne_montage(montage)
#     mni2mri = _match_fiducials(montage, io.bids)
#     mri2mni = mne.transforms.invert_transform(mni2mri)

#     # trans = mne.transforms.invert_transform(charm_affine(io.simnibs.get_path("m2m")))

#     # Convert from mm to m for MNE and to cm for Brainstorm
#     # (Brainstorm seems to expect coordinates in ASCII XYZ format to be in cm)
#     arr = 1e3 * mne.transforms.apply_trans(mri2mni, arr)

#     subject_dir = io.subject_dir.get_path("subject")
#     if not subject_dir.exists():
#         subject_dir.mkdir(parents=True)
#     np.savetxt(subject_dir / "headpoints.csv", arr)
#     mne.write_trans(subject_dir / "mni2mri-trans.fif", mni2mri)
#     mne.write_trans(subject_dir / "mri2mni-trans.fif", mri2mni)


# def charm_get_trans(m2m_dir):
#     # `worldToWorldTransformMatrix` is MNI (head) to subject MRI (mri)
#     seg = m2m_dir / "segmentation"
#     mat = scipy.io.loadmat(seg / "coregistrationMatrices.mat")
#     trans = mat["worldToWorldTransformMatrix"]
#     trans[:3, 3] *= 1e-3  # mm to m
#     return mne.transforms.Transform("head", "mri", trans)

# def get_mri_landmarks(bids_path):
#     t1w = nib.load(mne_bids.path._find_matching_sidecar(bids_path, "T1w", ".nii.gz"))
#     fname = mne_bids.path._find_matching_sidecar(bids_path, "T1w", ".json")
#     with open(fname, "r", encoding="utf-8") as f:
#         landmarks = json.load(f).get("AnatomicalLandmarkCoordinates", {})
#     # Convert to world coordinates and m
#     return {
#         LANDMARKS_BIDS_TO_MNE[k]: 1e-3 * (v @ t1w.affine[:3, :3].T + t1w.affine[:3, 3])
#         for k, v in landmarks.items()
#     }

# def _match_fiducials(montage, bids_path, scale=True):
#     """Match fiducials from an MNE-Python montage object to those in the
#     sidecar file of the T1w image. This provides a mapping from head
#     coordinates (EEG space) to MRI coordinates (subject MRI space).
#     """
#     LANDMARK_ORDER = ("lpa", "nasion", "rpa")

#     # Get EEG landmarks (i.e., from the standard montage)
#     eeg_coords_dict = mne_bids.utils._extract_landmarks(montage.dig)
#     eeg_coords_dict = {LANDMARKS_BIDS_TO_MNE[k]: v for k, v in eeg_coords_dict.items()}
#     eeg_landmarks = np.array([eeg_coords_dict[f] for f in LANDMARK_ORDER])

#     # Get MRI landmarks (from T1w sidecar)
#     mri_landmarks = get_mri_landmarks(bids_path)
#     mri_landmarks = np.array([mri_landmarks[f] for f in LANDMARK_ORDER])

#     # Fit the transform (allow scaling as this is a template match)
#     trans = mne.coreg.fit_matched_points(
#         src_pts=eeg_landmarks, tgt_pts=mri_landmarks, scale=scale
#     )

#     return mne.Transform("head", "mri", trans)


def get_charm_affine(m2m_dir):
    """Affine transformation from MNI to subject MRI coordinates."""
    template = nib.load(
        Path(file_finder.templates.charm_atlas_path)
        / "charm_atlas_mni"
        / "template.nii"
    )
    template_coreg = nib.load(m2m_dir / "segmentation" / "template_coregistered.nii.gz")
    trans = template_coreg.affine @ np.linalg.inv(template.affine)
    trans[:3, 3] *= 1e-3  # mm to m
    return mne.transforms.Transform("unknown", "mri", trans)


def mne_montage_remove_fiducials(montage):
    """This is only available on the github master branch..."""
    for d in montage.dig.copy():
        if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            montage.dig.remove(d)
    return montage


def mne_montage_apply_trans(montage, trans):
    """This is only available on the github master branch..."""
    coord_frame = montage.get_positions()["coord_frame"]
    trans = mne.transforms._ensure_trans(trans, fro=coord_frame, to=trans["to"])
    for d in montage.dig:
        d["r"] = mne.transforms.apply_trans(trans, d["r"])
        d["coord_frame"] = trans["to"]


def mne_coregister_fiducials(info, fiducials, scale=False, tol=0.01):
    """Adapted from mne.coreg.coregister_fiducials to allow for scaling."""
    coord_frame_to = FIFF.FIFFV_COORD_MRI
    frames_from = {d["coord_frame"] for d in info["dig"]}
    if len(frames_from) > 1:
        raise ValueError("info contains fiducials from different coordinate " "frames")
    else:
        coord_frame_from = frames_from.pop()
    coords_from = mne.viz._3d._fiducial_coords(info["dig"])
    coords_to = mne.viz._3d._fiducial_coords(fiducials, coord_frame_to)
    trans = mne.coreg.fit_matched_points(coords_from, coords_to, scale=scale, tol=tol)
    return mne.Transform(coord_frame_from, coord_frame_to, trans)


def mne_montage_to_head_frame(montage, info, scale=False, tol=0.01):
    """Transform an MNE montage from MRI to head coordinate frame using
    landmarks specified in the montage and Info.
    """
    mne_montage_apply_trans(
        montage,
        mne.transforms.invert_transform(
            mne_coregister_fiducials(info, montage.dig, scale, tol)
        ),
    )


def tps_deform_estimate(pts_from, pts_to):
    """Estimate a transformation taking `pts_from` to `pts_to`. The
    transformation consists of two parts (n = number of points)

    Thin-plate splines.

    (1) a nonlinear deformation [:n]
    (2) an affine transformation (the last four rows) where [n:-1] rotates and
        scales and [-1] translates.

    Implementation of `warp_transform` from `bst_warp` in Brainstorm with the
    modification that we fit `pts_to` and not the difference between `pts_to`\
    and `pts_from`.

    RETURNS
    -------
    deform :
        nonlinear, affine

    """
    n, d = pts_from.shape
    m = n + d + 1

    # Left-hand side (kernel and homogeneous coordinates of `pts_from`)
    A = np.zeros((m, m))
    A[:n, :n] = distance.squareform(distance.pdist(pts_from))
    A[:n, n:-1] = pts_from
    A[n:-1, :n] = pts_from.T
    A[:n, -1] = A[-1, :n] = np.ones(n)

    # Right-hand side (the target points)
    D = np.zeros((m, d))
    D[:n] = pts_to

    return np.linalg.solve(A, D)


def tps_deform_apply(deform, pts_from, pts):
    """Apply nonlinear warp and

    Implementation of `warp_lm` from `bst_warp` in Brainstorm (with the
    modification that it adds `pts` as well).

    """
    n, d = pts_from.shape
    nonlin, affine = deform[:n], deform[n:]
    D = distance.cdist(pts, pts_from)
    return pts @ affine[:d, :d] + affine[-1] + D @ nonlin


# def get_outer_surface_points(m, tol: float = 1e-5):
#     """

#     m : mesh
#     tol : float

#     """
#     idx, pos = m.intersect_ray(m.nodes.node_coord, m.nodes_normals().value)

#     # Get the self-intersections (i.e., those which are with a point's
#     # own triangles) and compare with its the total number of intersections
#     n_intersect = np.bincount(idx[:, 0])
#     is_self_intersect = (
#         np.linalg.norm(pos - np.repeat(m.nodes.node_coord, n_intersect, axis=0), axis=1)
#         < tol
#     )
#     return np.where(
#         np.bincount(idx[:, 0], is_self_intersect.astype(int)) == n_intersect
#     )[0]


def get_outer_surface_points(m, tol: float = 1e-3):
    """Return indices of points estimated to be on the outer surface."""
    # Avoid self-intersections by moving each point slightly along the test
    # direction
    n = m.nodes_normals().value
    idx = np.unique(m.intersect_ray(m.nodes.node_coord + tol * n, n)[0][:, 0])
    return np.setdiff1d(np.arange(m.nodes.nr), idx, assume_unique=True)


def mesh_to_volume(m, subfiles, subfiles_template):
    # Modified from charm_main
    mesh = m.crop_mesh(elm_type=4)
    ed = mesh_io.ElementData(mesh.elm.tag1.astype(np.uint16))
    ed.mesh = mesh
    ed.to_deformed_grid(
        subfiles.mni2conf_nonl,
        file_finder.Templates().mni_volume,
        out_original=subfiles_template.final_labels,
        method="assign",
        reference_original=subfiles.reference_volume,
    )

    fn_LUT = subfiles_template.final_labels.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(file_finder.templates.final_tissues_LUT, fn_LUT)


def _init_template_nonlin(io):
    """Creates a head model adapted slightly to the individual anatomy by
    warping the MNI152 head model. Performs the following steps

    1. Get head points (chosen as points close to each electrode)
    2. Apply affine transform to MNI152 head model
    3. Project head points onto this surface
    4. Estimate nonlinear transform bringing the projected head points to the
        original positions
    5. Apply this transform to the head model
    6. Apply affine and nonlinear transform to central cortical surface.
    """

    iomni = utils.SubjectIO("mni152", raise_on_exclude=False)

    m2m = io.simnibs.get_path("m2m")
    m2m_template = io.simnibs_template.get_path("m2m")
    m2m_mni = iomni.simnibs.get_path("m2m")
    surface_template = m2m_template / "surfaces"
    if not surface_template.exists():
        surface_template.mkdir(parents=True)

    subfiles = file_finder.SubjectFiles(subpath=str(m2m))
    subfiles_template = file_finder.SubjectFiles(subpath=str(m2m_template))
    subfiles_mni = file_finder.SubjectFiles(subpath=str(m2m_mni))

    # prepare head points and affine transforms between mri and mni spaces
    head_pts, _, mni2mri = prepare_head_points(io)

    # Get MNI152 head model and apply affine transform to subject space
    m = mesh_io.read_msh(m2m_mni / f"sub-{iomni.subject}.msh")
    m.nodes.node_coord = 1e3 * mne.transforms.apply_trans(
        mni2mri, 1e-3 * m.nodes.node_coord
    )

    # Get outer surface (skin)
    skin = m.crop_mesh(1005)  # crop makes a copy
    subset = get_outer_surface_points(skin)
    surf = {
        "points": skin.nodes.node_coord,
        "tris": skin.elm.node_number_list[:, :3] - 1,
    }

    # Project head point on skin surface
    montage = eeg.Montage(headpoints=head_pts.copy())
    montage.project_to_surface(surf, subset)
    head_pts_proj = montage.headpoints
    _, idx = np.unique(head_pts_proj, return_index=True, axis=0)
    if (nhpp := len(idx)) < (nhp := len(head_pts)):
        warnings.warn(
            "Some points were projected to the same nodes on the skin surface. "
            + f"Reducing the number of head points from {nhp} to {nhpp}."
        )
        head_pts = head_pts[idx]
        head_pts_proj = head_pts_proj[idx]

    # Estimate the nonlinear transformation by registering the projected
    # head points to the original ones and apply it to the head model
    warp = tps_deform_estimate(head_pts_proj, head_pts)
    m.nodes.node_coord = tps_deform_apply(warp, head_pts_proj, m.nodes.node_coord)
    m.write(str(m2m_template / f"sub-{io.subject}.msh"))
    mesh_to_volume(m, subfiles, subfiles_template)

    # Also transform the (central) cortical surfaces (the spherical
    # registrations are unaffected so just copy them)
    surfs = eeg.load_surface(subfiles_mni, "central")
    for hemi, surf in surfs.items():
        pts = 1e3 * mne.transforms.apply_trans(mni2mri, 1e-3 * surf["points"])
        pts = tps_deform_apply(warp, head_pts_proj, pts)
        darrays = (
            nib.gifti.gifti.GiftiDataArray(pts, "pointset"),
            nib.gifti.gifti.GiftiDataArray(surf["tris"], "triangle"),
        )
        subsamp = nib.GiftiImage(darrays=darrays)
        subsamp.to_filename(surface_template / f"{hemi}.central.gii")

        shutil.copy(
            subfiles_mni.get_surface(hemi, "sphere_reg"),
            surface_template / f"{hemi}.sphere.reg.gii",
        )


def prepare_for_forward_init(io, model):
    """Create info object (containing electrode positions in head coordinates),
    transformation (between head and mri spaces), and CSV cap file for SimNIBS.

    What head space is depends on the particular model, however, mri space is
    always subject MRI space (except for the model which uses template
    anatomy).

    Apply the head -> mri transform to bring all montages to mri space and set
    the transform to identity to avoid any strange things happening later in
    the analysis.
    """
    info = mne.io.read_info(io.data.get_filename(stage="preprocessing", suffix="eeg"))
    mri_head_t = mne.read_trans(io.data.get_filename(stage=None, suffix="trans"))

    montage_out_path = io.simnibs.get_path("subject")

    if model == "digitized":
        # These are the positions already present in Info
        pass

    elif model == "template_nonlin":
        # Warp the MNI head model to fit the invidivual skin surface. The
        # positions are those already prsent in Info
        _init_template_nonlin(io)
        # The simulation is performed in another directory
        montage_out_path = io.simnibs_template.get_path("subject")

    elif model.startswith("custom"):
        # Use our custom positions sampled in MNI space
        montage = eeg.make_montage(Config.forward.MONTAGE_SIMNIBS)
        montage.add_landmarks()

        m2m = io.simnibs.get_path("m2m")
        if model.endswith("affine_mri"):
            # Affine transformation from CHARM (MNI -> MRI) and then mri_head_t
            montage = eeg_mne_tools.simnibs_montage_to_mne_montage(montage)
            mne_montage_apply_trans(montage, get_charm_affine(m2m))
            mne_montage_apply_trans(montage, mri_head_t)

        elif model.endswith("affine_lm"):
            # Affine transformation of custom MNI coordinates based on
            # landmark coregistration (MNI -> head)
            # We set the coord_frame of `montage` to `mri` for
            # `coregister_fiducials` (which creates a head-to-mri trans) to
            # work
            montage = eeg_mne_tools.simnibs_montage_to_mne_montage(montage, "mri")
            mne_montage_to_head_frame(montage, info, scale=True)

        elif model.endswith("nonlin"):
            # Warp positions from MNI to MRI and then mri_head_t
            #
            # Conform2MNI_nonl.nii.gz
            #     Forward deformation field (y_) converts from MRI to MNI space.
            #     It gives, for each voxel, its corresponding MNI coordinates.
            #     (Same shape and affine as the T1 from which the transform was
            #     derived.)
            # MNI2Conform_nonl.nii.gz
            #     Inverse deformation field (iy_) converts from MNI to MRI space.
            #     It gives, for each voxel, its corresponding MRI coordinates.
            #     (Same shape and affine as the MNI template from which transform
            #     was derived.)

            mni2mri = nib.load(m2m / "toMNI" / "MNI2Conform_nonl.nii.gz")
            montage.apply_deform(mni2mri)
            montage = eeg_mne_tools.simnibs_montage_to_mne_montage(
                montage, coord_frame="mri"
            )
            mne_montage_apply_trans(montage, mri_head_t)
        else:
            raise ValueError

        # montage should be in head coords
        # mne_montage_remove_fiducials(montage)
        info.set_montage(montage)

    elif model.startswith("manufacturer"):
        # Use manufacturer layout (idealized positions on a sphere)
        fname = Config.path.RESOURCES / (Config.forward.MONTAGE_MNE + ".txt")
        montage = _read_theta_phi_in_degrees(
            fname, mne.defaults.HEAD_SIZE_DEFAULT, add_fiducials=True
        )
        for d in montage.dig:
            d["coord_frame"] = FIFF.FIFFV_COORD_MRI

        if model.endswith("affine_lm"):
            # Affine transformation of manufacturer coordinates based on
            # landmark coregistration (MNI -> head)
            # Increase `tol` as this will not work otherwise
            mne_montage_to_head_frame(montage, info, scale=True, tol=0.1)

        # montage should be in head coords
        # mne_montage_remove_fiducials(montage)
        info.set_montage(montage)

    # Info with updated channel positions
    mne.io.write_info(io.data.get_filename(forward=model, suffix="info"), info)

    fname = montage_out_path / f"montage_{model}.csv"
    eeg_mne_tools.prepare_montage(fname, info, mri_head_t)

    return fname


def prepare_for_forward(subject_id):
    """Prepare info, trans, and cap definition file for SimNIBS for all
    relevant forward models. Also, create a projected cap file.

    This does most of the heavy lifting in terms of being able to create each
    forward model.
    """
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    io.data.path.ensure_exists()

    # Find the vertices on the scalp to use for projection
    m = mesh_io.read_msh(io.simnibs.get_path("m2m") / f"sub-{io.subject}.msh")
    m = m.crop_mesh(1005)
    surf = {"points": m.nodes.node_coord, "tris": m.elm.node_number_list[:, :3] - 1}
    subset = get_outer_surface_points(m)

    skin = pv.make_tri_mesh(surf["points"], surf["tris"])
    skin["outer_points"] = np.zeros(skin.n_points, dtype=int)
    skin["outer_points"][subset] = 1
    skin.save(io.simnibs.get_path("subject") / "skin_outer_annot.vtk")

    print("Preparing for forward calculations")
    montages = pv.MultiBlock()
    # montages['skin'] = skin
    for fm in Config.forward.ALL_MODELS:
        print("Model :", fm)
        fname = prepare_for_forward_init(io, fm)

        montage = eeg.make_montage(fname)
        montages[fm] = montage_to_polydata(montage)
        montage.project_to_surface(surf, subset)
        montages[fm + "_proj"] = montage_to_polydata(montage)
        montage.write(fname.with_name(fname.stem + "_proj" + fname.suffix))
    montages.save(io.simnibs.get_path("subject") / "montages.vtm")

    print("Subsampling each hemisphere to", Config.forward.SUBSAMPLING, "nodes")
    brain_surface.subsample_surfaces(
        io.simnibs.get_path("m2m"), Config.forward.SUBSAMPLING
    )
    brain_surface.subsample_surfaces(
        io.simnibs_template.get_path("m2m"), Config.forward.SUBSAMPLING
    )


def montage_to_polydata(montage):
    points = montage.ch_pos
    if montage.landmarks:
        points = np.row_stack((points, montage.get_landmark_pos()))
    return pv.PolyData(points)


def compute_forward(subject_id):
    """Run a forward simulation, prepare it for use with MNE-Python, and
    construct source space and source morph.
    """
    io = utils.SubjectIO(subject_id)

    for fm in Config.forward.MODELS:
        print(f"Running simulation for model {fm}")
        if "template" in fm:
            subject_dir = io.simnibs_template.get_path("subject")
            m2m_dir = io.simnibs_template.get_path("m2m")
        else:
            subject_dir = io.simnibs.get_path("subject")
            m2m_dir = io.simnibs.get_path("m2m")

        fem_dir = subject_dir / f"fem_{fm}"
        montage = subject_dir / f"montage_{fm}_proj.csv"  # f"montage_{fm}_proj.csv"

        _ = eeg.compute_leadfield(
            m2m_dir, fem_dir, montage, subsampling=Config.forward.SUBSAMPLING
        )


def sample_conductivities(seed=None):
    # All values from Saturnino (2019)
    cond_limits = dict(
        white_matter=(0.1, 0.4),
        gray_matter=(0.1, 0.6),
        csf=(1.2, 1.8),
        spongy_bone=(0.015, 0.040),
        compact_bone=(0.003, 0.012),
        skin=(0.2, 0.5),
    )
    tissue_to_index = dict(
        white_matter=0, gray_matter=1, csf=2, spongy_bone=7, compact_bone=6, skin=4,
    )
    s = cond.standard_cond()
    header = "{:20s} {:16s}".format("Tissue", "Sample (default)")
    print(header)
    print("-" * len(header))
    samples = {}
    for tissue in tissue_to_index:
        cmin, cmax = cond_limits[tissue]
        beta = scipy.stats.beta(a=3, b=3, loc=cmin, scale=cmax - cmin)
        sample = beta.rvs(random_state=seed)
        default = s[tissue_to_index[tissue]].value
        s[tissue_to_index[tissue]].value = sample
        print(f"{tissue:20s} {sample:.4f} ({default:.4f})")
        samples[tissue] = sample
    return s, samples


def compute_forward_sample_cond(subject_id):

    io = utils.SubjectIO(subject_id)
    subject_dir = io.simnibs.get_path("subject")
    m2m_dir = io.simnibs.get_path("m2m")

    fm = Config.forward.REFERENCE
    fem_dir = subject_dir / f"fem_{fm}_sample_cond"
    montage = subject_dir / f"montage_{fm}_proj.csv"  # f"montage_{fm}_proj.csv"

    seed = int(subject_id)
    conds, samples = sample_conductivities(seed)
    with open(subject_dir / "sample_cond.json", "w") as f:
        json.dump(samples, f, indent=4)

    _ = eeg.compute_leadfield(
        m2m_dir,
        fem_dir,
        montage,
        subsampling=Config.forward.SUBSAMPLING,
        init_kwargs=dict(cond=conds),
    )


def prepare_for_inverse(subject_id):
    io = utils.SubjectIO(subject_id)
    mri_head_t = io.data.get_filename(suffix="trans")
    io.data.update(stage="forward")

    # We pretend that fsaverage is in head coords
    # (on reading or writing the forward solution, MNE will apply the
    # mri_head_t to the source space to bring it to head coords)
    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_central = fsavg.get_central_surface()
    fsavg_central = eeg_mne_tools.make_source_spaces(fsavg_central, fsavg.name, "head")

    out_format = "mne"
    for fm in Config.forward.MODELS:
        print(f"Preparing model {fm}")
        if "template" in fm:
            subject_dir = io.simnibs_template.get_path("subject")
            m2m_dir = io.simnibs_template.get_path("m2m")
        else:
            subject_dir = io.simnibs.get_path("subject")
            m2m_dir = io.simnibs.get_path("m2m")
        # _proj
        leadfield = (
            subject_dir
            / f"fem_{fm}"
            / f"sub-{io.subject}_leadfield_montage_{fm}_proj.hdf5"
        )

        io.data.update(forward=fm)
        info = io.data.get_filename(suffix="info")

        src, forward, morph = eeg.prepare_for_inverse(
            m2m_dir, leadfield, out_format, info, mri_head_t, Config.inverse.FSAVERAGE
        )

        # `src` and `morph` will mostly be duplicates (except for the template
        # warp) but we just do this for convenience
        src.save(io.data.get_filename(suffix="src"), overwrite=True)
        mne.write_forward_solution(
            io.data.get_filename(suffix="fwd"), forward, overwrite=True
        )
        morph.save(io.data.get_filename(suffix="morph", extension="h5"), overwrite=True)

        # morph forward solution to fsaverage for comparison between models
        # (template_nonlin is another source space than the others)
        morph_forward(forward, morph, fsavg_central)
        mne.write_forward_solution(
            io.data.get_filename(space="fsaverage", suffix="fwd"),
            forward,
            overwrite=True,
        )


def prepare_for_inverse_sampled_cond(subject_id):
    io = utils.SubjectIO(subject_id)
    mri_head_t = io.data.get_filename(suffix="trans")
    io.data.update(stage="forward")

    # We pretend that fsaverage is in head coords
    # (on reading or writing the forward solution, MNE will apply the
    # mri_head_t to the source space to bring it to head coords)
    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_central = fsavg.get_central_surface()
    fsavg_central = eeg_mne_tools.make_source_spaces(fsavg_central, fsavg.name, "head")

    out_format = "mne"
    fm_base = Config.forward.REFERENCE
    fm = fm_base + "_sample_cond"

    subject_dir = io.simnibs.get_path("subject")
    m2m_dir = io.simnibs.get_path("m2m")

    # _proj
    leadfield = (
        subject_dir
        / f"fem_{fm}"
        / f"sub-{io.subject}_leadfield_montage_{fm_base}_proj.hdf5"
    )

    io.data.update(forward=fm)
    info = io.data.get_filename(forward=fm_base, suffix="info")

    src, forward, morph = eeg.prepare_for_inverse(
        m2m_dir, leadfield, out_format, info, mri_head_t, Config.inverse.FSAVERAGE
    )

    # `src` and `morph` will mostly be duplicates (except for the template
    # warp) but we just do this for convenience
    src.save(io.data.get_filename(suffix="src"), overwrite=True)
    mne.write_forward_solution(
        io.data.get_filename(suffix="fwd"), forward, overwrite=True
    )
    morph.save(io.data.get_filename(suffix="morph", extension="h5"), overwrite=True)

    # morph forward solution to fsaverage for comparison between models
    # (template_nonlin is another source space than the others)
    morph_forward(forward, morph, fsavg_central)
    mne.write_forward_solution(
        io.data.get_filename(space="fsaverage", suffix="fwd"), forward, overwrite=True,
    )


def morph_forward(fwd, morph, src):
    """Morph forward solution to the space defined by the output of `morph`
    (correspondong to `src`).
    """
    assert morph.subject_to == src[0]["subject_his_id"]

    n = fwd["nchan"]
    m = fwd["nsource"]
    if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
        p = 3
    elif fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
        p = 1
    else:
        raise ValueError("Unknown source orientation")

    # head_mri_t = mne.transforms.invert_transform(fwd['mri_head_t'])['trans']
    data = fwd["sol"]["data"].reshape(n, m, p)  # @ head_mri_t[:3,:3].T
    data = np.array([morph.morph_mat @ r for r in data]).reshape(n, -1)

    fwd["nsource"] = morph.morph_mat.shape[0]
    fwd["sol"]["data"] = data.astype(np.float32)
    fwd["sol"]["ncol"] = data.shape[1]
    fwd["_orig_sol"] = data
    fwd["src"] = src
    fwd["source_rr"] = np.row_stack([s["rr"] for s in fwd["src"]])
    fwd["source_nn"] = np.ascontiguousarray(np.tile(np.eye(p), fwd["nsource"]).T)

    # NOT USED!!!!
    # elif model.endswith("affine_fids_template"):
    #     """Prepare a template forward model which uses fsaverage as source
    #     space and standard layout of the relevant cap as electrode
    #     positions. Electrodes are coregistered to MNI using fiducials.

    #     This is the same across all subjects.

    #     head space : manufacturer layout space
    #     mri space  : MNI space
    #     trans(fro=mri, to=head) = ~I (used)
    #     """
    #     # Overwrite subject info
    #     info = mne.create_info(montage.ch_names, sfreq=100, ch_types="eeg")

    #     # Replace CAT surfaces with fsaverage (this way we can just use the
    #     # subsampling function in simnibs as is) and transform to MNI152
    #     # -----------------------------------------------------------------
    #     # Transformation from MNI152 to MNI305
    #     # (fsaverage is in MNI305 but SimNIBS uses the MNI152 as its MNI space)
    #     # (Everything is mm so this is fine.)
    #     # mni305_to_mni152 = np.array(
    #     #     [
    #     #         [0.9975, -0.0073, 0.0176, -0.0429],
    #     #         [0.0146, 1.0009, -0.0024, 1.5496],
    #     #         [-0.0130, -0.0093, 0.9971, 1.1840],
    #     #         [0, 0, 0, 1],
    #     #     ]
    #     # )
    #     # trans = mne.transforms.Transform('unknown', 'unknown', mni305_to_mni152)
    #     # fname = r'C:\Users\jdue\googledrive\mni152_fiducials\mni305_to_mni152-trans.fif'
    #     # mne.write_trans(fname, trans)
    #     # Fiducials in MNI152 space
    #     # mni152_fids = (
    #     #     (FIFF.FIFFV_POINT_LPA, [-80, -12, -52]),
    #     #     (FIFF.FIFFV_POINT_NASION, [-1, 81, -45]),
    #     #     (FIFF.FIFFV_POINT_RPA, [80, -12, -52]),
    #     # )

    #     # mni_landmarks = [
    #     #         dict(
    #     #             kind=FIFF.FIFFV_POINT_CARDINAL,
    #     #             r=np.array(r) * 1e-3,  # mm to m
    #     #             ident=i,

    #     #         )
    #     #     for i, r in mni152_fids
    #     # ]
    #     # fname = r'C:\Users\jdue\googledrive\mni152_fiducials\mni152-fiducials.fif'
    #     # mne.io.write_fiducials(fname, mni_landmarks, FIFF.FIFFV_COORD_MRI)

    #     file_path = Path(__file__)
    #     mni305_to_mni152 = mne.read_trans(file_path / "mni305_to_mni152-trans.fif")
    #     mni152_landmarks, _ = mne.io.read_fiducials(
    #         file_path / "mni152-fiducials.fif"
    #     )

    #     m2m = io.simnibs.get_path("m2m")
    #     surf = m2m / "surfaces"

    #     # Remove all the CAT stuff
    #     for f in surf.glob("*"):
    #         os.remove(f)

    #     # Transform fsaverage to MNI152 and write
    #     fsavg = eeg.FsAverage(5)
    #     # pial = fsavg.get_surface('pial')
    #     central = fsavg.get_central_surface()
    #     reg = fsavg.get_surface("sphere")

    #     # fsavg = eeg.load_fsaverage("central")
    #     # fsavg_reg = eeg.load_fsaverage("sphere_reg")
    #     for hemi in fsavg:
    #         central[hemi]["points"] = mne.transforms.apply_trans(
    #             mni305_to_mni152, central[hemi]["points"]
    #         )
    #         nib.GiftiImage(
    #             darrays=(
    #                 nib.gifti.gifti.GiftiDataArray(
    #                     central[hemi]["points"], "pointset"
    #                 ),
    #                 nib.gifti.gifti.GiftiDataArray(
    #                     central[hemi]["tris"], "triangle"
    #                 ),
    #             )
    #         ).to_filename(surf / f"{hemi}.central.gii")

    #         nib.GiftiImage(
    #             darrays=(
    #                 nib.gifti.gifti.GiftiDataArray(reg[hemi]["points"], "pointset"),
    #                 nib.gifti.gifti.GiftiDataArray(reg[hemi]["tris"], "triangle"),
    #             )
    #         ).to_filename(surf / f"{hemi}.sphere.reg.gii")

    #     # Make transformation from head (standard layout) to mri (MNI152)
    #     # -----------------------------------------------------------------
    #     # Fit the transform (allow scaling as this is a template match)
    #     # (_fiducial_coords takes care of the order)
    #     trans = mne.coreg.fit_matched_points(
    #         src_pts=mne.viz._3d._fiducial_coords(montage.dig),
    #         tgt_pts=mne.viz._3d._fiducial_coords(mni152_landmarks),
    #         scale=True,
    #     )
    #     trans = mne.Transform("head", "mri", trans)

