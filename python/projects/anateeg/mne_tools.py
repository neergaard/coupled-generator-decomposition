import subprocess

import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
import scipy.sparse

from simnibs.simulation import eeg, eeg_mne_tools
from simnibs.utils.file_finder import path2bin

import meshio

from projects.anateeg import utils
from projects.anateeg.config import Config

from projects.mnieeg.forward import morph_forward


def make_data(subject_id):
    """Create the following data structures

    - mri-head transformation
    - info object
    - covariance object

    """
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")

    fname = io.data.get_filename(
        forward=Config.forward.REFERENCE, suffix="montage", extension="csv"
    )

    montage = eeg.make_montage(fname)
    montage = eeg_mne_tools.simnibs_montage_to_mne_montage(montage, coord_frame="mri")

    mri_head_t = mne.channels.compute_native_head_t(montage)

    info = mne.create_info(montage.ch_names, 100, ch_types="eeg")
    info.set_montage(montage)

    cov = mne.read_cov(Config.path.RESOURCES / "mnieeg_group-cov.fif")

    mne.write_trans(io.data.get_filename(suffix="trans"), mri_head_t)
    mne.io.write_info(io.data.get_filename(suffix="info"), info)
    mne.write_cov(io.data.get_filename(suffix="cov"), cov)


def make_bem_surfaces(subject_id):
    io = utils.SubjectIO(subject_id)

    params = dict(subject=f"sub-{io.subject}", freesurfer_dir=Config.path.FREESURFER)
    call = "mne watershed_bem -s {subject} -d {freesurfer_dir} --overwrite"
    call = " && ".join([utils.load_module("freesurfer"), call.format(**params),])
    subprocess.run(["bash", "-c", call])

    # mne.bem.make_watershed_bem(io.subject, subjects_dir=Config.path.FREESURFER)


def decouple_inner_outer_skull(subject_id):
    io = utils.SubjectIO(subject_id)

    bem_dir = Config.path.FREESURFER / f"sub-{io.subject}" / "bem"

    min_dist = 2
    n_points = 5120  # downsample to ico=4 number of points
    meshfix = path2bin("meshfix")
    which_decouple = "inin"

    print(f"Downsampling surfaces to {n_points} points")
    surfs = {}
    surfs_off = {}
    for surf in ("inner_skull", "outer_skull", "outer_skin"):
        this_surf = bem_dir / f"{surf}.surf"
        this_surf_off = this_surf.with_suffix(".off")
        surfs[surf] = this_surf
        surfs_off[surf] = this_surf_off

        verts, tris = nib.freesurfer.read_geometry(this_surf)
        mesh = meshio.Mesh(verts, [("triangle", tris)])
        mesh.write(this_surf_off)

        downsample = f"{meshfix} {this_surf_off} -u 5 --vertices {n_points} -o {this_surf_off}".split()
        subprocess.run(downsample)

    check_intersect = f"{meshfix} {surfs_off['inner_skull']} {surfs_off['outer_skull']} --shells 2 --no-clean --intersect".split()
    decouple = f"{meshfix} {surfs_off['inner_skull']} {surfs_off['outer_skull']} --shells 2 --no-clean --decouple-{which_decouple} {min_dist} -o {surfs_off['inner_skull']}".split()

    print(f"Ensuring {min_dist} mm between inner skull and outer skull")
    subprocess.run(decouple)
    # if subprocess.run(check_intersect).returncode == 0:
    #     for _ in range(3):
    #         if subprocess.run(check_intersect).returncode == 1:
    assert subprocess.run(check_intersect).returncode == 1, "Error: decoupling failed"
    print("Surfaces successfully decoupled")

    print("Writing the downsampled and decoupled surfaces")
    for k, v in surfs.items():
        mesh = meshio.read(surfs_off[k])
        nib.freesurfer.write_geometry(v, mesh.points, mesh.cells[0][1])

    # Clean up
    for surf in surfs_off.values():
        surf.unlink()


def make_forward_solution(subject_id):
    """

    Create the BEM surfaces. These are in surface RAS (FS coordinates) so convert to scanner RAS (world coordinates)

    """
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    ref = Config.forward.REFERENCE

    fsavg = mne.read_source_spaces(Config.path.RESOURCES / "fsaverage_central-src.fif")

    info = mne.io.read_info(io.data.get_filename(suffix="info"))
    trans = mne.read_trans(io.data.get_filename(suffix="trans"))
    src = mne.read_source_spaces(io.data.get_filename(forward=ref, suffix="src"))
    morph = mne.read_source_morph(
        io.data.get_filename(forward=ref, suffix="morph", extension="h5")
    )

    surfs = mne.make_bem_model(
        f"sub-{io.subject}", ico=None, subjects_dir=Config.path.FREESURFER
    )

    # vox : MRI (voxel) voxel indices
    # mri : MRI (surface RAS) freesurfer coordinates
    # ras : RAS (non-zero origin) real world coordinates (scanner coordinates)
    # (mri_ras_t is tkr-RAS to scanner-RAS)
    # outputs: vox_ras_t, vox_mri_t, mri_ras_t, dims, zooms
    _, _, mri_ras_t, _, _ = mne._freesurfer._read_mri_info(
        Config.path.FREESURFER / f"sub-{io.subject}" / "mri" / "orig.mgz"
    )
    surfs = [
        mne.transform_surface_to(surf, mri_ras_t["to"], mri_ras_t) for surf in surfs
    ]

    # Hack otherwise `make_forward_solution`` will complain that the BEM model is
    # not in MRI (surface RAS) space, however, we work in scanner space and not
    # FreeSurfer space so just pretend like we are in MRI (surface RAS)
    for surf in surfs:
        surf["coord_frame"] = mri_ras_t["from"]

    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(info, trans, src, bem)

    # Rereference to average reference to be consistent with SimNIBS and
    # FieldTrip
    fwd["sol"]["data"] -= fwd["sol"]["data"].mean(0)
    fwd["_orig_sol"] -= fwd["_orig_sol"].mean(0)

    update_source_morph(morph, fwd["src"])

    io.data.update(forward="mne")

    morph.save(io.data.get_filename(suffix="morph", extension="h5"), overwrite=True)
    src.save(io.data.get_filename(suffix="src"), overwrite=True)
    mne.write_forward_solution(io.data.get_filename(suffix="fwd"), fwd, overwrite=True)
    morph_forward(fwd, morph, fsavg)
    mne.write_forward_solution(
        io.data.get_filename(space="fsaverage", suffix="fwd"), fwd, overwrite=True,
    )


def update_source_morph(morph, src, cutoff=0.2):
    """
    Update a source morph if needed.

    Update a source morph by smoothing such that all sources with a coverage
    less than or equal to `cutoff` is replaced by a smoothed version.

    All rows are normalized to one.
    """
    if all(len(s["vertno"]) == s["np"] for s in src):
        print("No smoothing needed")
        return

    assert cutoff >= 0 and cutoff <= 1

    vertno_cols = np.concatenate(
        [s["vertno"] + n for s, n in zip(src, (0, src[0]["np"]))]
    )
    need_smooth = np.ravel(morph.morph_mat[:, vertno_cols].sum(1) <= cutoff)
    print(f"{need_smooth.sum()} sources requires smoothing")

    es = []
    for s in src:
        e = mne.surface.mesh_edges(s["tris"])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e += scipy.sparse.eye(n_vertices, format="csr")
        es.append(e)
    es = scipy.sparse.block_diag(es).tocsr()
    smoother = mne.morph._surf_upsampling_mat(vertno_cols, es, None)[0]

    # Only smooth where needed
    repl = morph.morph_mat[need_smooth] @ smoother
    morph.morph_mat = morph.morph_mat[:, vertno_cols]
    morph.morph_mat = morph.morph_mat.tolil()
    morph.morph_mat[need_smooth] = repl
    morph.morph_mat = morph.morph_mat.tocsr()
    # normalize all rows to one
    morph.morph_mat.data /= np.repeat(
        np.ravel(morph.morph_mat.sum(1)), morph.morph_mat.getnnz(1)
    )

    morph.src_data["vertices_from"] = [s["vertno"] for s in src]


def match_fwd_to_src(fwd, src):
    """
    where `src` is a subset of `fwd['src']`
    """

    assert fwd["src"][0]["subject_his_id"] == src[0]["subject_his_id"]
    # assert all(np.allclose(s0['rr'], s1['rr']) for s0,s1 in zip(src, fwd['src']))

    if all(s0["nuse"] == s1["nuse"] for s0, s1 in zip(src, fwd["src"])):
        return slice(None)  # selects everything

    vertno_cols = np.concatenate(
        [s["vertno"] + n for s, n in zip(src, (0, src[0]["np"]))]
    )

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
    data = data[:, vertno_cols].reshape(n, -1)

    fwd["nsource"] = len(vertno_cols)
    fwd["sol"]["data"] = data.astype(np.float32)
    fwd["sol"]["ncol"] = data.shape[1]
    fwd["_orig_sol"] = data
    fwd["src"] = src
    fwd["source_rr"] = fwd["source_rr"][vertno_cols]
    fwd["source_nn"] = fwd["source_nn"][vertno_cols]

    return vertno_cols
