import subprocess

import mne
import meshio
import nibabel as nib

from simnibs.utils.file_finder import path2bin

from projects.mnieeg.forward import morph_forward
from projects.anateeg.mne_tools import update_source_morph
from projects.anateeg.utils import load_module

from projects.facerecognition import utils
from projects.facerecognition.config import Config


def make_bem_surfaces(subject_id):
    io = utils.SubjectIO(subject_id)

    params = dict(subject=f"sub-{io.subject}", freesurfer_dir=Config.path.FREESURFER)
    call = "mne watershed_bem -s {subject} -d {freesurfer_dir} --overwrite"
    call = " && ".join([load_module("freesurfer"), call.format(**params),])
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

    update_source_morph(morph, fwd["src"])

    # Rereference to average reference to be consistent with SimNIBS and
    # FieldTrip
    fwd["sol"]["data"] = fwd["sol"]["data"] - fwd["sol"]["data"].mean(0)

    io.data.update(forward="mne")

    morph.save(io.data.get_filename(suffix="morph", extension="h5"), overwrite=True)
    src.save(io.data.get_filename(suffix="src"), overwrite=True)
    mne.write_forward_solution(io.data.get_filename(suffix="fwd"), fwd, overwrite=True)
    morph_forward(fwd, morph, fsavg)
    mne.write_forward_solution(
        io.data.get_filename(space="fsaverage", suffix="fwd"), fwd, overwrite=True,
    )
