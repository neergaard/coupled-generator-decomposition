import shutil
import warnings

import mne
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from simnibs.utils.file_finder import SubjectFiles
from simnibs.mesh_tools import mesh_io
from simnibs.segmentation import brain_surface
from simnibs.simulation import eeg, eeg_mne_tools

from projects.facerecognition import utils
from projects.facerecognition.config import Config

from projects.mnieeg.forward import (
    get_outer_surface_points,
    mesh_to_volume,
    morph_forward,
    tps_deform_estimate,
    tps_deform_apply,
)


def make_forward_solution(subject_id):
    """Run a forward simulation."""

    out_format = "mne"
    contains_ref = False

    io = utils.SubjectIO(subject_id)
    # montage_proj...
    io.data.update(stage="forward")
    info = io.data.get_filename(suffix="info")
    mri_head_t = io.data.get_filename(suffix="trans")

    # We pretend that fsaverage is in head coords
    # (on reading or writing the forward solution, MNE will apply the
    # mri_head_t to the source space to bring it to head coords)
    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_central = fsavg.get_central_surface()
    fsavg_central = eeg_mne_tools.make_source_spaces(fsavg_central, fsavg.name, "head")

    models = [
        m
        for m, t in zip(Config.forward.MODELS, Config.forward.MODEL_TYPE)
        if t == "simnibs"
    ]
    for fm in models:
        print(f"Running simulation for model {fm.upper()}")
        subject_dir = io.simnibs[fm].get_path("subject")
        m2m_dir = io.simnibs[fm].get_path("m2m")
        io.data.update(forward=fm)
        montage = io.data.get_filename(suffix="montage", extension="csv")

        print("Subsampling each hemisphere to", Config.forward.SUBSAMPLING, "nodes")
        brain_surface.subsample_surfaces(m2m_dir, Config.forward.SUBSAMPLING)

        fem_dir = subject_dir / "fem"

        leadfield = eeg.compute_leadfield(
            m2m_dir, fem_dir, montage, subsampling=Config.forward.SUBSAMPLING
        )

        src, forward, morph = eeg.prepare_for_inverse(
            m2m_dir,
            leadfield,
            out_format,
            info,
            mri_head_t,
            Config.inverse.FSAVERAGE,
            contains_ref,
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


def deform_template_to_subject(subject_id):
    """Deform the MNI template head model (and central gray matter surfaces) to
    subject space using head points defined around each electrode.
    """

    simnibs_dir = "template"

    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward", suffix="montage", extension="csv")

    m2m_ref = io.simnibs[Config.forward.REFERENCE].get_path("m2m")
    subfiles_ref = SubjectFiles(subpath=str(m2m_ref))

    # The new subject
    m2m = io.simnibs[simnibs_dir].get_path("m2m")
    subfiles = SubjectFiles(subpath=str(m2m))
    subject_dir = io.simnibs[simnibs_dir].get_path("subject")
    surface_dir = m2m / "surfaces"
    if not surface_dir.exists():
        surface_dir.mkdir(parents=True)

    # We get the surfaces and mesh the MNI152 run
    iomni = utils.SubjectIO("mni152")
    m2m_mni = iomni.simnibs[simnibs_dir].get_path("m2m")
    subfiles_mni = SubjectFiles(subpath=str(m2m_mni))

    # prepare head points and affine transforms between mri and mni spaces
    headpoints_per_ch = 4
    fname_montage = io.data.get_filename(forward=Config.forward.REFERENCE)
    shutil.copy(fname_montage, io.data.get_filename(forward="template"))
    montage = eeg.make_montage(fname_montage)
    skin = pv.read(
        io.simnibs[Config.forward.REFERENCE].get_path("subject")
        / "skin_outer_annot.vtk"
    )
    # query nearest neighbors of each channel
    tree = cKDTree(skin.points[skin["outer_points"].astype(bool)])
    _, idx = tree.query(montage.ch_pos, headpoints_per_ch)
    head_pts = tree.data[idx.ravel()]
    np.savetxt(subject_dir / "headpoints.csv", head_pts)

    # Make MNI to MRI affine transformation
    lm_mni = eeg.Montage()
    lm_mni.add_landmarks()
    mni2mri = lm_mni.fit_to(montage, scale=True)
    np.savetxt(subject_dir / "mni_mri-trans.txt", mni2mri)
    # mni2mri = np.loadtxt(
    #     io.simnibs[Config.forward.REFERENCE].get_path("subject") / "mni_mri-trans.txt"
    # )

    # Get template head model and transform to subject space
    m = mesh_io.read_msh(m2m_mni / f"sub-{iomni.subject}.msh")
    m.nodes.node_coord = eeg.apply_trans(mni2mri, m.nodes.node_coord)
    skin = m.crop_mesh(1005)  # crop makes a copy
    subset = get_outer_surface_points(skin)
    surf = {
        "points": skin.nodes.node_coord,
        "tris": skin.elm.node_number_list[:, :3] - 1,
    }

    # Project head point on template skin surface
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
    m.write(str(m2m / f"sub-{io.subject}.msh"))
    mesh_to_volume(m, subfiles_ref, subfiles)

    # Also transform the (central) cortical surfaces (the spherical
    # registrations are unaffected so just copy them)
    surfs = eeg.load_surface(subfiles_mni, "central")
    for hemi, surf in surfs.items():
        pts = eeg.apply_trans(mni2mri, surf["points"])
        pts = tps_deform_apply(warp, head_pts_proj, pts)
        darrays = (
            nib.gifti.gifti.GiftiDataArray(pts, "pointset"),
            nib.gifti.gifti.GiftiDataArray(surf["tris"], "triangle"),
        )
        subsamp = nib.GiftiImage(darrays=darrays)
        subsamp.to_filename(surface_dir / f"{hemi}.central.gii")

        shutil.copy(
            subfiles_mni.get_surface(hemi, "sphere_reg"),
            surface_dir / f"{hemi}.sphere.reg.gii",
        )
