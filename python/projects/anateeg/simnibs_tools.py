import copy
import json
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
from simnibs.simulation import cond, eeg, eeg_mne_tools

from projects.anateeg import utils
from projects.anateeg.config import Config
from projects.anateeg.skull_reco import cv_atlases
from projects.anateeg.skull_reco.landmarks import MRILandmarks

from projects.mnieeg.forward import (
    get_outer_surface_points,
    mesh_to_volume,
    montage_to_polydata,
    morph_forward,
    sample_conductivities,
    tps_deform_estimate,
    tps_deform_apply,
)


def make_skin_outer_annot(io_simnibs):

    # Find the vertices on the scalp to use for projection
    m = mesh_io.read_msh(io_simnibs.get_path("m2m") / f"sub-{io_simnibs.subject}.msh")
    m = m.crop_mesh(1005)
    surf = {"points": m.nodes.node_coord, "tris": m.elm.node_number_list[:, :3] - 1}
    subset = get_outer_surface_points(m)

    skin = pv.make_tri_mesh(surf["points"], surf["tris"])
    skin["outer_points"] = np.zeros(skin.n_points, dtype=int)
    skin["outer_points"][subset] = 1
    skin.save(io_simnibs.get_path("subject") / "skin_outer_annot.vtk")
    return skin


def make_montage(subject_id):
    """
    The montage is created by registering the template layout to subject mri
    space using landmarks (one [uniform] scaling parameter) and then projecting
    it onto the skin surface of the reference model.

    For the other models, the electrodes are then projected onto the skin of
    that model (here it is done for SimNIBS; should be taken care of but MNE
    and FieldTrip for their models).
    """
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward", suffix="montage", extension="csv")
    io.data.path.ensure_exists()
    ref = Config.forward.REFERENCE

    skin = make_skin_outer_annot(io.simnibs[ref])
    subset = np.flatnonzero(skin["outer_points"])
    skin = dict(points=skin.points, tris=skin.faces.reshape(-1, 4)[:, 1:])

    # MNI to MRI transformation (with scaling)
    subject_dir = io.simnibs[ref].get_path("subject")
    with open(Config.path.DATA / "subjects.json", "r") as f:
        original_subject = json.load(f)[io.subject]
    lm_mni = eeg.Montage()
    lm_mni.add_landmarks()
    lm_sub = eeg.Montage(landmarks=MRILandmarks()[original_subject])
    mni2mri = lm_mni.fit_to(lm_sub, scale=True)
    np.savetxt(subject_dir / "mni_mri-trans.txt", mni2mri)

    montage = eeg.make_montage(Config.forward.MONTAGE)
    montage.add_landmarks()

    # Remove the reference electrode to match the covariance from mnieeg. We
    # loose one DOF which is not really necessary but do this for convenience
    not_ref = montage.ch_types != "ReferenceElectrode"
    montage.ch_names = montage.ch_names[not_ref]
    montage.ch_pos = montage.ch_pos[not_ref]
    montage.ch_types = montage.ch_types[not_ref]
    montage.n_channels -= 1

    montage.apply_trans(mni2mri)
    montage_to_polydata(montage).save(subject_dir / "montage.vtk")
    montage.project_to_surface(skin, subset)
    montage.write(io.data.get_filename(forward=ref))
    montage_to_polydata(montage).save(subject_dir / "montage_proj.vtk")

    for model, model_type in zip(Config.forward.MODELS, Config.forward.MODEL_TYPE):
        if model_type == "simnibs" and model != ref and model != "template":
            skin = make_skin_outer_annot(io.simnibs[model])
            subset = np.flatnonzero(skin["outer_points"])
            skin = dict(points=skin.points, tris=skin.faces.reshape(-1, 4)[:, 1:])
            this_montage = copy.copy(montage)
            this_montage.project_to_surface(skin, subset)
            this_montage.write(io.data.get_filename(forward=model))
            montage_to_polydata(montage).save(
                io.simnibs[model].get_path("subject") / "montage_proj.vtk"
            )


def make_forward_solution(subject_id):
    """Run a forward simulation."""

    out_format = "mne"
    apply_average_ref = True

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
    # models = ["charm"] # when using T1_only
    for fm in models:
        print(f"Running simulation for model {fm.upper()}")
        subject_dir = io.simnibs[fm].get_path("subject")
        m2m_dir = io.simnibs[fm].get_path("m2m")

        # m2m_dir = m2m_dir.with_name(m2m_dir.name + "_T1_only")

        io.data.update(forward=fm)
        montage = io.data.get_filename(suffix="montage", extension="csv")

        # io.data.update(forward=fm+"_T1_only")

        print("Subsampling each hemisphere to", Config.forward.SUBSAMPLING, "nodes")
        brain_surface.subsample_surfaces(m2m_dir, Config.forward.SUBSAMPLING)

        fem_dir = subject_dir / "fem"
        # fem_dir = subject_dir / "fem_T1_only"

        leadfield = eeg.compute_tdcs_leadfield(
            m2m_dir, fem_dir, montage, subsampling=Config.forward.SUBSAMPLING
        )

        src, forward, morph = eeg.prepare_for_inverse(
            m2m_dir,
            leadfield,
            out_format,
            info,
            mri_head_t,
            Config.inverse.FSAVERAGE,
            apply_average_ref,
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


def get_unbiased_template(original_subject_id):
    atlas_id = [
        atlas
        for atlas, subjects in cv_atlases.included_subjects.items()
        if original_subject_id not in subjects
    ]
    assert len(atlas_id) == 1
    return atlas_id[0]


def deform_template_to_subject(subject_id):
    """Deform the MNI template head model (and central gray matter surfaces) to
    subject space using head points defined around each electrode.
    """

    simnibs_dir = "template"

    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward", suffix="montage", extension="csv")

    with open(Config.path.DATA / "subjects.json", "r") as f:
        original_subject = json.load(f)[io.subject]
    atlas_id = get_unbiased_template(original_subject)

    m2m_ref = io.simnibs[Config.forward.REFERENCE].get_path("m2m")
    subfiles_ref = SubjectFiles(subpath=str(m2m_ref))

    # The new subject
    m2m = io.simnibs[simnibs_dir].get_path("m2m")
    subfiles = SubjectFiles(subpath=str(m2m))
    subject_dir = io.simnibs[simnibs_dir].get_path("subject")
    surface_dir = m2m / "surfaces"
    if not surface_dir.exists():
        surface_dir.mkdir(parents=True)

    # Run using the default charm atlas. We get the surfaces from here
    iomni = utils.SubjectIO("mni152", raise_on_exclude=False)
    m2m_mni = iomni.simnibs[simnibs_dir].get_path("m2m")
    subfiles_mni = SubjectFiles(subpath=str(m2m_mni))

    # Run using the clean atlas. We get the head mesh from here
    iomni_x = utils.SubjectIO(f"mni152_{atlas_id}", raise_on_exclude=False)
    m2m_mni_x = iomni_x.simnibs[simnibs_dir].get_path("m2m")
    subfiles_mni_x = SubjectFiles(subpath=str(m2m_mni_x))

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

    mni2mri = np.loadtxt(
        io.simnibs[Config.forward.REFERENCE].get_path("subject") / "mni_mri-trans.txt"
    )

    # Get template head model and transform to subject space
    m = mesh_io.read_msh(m2m_mni_x / f"sub-{iomni_x.subject}.msh")
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


def compute_forward_sample_cond(subject_id):

    io = utils.SubjectIO(subject_id)
    subject_dir = io.simnibs["reference"].get_path("subject")
    m2m_dir = io.simnibs["reference"].get_path("m2m")

    fem_dir = subject_dir / "fem_sample_cond"
    # montage = subject_dir / f"montage_proj.csv"  # f"montage_{fm}_proj.csv"

    fname_montage = io.data.get_filename(
        stage="forward",
        forward=Config.forward.REFERENCE,
        suffix="montage",
        extension="csv",
    )

    seed = int(subject_id)
    conds, samples = sample_conductivities(seed)
    with open(subject_dir / "sample_cond.json", "w") as f:
        json.dump(samples, f, indent=4)

    _ = eeg.compute_leadfield(
        m2m_dir,
        fem_dir,
        fname_montage,
        subsampling=Config.forward.SUBSAMPLING,
        init_kwargs=dict(cond=conds),
    )


def prepare_for_inverse_sampled_cond(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    mri_head_t = io.data.get_filename(suffix="trans")

    # We pretend that fsaverage is in head coords
    # (on reading or writing the forward solution, MNE will apply the
    # mri_head_t to the source space to bring it to head coords)
    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_central = fsavg.get_central_surface()
    fsavg_central = eeg_mne_tools.make_source_spaces(fsavg_central, fsavg.name, "head")

    out_format = "mne"

    subject_dir = io.simnibs["reference"].get_path("subject")
    m2m_dir = io.simnibs["reference"].get_path("m2m")

    fname_montage = io.data.get_filename(
        forward=Config.forward.REFERENCE, suffix="montage", extension="csv"
    )
    fm = Config.forward.REFERENCE + "_sample_cond"
    io.data.update(forward=fm)

    # _proj
    leadfield = (
        subject_dir
        / "fem_sample_cond"
        / f"sub-{io.subject}_leadfield_{fname_montage.stem}.hdf5"
    )

    info = io.data.get_filename(forward=None, suffix="info")

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


def compute_forward_ref_matched_fieldtrip(subject_id):

    io = utils.SubjectIO(subject_id)
    subject_dir = io.simnibs["reference"].get_path("subject")
    m2m_dir = io.simnibs["reference"].get_path("m2m")

    fem_dir = subject_dir / "fem_match_fieldtrip"
    # montage = subject_dir / f"montage_proj.csv"  # f"montage_{fm}_proj.csv"

    fname_montage = io.data.get_filename(
        stage="forward",
        forward=Config.forward.REFERENCE,
        suffix="montage",
        extension="csv",
    )

    # FieldTrip conductivities
    # conductivities = [0.33 0.14 1.79 0.01 0.43]
    # tissues = {'gray','white','csf','skull','scalp'}

    conductivities = dict(
        white_matter=0.14,
        gray_matter=0.33,
        csf=1.79,
        spongy_bone=0.01,
        compact_bone=0.01,
        skin=0.43,
        blood=1.79,
        eyes=0.43,
        muscle=0.43,
    )
    tissue_to_index = dict(
        white_matter=0,
        gray_matter=1,
        csf=2,
        skin=4,
        eyes=5,
        compact_bone=6,
        spongy_bone=7,
        blood=8,
        muscle=9,
    )
    s = cond.standard_cond()
    for t, c in conductivities.items():
        s[tissue_to_index[t]].value = c

    _ = eeg.compute_leadfield(
        m2m_dir,
        fem_dir,
        fname_montage,
        subsampling=Config.forward.SUBSAMPLING,
        init_kwargs=dict(cond=s),
    )


def prepare_for_inverse_ref_matched_fieldtrip(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    mri_head_t = io.data.get_filename(suffix="trans")

    # We pretend that fsaverage is in head coords
    # (on reading or writing the forward solution, MNE will apply the
    # mri_head_t to the source space to bring it to head coords)
    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_central = fsavg.get_central_surface()
    fsavg_central = eeg_mne_tools.make_source_spaces(fsavg_central, fsavg.name, "head")

    out_format = "mne"

    subject_dir = io.simnibs["reference"].get_path("subject")
    m2m_dir = io.simnibs["reference"].get_path("m2m")

    fname_montage = io.data.get_filename(
        forward=Config.forward.REFERENCE, suffix="montage", extension="csv"
    )
    fm = Config.forward.REFERENCE + "_match_fieldtrip"
    io.data.update(forward=fm)

    # _proj
    leadfield = (
        subject_dir
        / "fem_match_fieldtrip"
        / f"sub-{io.subject}_leadfield_{fname_montage.stem}.hdf5"
    )

    info = io.data.get_filename(forward=None, suffix="info")

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

