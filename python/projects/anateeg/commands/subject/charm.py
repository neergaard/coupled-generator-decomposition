import json
from pathlib import Path
import shutil
import sys

import nibabel as nib
import nibabel.processing
import numpy as np

from simnibs import SIMNIBSDIR
from simnibs.segmentation import charm_main, charm_utils
from simnibs.utils.file_finder import SubjectFiles, templates
from simnibs.utils.settings_reader import read_ini

from projects.anateeg.config import Config
from projects.anateeg import utils


def charm_segment(subject_id):
    """Run initial segmentation steps."""

    io = utils.SubjectIO(subject_id)
    io.data.update(task=None, session="mri", extension="nii.gz")

    m2m = io.simnibs["charm"].get_path("m2m")
    m2m = m2m.with_name(m2m.name + "_T1_only")
    if not m2m.exists():
        m2m.mkdir(parents=True)

    # subject_ini = Config.path.RESOURCES / "charm" / f"sub-{io.subject}.ini"
    # settings_file = str(subject_ini) if subject_ini.exists() else None

    charm_main.run(
        str(m2m),
        str(io.data.get_filename(suffix="t1w")),
        # str(io.data.get_filename(suffix="t2w")),
        # registerT2=True,
        initatlas=True,
        segment=True,
        create_surfaces=False,
        mesh_image=False,
        usesettings=None,
        noneck=False,
        force_forms=True,
        # init_transform=fname_trans,
        options_str=None,
    )

    # fname_trans.unlink()


def charm_mesh(subject_id):
    """Run surface creation and meshing."""
    io = utils.SubjectIO(subject_id)

    charm_main.run(
        str(io.simnibs["charm"].get_path("m2m"))  + "_T1_only",
        registerT2=False,
        initatlas=False,
        segment=False,
        create_surfaces=True,
        mesh_image=True,
        usesettings=None,
        noneck=False,
        options_str=None,
    )


def construct_clean_labeling(subject_id):
    io = utils.SubjectIO(subject_id)
    sf = SubjectFiles(subpath=str(io.simnibs["charm"].get_path("m2m")) + "_T1_only")

    with open(Config.path.DATA / "subjects.json", "r") as f:
        original_subject = json.load(f)[io.subject]

    # Get the simnibs tissue mapping
    atlas_name = read_ini(str(Path(SIMNIBSDIR) / "charm.ini"))["samseg"]["atlas_name"]
    atlas_path = Path(templates.charm_atlas_path) / atlas_name
    atlas_settings = read_ini(str(atlas_path / (atlas_name + ".ini")))
    simnibs_tissues = atlas_settings["conductivity_mapping"]["simnibs_tissues"]
    simnibs_tissues["Background"] = 0

    posterior_dir = Config.path.CLEAN_SEGS / original_subject / "T1T2"

    # Map posteriors to SimNIBS tissues
    posteriors = {
        "Air_internal_posterior": "Air_pockets",
        "Artery_posterior": "Blood",
        "Background_posterior": "Background",  # goes to zero
        "Bone_cancellous_posterior": "Spongy_bone",
        "Bone_cortical_posterior": "Compact_bone",
        "Cerebrospinal_fluid_posterior": "CSF",
        "Cerebrum_grey_matter_posterior": "GM",
        "Cerebrum_white_matter_posterior": "WM",
        "Eyes_posterior": "Eyes",
        "Mucosa_posterior": "Scalp",
        "Other_tissues_posterior": "Scalp",
        "Rectus_muscles_posterior": "Muscle",
        "Skin_posterior": "Scalp",
        "Spinal_cord_posterior": "WM",
        "Vein_posterior": "Blood",
        "Visual_nerve_posterior": "WM",
    }

    # Running argmax to avoid ~18 Gb memory usage
    print("Resampling posteriors and applying running argmax")
    tlu = nib.load(sf.tissue_labeling_upsampled)
    labeling = np.zeros(tlu.shape, dtype=tlu.get_data_dtype())
    posterior_val = np.zeros(labeling.shape)
    posterior2simnibs = np.array(
        [simnibs_tissues[v] for v in posteriors.values()], dtype=tlu.get_data_dtype()
    )
    for i, p in enumerate(posteriors):
        img = nib.load(posterior_dir / (p + ".nii.gz"))
        rimg = nibabel.processing.resample_from_to(img, tlu, order=1, mode="nearest")
        mask = rimg.get_fdata() > posterior_val
        posterior_val[mask] = rimg.get_fdata()[mask]
        labeling[mask] = i
    labeling = posterior2simnibs[labeling]

    print("Applying morphological operations")
    upper_part = nib.load(sf.upper_mask).get_fdata().astype(bool)
    labeling = charm_utils._morphological_operations(
        labeling, upper_part, simnibs_tissues
    )

    fname = tlu.get_filename()
    fname_cp = fname.rstrip(".nii.gz") + "_orig.nii.gz"
    shutil.copy(fname, fname_cp)
    labeling = nib.Nifti1Image(labeling, tlu.affine, tlu.header)
    labeling.to_filename(fname)


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")

    print("Creating segmentation")
    charm_segment(subject_id)
    print("Replacing labels with clean run")
    construct_clean_labeling(subject_id)
    print("Creating surfaces and mesh")
    charm_mesh(subject_id)

