import sys

import mne
import nibabel as nib
import numpy as np

from simnibs.segmentation import charm_main

from projects.mnieeg import utils


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    io = utils.SubjectIO(subject_id, raise_on_exclude=True)

    m2m_dir = io.simnibs_template.get_path("m2m")
    if not m2m_dir.exists():
        m2m_dir.mkdir()

    # Since the head points were transformed from subject space to mni space,
    # the warped template is still in MNI coordinates. Thus, we need to
    # transform it back
    print("Transforming warped T1 to subject MRI space")
    sub_dir = io.simnibs_template.get_path("subject")
    mni2mri = mne.read_trans(sub_dir / "mni2mri-trans.fif")
    mni2mri = mni2mri["trans"]
    mni2mri[:3, 3] *= 1e3

    # Pad such that the neck is also reconstructed
    pad_width = 100
    t1w = nib.load(sub_dir / "t1_warped_mni.nii")
    data = np.concatenate(
        (
            np.zeros((*t1w.shape[:2], pad_width), dtype=t1w.get_data_dtype()),
            t1w.get_fdata(),
        ),
        axis=-1,
    )
    affine = t1w.affine
    affine[2, 3] -= pad_width  # this is OK because the resolution is 1 mm
    img = nib.Nifti1Image(data, mni2mri @ t1w.affine)
    img.to_filename(sub_dir / "t1_warped.nii.gz")

    # Do everything
    charm_main.run(
        str(m2m_dir),
        img.get_filename(),
        registerT2=True,
        initatlas=True,
        segment=True,
        create_surfaces=True,
        mesh_image=True,
        usesettings=None,
        noneck=False,
        options_str=None,
    )

