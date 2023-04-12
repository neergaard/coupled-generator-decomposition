import sys

import mne
import nibabel as nib
import numpy as np

from projects.facerecognition_dtu import utils
from projects.facerecognition_dtu.config import Config

from projects.anateeg.utils import parse_args

import subprocess

def interpolate_vol2surf(subject_id):
    io = utils.SubjectIO(subject_id)

    fmri_func = io.data.path.root / f"sub-{io.subject}" / "ses-mri" / "func"
    realign_bold_mean = list(fmri_func.glob("mean*.nii"))
    assert len(realign_bold_mean) == 1
    realign_bold_mean = realign_bold_mean[0]
    fmri_runs = fmri_func.glob("sa*_bold.nii")
    regdat = fmri_func / 'register.dat'

    # create register.dat (reg. between scanner RAS and FS RAS)
    call = " && ".join([
        "module load freesurfer",
        f"export SUBJECTS_DIR={Config.path.FREESURFER}",
        f"tkregister2 --mov {realign_bold_mean} --s sub-{io.subject} --regheader --noedit --reg {regdat}",
    ])
    subprocess.run(["bash", "-c", call])

    # now map to surface
    # sample from white matter (0) to pial surface (1) in steps of 0.1
    for fmri_run in fmri_runs:
        lh_name = fmri_func / f"surf_lh_{fmri_run.name}"
        rh_name = fmri_func / f"surf_rh_{fmri_run.name}"
        call = " && ".join([
            "module load freesurfer",
            f"export SUBJECTS_DIR={Config.path.FREESURFER}",
            f"mri_vol2surf --src {fmri_run} --out {lh_name} --srcreg {regdat} --hemi lh --float2int round --projfrac-avg 0 1 0.1",
            f"mri_vol2surf --src {fmri_run} --out {rh_name} --srcreg {regdat} --hemi rh --float2int round --projfrac-avg 0 1 0.1"
        ])
        subprocess.run(["bash", "-c", call])

        # construct MNE source estimate object
        src = mne.read_source_spaces(io.data.get_filename(stage="forward", forward="mne", suffix="src"))
        lh_data = nib.load(lh_name).get_fdata().squeeze()[src[0]["vertno"]]
        rh_data = nib.load(rh_name).get_fdata().squeeze()[src[1]["vertno"]]

        stc = mne.SourceEstimate(
            np.concatenate([lh_data, rh_data]),
            vertices=[s["vertno"] for s in src],
            tmin=0,
            tstep=2, # TR=2000 ms
            subject=f"sub-{io.subject}",
        )
        stc.save(fmri_func / f"surf_{fmri_run.stem}", overwrite=True)

        lh_name.unlink()
        rh_name.unlink()


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    interpolate_vol2surf(subject_id)