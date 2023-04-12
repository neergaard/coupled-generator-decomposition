import itertools
import subprocess
import sys

import nibabel as nib

from projects.anateeg.utils import load_module, parse_args
from projects.facerecognition import utils


def prepare_flash(subject_id):
    io = utils.SubjectIO(subject_id)

    subject_flash_dir = io.simnibs["charm"].get_path("subject") / "flash"
    if not subject_flash_dir.exists():
        subject_flash_dir.mkdir(parents=True)

    anat_dir = io.bids["mri"].directory
    tr = [20]
    fa = [5, 30]
    te = [1.85, 4.15, 6.45, 8.75, 11.05, 13.35, 15.65]
    images = sorted(anat_dir.glob("*FLASH.nii.gz"))

    fitparms = (
        "mri_ms_fitparms "
        + " ".join(
            [
                f"-tr {i} -te {k} -fa {j} {m}"  # NB. order changed
                for (i, j, k), m in zip(itertools.product(tr, fa, te), images)
            ]
        )
        + f" {subject_flash_dir}"
    )
    call = " && ".join([load_module("freesurfer"), fitparms])
    subprocess.run(["bash", "-c", call])

    pd = nib.load(subject_flash_dir / "PD.mgz")
    nib.save(pd, subject_flash_dir / "PD.nii.gz")

    # Move output files to simnibs directory
    # call = ' && '.join([
    #     f"mv {anat_dir}/*.lta {subject_flash_dir}",
    #     f"mv {anat_dir}/*.mgz {subject_flash_dir}",
    #     mri_convert {subject_fl}/PD.mgz {subject_flash_dir}/PD.nii.gz,
    #     mv $SIMNIBS_DIR/$subject/flash/PD.nii.gz $SIMNIBS_DIR/$subject,
    #     cp $FLASH_DIR/*T1w.nii.gz $SIMNIBS_DIR/$subject/T1.nii.gz,
    # ])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    prepare_flash(subject_id)
