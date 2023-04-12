import subprocess
import sys

from projects.anateeg.utils import load_module, parse_args
from projects.facerecognition import utils
from projects.facerecognition.config import Config


def fs_recon(subject_id, config):

    io = utils.SubjectIO(subject_id)
    io.bids["mri"].update(extension="nii.gz")
    t1w = io.bids["mri"].fpath

    if not config.path.FREESURFER.exists():
        config.path.FREESURFER.mkdir()

    params = dict(
        subject=f"sub-{io.subject}", t1w=t1w, freesurfer_dir=config.path.FREESURFER
    )
    call = " && ".join(
        [
            load_module("freesurfer"),
            "recon-all -s {subject} -i {t1w} -sd {freesurfer_dir} -all".format(
                **params
            ),
        ]
    )
    subprocess.run(["bash", "-c", call])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    fs_recon(subject_id, Config)

