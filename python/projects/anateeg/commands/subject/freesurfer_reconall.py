import subprocess
import sys

from projects.anateeg import utils
from projects.anateeg.config import Config


def fs_recon(subject_id, config):

    io = utils.SubjectIO(subject_id)
    t1w = io.data.get_filename(session="mri", suffix="t1w", extension="nii.gz")
    if not config.path.FREESURFER.exists():
        config.path.FREESURFER.mkdir()

    params = dict(
        subject=f"sub-{io.subject}", t1w=t1w, freesurfer_dir=config.path.FREESURFER
    )
    call = " && ".join(
        [
            utils.load_module("freesurfer"),
            "recon-all -s {subject} -i {t1w} -sd {freesurfer_dir} -all".format(
                **params
            ),
        ]
    )
    subprocess.run(["bash", "-c", call])


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    fs_recon(subject_id, Config)

