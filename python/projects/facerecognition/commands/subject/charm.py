import sys

from simnibs.segmentation import charm_main

from projects.mnieeg.utils import parse_args
from projects.facerecognition import utils


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")

    io = utils.SubjectIO(subject_id)
    subject_dir = io.simnibs["charm"].get_path("m2m")
    if not subject_dir.exists():
        subject_dir.mkdir(parents=True)

    io.bids["mri"].update(extension="nii.gz")
    t1w = io.bids["mri"].fpath
    PDw = io.simnibs["charm"].get_path("subject") / "flash" / "PD.nii.gz"
    assert PDw.exists()

    # subject_ini = Config.path.RESOURCES / "charm" / f"sub-{io.subject}.ini"
    # settings_file = str(subject_ini) if subject_ini.exists() else None

    charm_main.run(
        str(subject_dir),
        str(t1w),
        str(PDw),
        registerT2=True,
        initatlas=True,
        segment=True,
        create_surfaces=True,
        mesh_image=True,
        usesettings=None,  # settings_file,
        noneck=False,
        options_str=None,
    )

