import sys

from simnibs.segmentation import charm_main

from projects.mnieeg import utils
from projects.mnieeg.config import Config


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    io = utils.SubjectIO(subject_id)

    subject_dir = io.simnibs.get_path("m2m")
    if not subject_dir.exists():
        subject_dir.mkdir(parents=True)

    io.data.update(task=None, session="mri", extension="nii.gz")
    t1w = io.data.get_filename(suffix="T1w")
    t2w = io.data.get_filename(suffix="T2w")

    subject_ini = Config.path.RESOURCES / "charm" / f"sub-{io.subject}.ini"
    settings_file = str(subject_ini) if subject_ini.exists() else None

    charm_main.run(
        str(subject_dir),
        str(t1w),
        str(t2w),
        registerT2=True,
        initatlas=True,
        segment=True,
        create_surfaces=True,
        mesh_image=True,
        usesettings=settings_file,
        noneck=False,
        options_str=None,
    )

