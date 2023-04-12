from simnibs.segmentation import charm_main

from projects.facerecognition import utils
from projects.facerecognition.config import Config


if __name__ == "__main__":
    io = utils.SubjectIO("mni152")

    t1w = Config.path.RESOURCES / "mni152" / "T1.nii.gz"
    assert t1w.exists()

    subject_dir = io.simnibs["template"].get_path("m2m")
    if not subject_dir.exists():
        subject_dir.mkdir(parents=True)

    charm_main.run(
        str(subject_dir),
        str(t1w),
        None,
        registerT2=True,
        initatlas=True,
        segment=True,
        create_surfaces=True,
        mesh_image=True,
        usesettings=None,
        noneck=False,
        options_str=None,
    )

