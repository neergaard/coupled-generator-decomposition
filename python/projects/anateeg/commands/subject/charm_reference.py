from pathlib import Path
import shutil
import sys

from simnibs.segmentation import charm_main
from simnibs.utils.file_finder import SubjectFiles

from projects.anateeg import utils
from projects.anateeg.config import Config


def charm_reference(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(task=None, session="mri", extension="nii.gz")

    m2m_ref = io.simnibs[Config.forward.REFERENCE].get_path("m2m")
    if not m2m_ref.exists():
        m2m_ref.mkdir(parents=True)
    sf_ref = SubjectFiles(subpath=m2m_ref)
    sf = SubjectFiles(subpath=io.simnibs["charm"].get_path("m2m"))

    # Create an m2m folder from scratch using some of the files from the CHARM
    # run on the T1 and T2. Based on these files, the surfaces and mesh can be
    # generated
    for d in ("label_prep_folder", "surface_folder", "mni_transf_folder"):
        dd = Path(getattr(sf_ref, d))
        if not dd.exists():
            dd.mkdir()

    shutil.copy(
        io.data.get_filename(suffix="usegmap"), sf_ref.tissue_labeling_upsampled
    )
    required_files = [
        "norm_image",
        "cereb_mask",
        "hemi_mask",
        "reference_volume",
        "mni2conf_nonl",
        "conf2mni_nonl",
    ]
    for f in required_files:
        shutil.copy(getattr(sf, f), getattr(sf_ref, f))

    fname_settings = Config.path.RESOURCES / "charm_reference" / "settings.ini"

    charm_main.run(
        str(m2m_ref),
        registerT2=False,
        initatlas=False,
        segment=False,
        create_surfaces=True,
        mesh_image=True,
        usesettings=str(fname_settings),
        noneck=False,
        init_transform=None,
        options_str=None,
    )


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    charm_reference(subject_id)
