import json
import shutil

import nibabel as nib

from projects.anateeg import utils
from projects.anateeg.config import Config

if __name__ == "__main__":

    subjects_dir = Config.path.SKULL_RECO / "subjects"
    subject_dirs = subjects_dir.glob("X*")

    target_dir = Config.path.DATA
    if not target_dir.exists():
        target_dir.mkdir()

    v_names = ("t1", "t2", "segmentation")

    n = 0
    included, session = {}, {}
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        coreg_dir = subject_dir / "coreg_0.85"
        t1 = coreg_dir / "T1_nobiascorr.nii.gz"
        t2 = coreg_dir / "T2_nofatsat_nobiascorr.nii.gz"
        ref = subject_dir / "segmentation" / "segmentation.nii"
        vols = (t1, t2, ref)
        if all(v.exists() for v in vols):
            n += 1
            io = utils.SubjectIO(n, read_session=False)
            io.data.update(session="mri", extension="nii.gz")
            io.data.path.ensure_exists()

            shutil.copy(t1, io.data.get_filename(suffix="t1w"))
            shutil.copy(t2, io.data.get_filename(suffix="t2w"))
            img = nib.load(ref)
            img.to_filename(io.data.get_filename(suffix="seg"))

            included[io.subject] = subject_dir.stem
            session[io.subject] = "01"
            print(subject_dir.stem, "-> included")
        else:
            print(subject_dir.stem, "-> not included")

    with open(target_dir / "subjects.json", "w") as f:
        json.dump(included, f, indent=4)

    with open(target_dir / "include.json", "w") as f:
        json.dump(session, f, indent=4)

    print(f"Copied {n} subjects")

