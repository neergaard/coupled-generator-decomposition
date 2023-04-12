from pathlib import Path
import numpy as np

from utils import get_path




path = get_path()
subjects = np.loadtxt(path["HCP_DATA"] / "subjects.txt", dtype=int)

mri_files = ("T1w_acpc_dc_restore.nii.gz", "T2w_acpc_dc_restore.nii.gz")
surf_files = tuple(f"{h}h.{f}" for h in ("l", "r") for f in ("curv", "curv.pial", "pial", "thickness", "white"))

data_dir = paths["HCP"] / "data"
for s in subjects:
    s = str(s)

    mri_dir = data_dir / s / "T1w"
    surf_dir = data_dir / s / "T1w" / s / "surf"

    for f in mri_files:
        ff = mri_dir / f
        assert ff.exists(), f"{ff} does not exist!"
    for f in surf_files:
        ff = surf_dir / f
        assert ff.exists(), f"{ff} does not exist!"

print("All files exist")