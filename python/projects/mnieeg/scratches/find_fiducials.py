import json
from pathlib import Path
import subprocess
import sys

import mne
import nibabel as nib
import skimage.measure
import pyvista as pv


def find_fiducials(subject_id: int):
    subject_id = int(subject_id)
    subject = f"sub-{subject_id:02d}"
    root = Path("/mnt/projects/PhaTMagS/jesper/")
    fname_t1w = (
        root / "data" / subject / "ses-mri" / "anat" / f"{subject}_ses-mri_T1w.nii.gz"
    )
    with open(root / "data" / "phatmags_to_bids.json", "r") as f:
        s = json.load(f)[f"{subject_id:02d}"]["original_subject"]
    print(f"subject : {s}")

    print("Extracting surface")
    t1w = nib.load(fname_t1w)
    trans = mne.transforms.Transform("unknown", "unknown", t1w.affine)
    v, f, _, _ = skimage.measure.marching_cubes(
        t1w.get_fdata(), level=100, step_size=3, allow_degenerate=False
    )
    vv = mne.transforms.apply_trans(trans, v)
    # nib.freesurfer.write_geometry('/mrhome/jesperdn/temp/sub01surf',vv,f)
    skin = pv.helpers.make_tri_mesh(vv, f)
    print("Smoothing")
    skin.smooth(300, inplace=True)
    skin.save("/mrhome/jesperdn/skin.vtk")

    print("Running Paraview")
    subprocess.run(
        "/mnt/depot64/paraview/paraview.5.8.0/bin/paraview /mrhome/jesperdn/skin.vtk".split()
    )


if __name__ == "__main__":
    find_fiducials(sys.argv[1])


