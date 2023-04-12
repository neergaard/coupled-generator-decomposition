import sys

import nibabel as nib
import nibabel.processing
import numpy as np

from projects.anateeg import utils
from projects.anateeg.config import Config

LABEL_GM = 3
LABEL_WM = 4
LABEL_CSF = 5

# RPA, LPA, NASION, INION

# 46, 309, 221
# 37, 289, 138
# -60 269 214


# 78.91, 10.68, -41.36
# -68.99, 9.48, -34.17
# 5.07, 105.91, -14.72
# 3.28, -89.97, -19.79


# manual segmentation to tissue_labeling_upsampled tissues
tissue_mapper = {
    1: 0,  # inner air
    2: 9,  # veins
    3: 2,  # gm
    4: 1,  # wm
    5: 3,  # csf
    6: 6,  # eyes
    7: 5,  # skin
    8: 10,  #
    9: 5,  # mucus
    10: 1,  # optical nerve
    11: 5,  # fat layer (outer skin)
    12: 1,  # spinal cord
    14: 9,  # veins
    15: 7,  # compact bone
    16: 8,  # spongy bone
    17: 0,  # outer air
}


def update_manual_segmentation(subject_id):
    """Update the manual segmentations. The main purpose is to 'open up' sulci.
    The following steps are performed

    - Enforce GM within cortical GM from FS
    - The GM voxels outside of this mask which overlap with FS WM but not FS
    cerebellum (GM and WM) is relabeled to WM.
    - The remaining 'unlabeled' voxels are labeled CSF.

    This might over-segment CSF a bit - particularly around cerebellum (as the
    FS segmentation is not as accurate here) and perhaps in the nasal area?
    """
    io = utils.SubjectIO(subject_id)
    io.data.update(session="mri", extension="nii.gz")

    manual_labels = list(tissue_mapper.keys())
    map_array = np.zeros(max(manual_labels) + 1, dtype=int)
    map_array[manual_labels] = list(tissue_mapper.values())

    img_seg = nib.load(io.data.get_filename(suffix="seg"))
    img_aseg = nib.load(
        Config.path.FREESURFER / f"sub-{io.subject}" / "mri" / "aseg.mgz"
    )
    img_aseg = nibabel.processing.resample_from_to(img_aseg, img_seg, order=0)
    seg = img_seg.get_fdata().astype(np.uint16)
    aseg = img_aseg.get_fdata().astype(np.uint16)

    seg_mapped = nib.Nifti1Image(map_array[seg], img_seg.affine, img_seg.header)
    seg_mapped.to_filename(io.data.get_filename(suffix="segmap"))

    # choroid plexus??? 31, 63
    # fmt: off
    fs_gm = np.isin(aseg, (3, 42, 17, 53, 18, 54))  # ribbon, hippocampus, amygdala
    fs_wm = np.isin(
        aseg,
        (2, 41, 12, 51, 13, 52, 26, 58, 28, 60, 10, 49, 251, 252, 253, 254, 255, 11, 50),
    )  # wm, subcortical stuff
    # fmt: on
    fs_cerebellum = np.isin(aseg, (7, 8, 46, 47))  # wm and gm

    seg[fs_gm] = LABEL_GM
    unassigned = (seg == LABEL_GM) & ~fs_gm & ~fs_cerebellum
    # seg[unassigned] = LABEL_UNASSIGN
    seg[unassigned & fs_wm] = LABEL_WM
    seg[unassigned & ~fs_wm] = LABEL_CSF  # some of it could perhaps be vein

    useg = nib.Nifti1Image(seg, img_seg.affine, img_seg.header)
    useg.to_filename(io.data.get_filename(suffix="useg"))

    useg = nib.Nifti1Image(map_array[seg], img_seg.affine, img_seg.header)
    useg.to_filename(io.data.get_filename(suffix="usegmap"))


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    update_manual_segmentation(getattr(args, "subject-id"))
