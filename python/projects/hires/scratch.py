from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np

subjectlist = "subjects_sub.txt"
src = "s3://hcp-openaccess/HCP_1200"
dest = "/mnt/projects/INN/jesper/nobackup/HiRes/hcp"
subject = 100307


while read subject; do # 'subject' is the variable name
    echo "Fetching data for $subject"
    aws s3 sync $hcp1200_root/$subject/T1w/ $dest/$subject/T1w/$subject --exclude="*" --include="T*w_acpc_dc_restore.nii.gz"
    # freesurfer data
    aws s3 sync $hcp1200_root/$subject/T1w/$subject/ $dest/$subject/T1w/$subject --exclude="*" --include="*white"
    aws s3 sync $hcp1200_root/$subject/T1w/$subject/ $dest/$subject/T1w/$subject --exclude="*" --include="*pial"
done < $subjectlist


root = Path("/mrhome/jesperdn/INN_JESPER/nobackup/HiRes/20221214_HiRes_forOula/")
nifti = root / "NIFTI"

# MP2RAGE
echoes = ["1008", "2908"]
magnitude = [nib.load(nifti / f"_MP2RAGE_04mm_20221214105951_901_t{echo}.nii.gz") for echo in echoes]
phase = [nib.load(nifti / f"_MP2RAGE_04mm_20221214105951_901_ph_t{echo}.nii.gz") for echo in echoes]

n = 2**16-1
def normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())
mdata = [normalize(m.get_fdata())-0.5 for m in magnitude]
mp2rage = np.nan_to_num(
    mdata[0] * mdata[1] / (mdata[0]**2 + mdata[1]**2)
)
mp2rage = nib.Nifti1Image(mp2rage.astype(np.float32), magnitude[0].affine)
mp2rage.to_filename(nifti / "MP2RAGE.nii.gz")

# T2*

magn = nib.load(nifti / "_T2_w0.2x0.2x1mm_20221214105951_1101.nii.gz")
re = nib.load(nifti / "_T2_w0.2x0.2x1mm_20221214105951_1101_real.nii.gz")
affine = re.affine
im = nib.load(nifti / "_T2_w0.2x0.2x1mm_20221214105951_1101_imaginary.nii.gz")

magnitude = np.sqrt(re.get_fdata() ** 2 + im.get_fdata() ** 2)
magnitude = nib.Nifti1Image(magnitude, affine)
# magnitude = iso_upsampling(magnitude)
magnitude.to_filename(nifti / "T2star_magnitude.nii.gz")

phase = np.arctan2(im.get_fdata(), re.get_fdata())  # argument (phase)
phase = nib.Nifti1Image(phase, affine)
# phase = iso_upsampling(phase)
phase.to_filename(nifti / "T2star_phase.nii.gz")



# SWI

# isotropic resampling to minimum voxel size

n_echoes = 4
name = dict(
    magnitude="_SWIp-0p6-opt_APfoldpver_20221214105951_1201_e{:d}.nii.gz",
    phase="_SWIp-0p6-opt_APfoldpver_20221214105951_1201_e{:d}_phase.nii.gz",
)
imgs = nib.concat_images(
    [
        nib.load(nifti / name["magnitude"].format(echo))
        for echo in range(1, n_echoes + 1)
    ]
)

def iso_upsampling(img):
    vx_size = img.header.get_zooms()
    assert np.isclose(vx_size[0], vx_size[1])

    affine = img.affine.copy()
    # if not np.isclose():
    affine[:, 2] *= vx_size[0] / vx_size[2]
    shape = list(img.shape)
    shape[2] = int(np.ceil(shape[2] * vx_size[2] / vx_size[0]))

    img_iso2 = nibabel.processing.resample_from_to(img, (shape, affine))
    # assert np.isclose(img_iso2.shape[0], img_iso2.shape[2])
    img_iso2.dataobj[:][img_iso2.get_fdata() < 0] = 0
    return img_iso2

img_iso2 = iso_upsampling()
img_iso2.to_filename(nifti / "SWI.nii.gz")

# imgs_iso = nibabel.processing.resample_to_output(imgs, [vx_size[0], vx_size[1], vx_size[0], vx_size[3]])

# run through charm
