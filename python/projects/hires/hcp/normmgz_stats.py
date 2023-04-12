import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from utils import get_path



def normmgz_tissue_stats():
    path = get_path()
    subjects = np.loadtxt(path["HCP_DATA"] / "subjects.txt", dtype=int)
    n = len(subjects)

    tissues = ["wm", "gm", "csf"]
    mean = {}.fromkeys(tissues, np.zeros(n))
    median = {}.fromkeys(tissues, np.zeros(n))
    std = {}.fromkeys(tissues, np.zeros(n))

    print_status = dict(zip(np.round(np.linspace(0, 1113, 11))[1:].astype(int), np.linspace(0,100,11, dtype=int)[1:]))

    for i,s in enumerate(subjects):
        s = str(s)
        img = nib.load(path["HCP_CHARM"] / f"m2m_{s}" / "T1.nii.gz")
        norm_mgz = nib.load(path["HCP_DATA"] / s / "T1w" / s / "mri" / "norm.mgz")
        normr_mgz = resample_from_to(norm_mgz, img)
        data = normr_mgz.get_fdata()

        prob_dir = path["HCP_CHARM"] / f"m2m_{s}" / "segmentation" / "probabilities"
        for t in tissues:
            img = nib.load(prob_dir / f"simnibs_posterior_{t}.nii.gz")
            d = data[img.get_fdata() > 0.5]
            mean[t][i] = d.mean()
            median[t][i] = np.median(d)
            std[t][i] = d.std()

        if i in print_status:
            print(f"{print_status[i]:3d} %")

    return mean, median, std

plt.hist(mean, bins=100)

grand_median = {k:v.mean() for k,v in median.items()}

def synthesize_t1():
    path = get_path()
    subjects = np.loadtxt(path["HCP_DATA"] / "subjects.txt", dtype=int)
    s = str(subjects[i])

    prob_dir = path["HCP_CHARM"] / f"m2m_{s}" / "segmentation" / "probabilities"

    norm_mgz = nib.load(path["HCP_DATA"] / s / "T1w" / s / "mri" / "norm.mgz")


    tissues = ["wm", "gm", "csf"]
    tissue_vals = [1,2,3]
    probs = {t: nib.load(prob_dir / f"simnibs_posterior_{t}.nii.gz") for t in tissues}
    vals = dict(zip(tissues, tissue_vals))

    data = np.zeros_like(probs[tissues[0]])
    for t in tissues:
        data += probs[t]*vals[t]

    synt1 = nib.Nifti1Image(data, affine)
    synt1 = resample_from_to(synt1, norm_mgz)

    # split
    hemisphere_crop(synt1, "lh")


    synt1.to_filename()


mri_to_vox = np.linalg.inv(norm_mgz.affine)
mni305_to_mni152 = utils.mni305_to_mni152
mni305_to_vox = mri_to_vox @ mni152_to_mri @ mni305_to_mni152

template_vertices =
crop_size = (96, 144, 208)
mapping = "6-to-1"

def hemisphere_crop(img, hemi, ):

    prepare_template()

    mni305_to_vox
