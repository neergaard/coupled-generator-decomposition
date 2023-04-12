import nibabel as nib
from projects.mnieeg import utils
from projects.mnieeg.config import Config
from simnibs.simulation import eeg
import sys

if __name__ == "__main__":

    subject_id = int(sys.argv[1])
    io = utils.SubjectIO(subject_id)

    print("Making montage")
    montage = eeg.make_montage(Config.forward.MONTAGE_SIMNIBS)
    montage.add_landmarks(Config.forward.LANDMARKS)
    # montage.ch_pos = []
    # montage.ch_types = []
    # montage.ch_names = []
    mni2mri = nib.load(io.simnibs.get_path("m2m") / "toMNI" / "MNI2Conform_nonl.nii.gz")
    print("Deforming")
    montage.apply_deform(mni2mri)

    mapper = {"Nz": "nasion", "LPA": "lpa", "RPA": "rpa", "Iz": "inion"}
    for k, v in montage.landmarks.items():
        print(f"{mapper[k]}=[{', '.join(v.astype(str))}]")
