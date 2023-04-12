import warnings

# import mne
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from simnibs.simulation.eeg_mne_tools import setup_source_space
from simnibs.simulation.eeg import apply_trans
from simnibs.utils.file_finder import SubjectFiles

from projects.base.geometry import vertex_normal

from projects.facerecognition.config import Config
from projects.facerecognition import utils


vol = nib.load(
    "/mrhome/jesperdn/INN_JESPER/projects/facerecognition/data/group/ses-mri/glm2/spmT_0006.nii"
)
m2m_dir = "/mrhome/jesperdn/INN_JESPER/projects/facerecognition/simnibs_template/sub-mni152/m2m_sub-mni152"
_, morph = setup_source_space(m2m_dir, morph_to_fsaverage=Config.inverse.FSAVERAGE)




def interpolate_fmri_activations():

    t_img = "spmT_0006"  # faces > scrambled for glm1 and glm2
    scale_in = 2
    fs_sub = 7  # Config.inverse.FSAVERAGE
    # ref = Config.forward.REFERENCE

    # kw_morph = dict(stage="forward", forward=ref, suffix="morph", extension="h5")

    io = utils.GroupIO()

    io_mni = utils.SubjectIO("mni152")
    m2m_dir = io_mni.simnibs["template"].get_path("m2m")
    sf_mni = SubjectFiles(subpath=str(m2m_dir))
    # morph = mne.read_source_morph(io_mni.data.get_filename(**kw_morph))
    _, morph = setup_source_space(m2m_dir, morph_to_fsaverage=fs_sub)

    surf = {s.region: nib.load(s.fn) for s in sf_mni.central_surfaces}
    thickness = {
        s.region: nib.freesurfer.read_morph_data(s.fn) for s in sf_mni.thickness
    }

    rows = utils.fsaverage_as_index(fs_sub)
    df = pd.DataFrame(index=rows, columns=io.subjects, dtype=float)
    df_group = pd.DataFrame(index=rows, columns=["group"], dtype=float)

    for subject in io.subjects + ["group"]:
        print(f"Interpolating subject {subject}")

        # m2m_dir = io.simnibs[ref].get_path("m2m")
        # sf = SubjectFiles(subpath=str(m2m_dir))
        # sub = Config.forward.SUBSAMPLING
        # surf = {
        #     s.region: nib.load(s.fn) for s in sf.central_surfaces if s.subsampling == sub
        # }
        # thickness = {
        #     s.region: nib.freesurfer.read_morph_data(s.fn)
        #     for s in sf.thickness
        #     if s.subsampling == sub
        # }

        io = utils.SubjectIO(subject)

        # morph = mne.read_source_morph(io.data.get_filename(**kw_morph))

        # Get the fMRI t-map
        if subject == "group":
            mri_dir = io.data.path.root / "group" / "ses-mri"
            vol = nib.load(mri_dir / "glm2" / f"{t_img}.nii")
        else:
            mri_dir = io.data.path.get(session="mri")
            vol = nib.load(mri_dir / "func" / "glm" / f"{t_img}.nii")

        # Deform t-map to subject space
        # ...

        interp = interpolate_volume_to_surface(vol, surf, thickness, scale_in=scale_in)

        # Morph to fsaverage (ensure correction lh/rh ordering for df)
        interp = morph.morph_mat @ np.concatenate(
            [interp[h] for h in rows.unique("Hemi")]
        )

        if subject == "group":
            df_group[subject] = interp
        else:
            df[subject] = interp

    res_dir = Config.path.RESULTS
    if not res_dir.exists():
        res_dir.mkdir()

    df.to_pickle(res_dir / "fmri_fsaverage_subject.pickle")
    df_group.to_pickle(res_dir / "fmri_fsaverage_group.pickle")


def interpolate_volume_to_surface(
    vol, surf, thickness, n_samples=11, scale_in=1, scale_out=1
):
    """

    vol : nibabel.Nifti1Image

    surf : dict
        Dictionary of nibabel gifti objects or dictionary of dicts with entries
        `points` and `tris`.
    thickness : dict
        Dictionary of arrays.

    """

    inv_affine = np.linalg.inv(vol.affine)
    data = vol.get_fdata()

    sampling_points = np.linspace(0, n_samples, n_samples)
    interp = {}
    for h in surf:
        if isinstance(surf[h], nib.GiftiImage):
            v, f = surf[h].agg_data()
        elif isinstance(surf[h], dict):
            v, f = surf["points"], surf["tris"]
        else:
            raise ValueError
        n = vertex_normal(v, f)
        vin = v - scale_in * thickness[h][:, None] * n
        # vout = v + scale_out * thickness[h][:, None] * n

        d = (scale_in + scale_out) * thickness[h]  # distance from vin to vout
        n_scaled = n * d[:, None] / n_samples
        # Sampling points
        v_samp = vin[:, None] + n_scaled[:, None] * sampling_points[None, :, None]
        coords = apply_trans(inv_affine, v_samp)
        interp_data = map_coordinates(data, coords.T)
        if np.isnan(interp_data).any():
            warnings.warn(
                f"nan values present in interpolated data for {h} surface!",
                RuntimeWarning,
            )
        interp[h] = interp_data.mean(0)  # .max(0)
    return interp
