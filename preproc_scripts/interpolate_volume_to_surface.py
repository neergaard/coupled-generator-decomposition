import sys
import warnings

import mne
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.ckdtree import cKDTree

from simnibs.simulation.eeg import apply_trans
from simnibs.utils.file_finder import SubjectFiles

from projects.base.geometry import vertex_normal

from projects.facerecognition.config import Config
from projects.facerecognition import utils
from projects.facerecognition_dtu import utils as utils_dtu

from projects.anateeg.utils import parse_args

def interpolate_fmri_activations(subject):

    # t_img = "spmT_0006"  # faces > scrambled for glm1 and glm2
    # img_prefix = "dswa"
    # scale_in = 1

    kw_morph = dict(stage="forward", forward="mne", suffix="morph", extension="h5")
    kw_src = dict(stage="forward", forward="mne", suffix="src", extension="fif")

    print(f"Interpolating subject {subject}")

    io = utils.SubjectIO(subject)
    iodtu = utils_dtu.SubjectIO(subject)

    m2m_dir = io.simnibs["charm"].get_path("m2m")
    sf = SubjectFiles(subpath=str(m2m_dir))
    sub = Config.forward.SUBSAMPLING
    surf = {
        s.region: nib.load(s.fn) for s in sf.central_surfaces if s.subsampling == sub
    }

    # get thickness - ugly
    surffull = {
        s.region: nib.load(s.fn) for s in sf.central_surfaces if s.subsampling == sub
    }
    thickness = {
        s.region: nib.freesurfer.read_morph_data(s.fn)
        for s in sf.thickness
    }
    idx = {}
    for h in surf:
        treefull = cKDTree(surffull[h].agg_data("pointset"))
        dist, idx[h] = treefull.query(surf[h].agg_data("pointset"))
        assert np.allclose(dist, 0)
    thickness = {h:thickness[h][idx[h]] for h in thickness}

    morph = mne.read_source_morph(io.data.get_filename(**kw_morph))
    src = mne.read_source_spaces(io.data.get_filename(**kw_src))

    # Get the fMRI t-map
    # if subject == "group":
    #     mri_dir = io.data.path.root / "group" / "ses-mri"
    #     vol = nib.load(mri_dir / "glm2" / f"{t_img}.nii")
    # else:
    mri_dir = io.data.path.get(session="mri")
    out_dir = iodtu.data.path.get(session="mri") / "func"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for i in range(1,10):
        session = f"{i:02d}"
        print(f"Interpolating session {session} to surface")
        vol = nib.load(mri_dir / "func" / f"dswasub-{io.subject}_ses-mri_task-facerecognition_run-{session}_bold.nii")

        # Deform t-map to subject space
        # ...
        interp = interpolate_volume_to_surface(vol, surf, thickness)
        stc = mne.SourceEstimate(
            np.concatenate(list(interp.values())),
            vertices=[s["vertno"] for s in src],
            tmin=0,
            tstep=2, # TR=2000 ms
            subject=f"sub-{io.subject}",
        )
        stc.save(out_dir / f"bold_interp-surf_run-{session}", overwrite=True)


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
        interp[h] = np.ascontiguousarray(np.stack([
            map_coordinates(data[...,i], coords.T).mean(0) for i in range(data.shape[3])
        ]).T)
        if np.isnan(interp[h]).any():
            warnings.warn(
                f"nan values present in interpolated data for {h} surface!",
                RuntimeWarning,
            )
        # interp[h] = interp_data.mean(0)  # .max(0)
    return interp


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    interpolate_fmri_activations(subject_id)