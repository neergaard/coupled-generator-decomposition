import sys

import mne

from projects.anateeg.utils import parse_args

from projects.facerecognition_dtu import utils

from projects.facerecognition.config import Config
from projects.facerecognition import utils as utils_orig

# sbatch segfaults when mne tries to execute some parallel code
import numba

numba.set_num_threads(1)

def forward(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    io.data.path.ensure_exists()

    full_subject_id = f"sub-{io.subject}"
    subjects_dir = Config.path.FREESURFER

    epo_kwargs = dict(stage="preprocess", processing="p", suffix="epo")
    info = mne.io.read_info(io.data.get_filename(**epo_kwargs))

    # read from faceregcognition folder!
    ioorig = utils_orig.SubjectIO(subject_id)
    trans_kwargs = dict(stage="forward", suffix="trans")
    trans = mne.read_trans(ioorig.data.get_filename(**trans_kwargs))
    trans = mne.transforms.invert_transform(trans)

    # vox : MRI (voxel) voxel indices
    # mri : MRI (surface RAS) freesurfer coordinates
    # ras : RAS (non-zero origin) real world coordinates (scanner coordinates)
    # (mri_ras_t is tkr-RAS to scanner-RAS)
    # outputs: vox_ras_t, vox_mri_t, mri_ras_t, dims, zooms
    fname = Config.path.FREESURFER / f"sub-{io.subject}" / "mri" / "orig.mgz"
    _, _, mri_ras_t, _, _ = mne._freesurfer._read_mri_info(fname)
    ras_mri_t = mne.transforms.invert_transform(mri_ras_t)

    trans["to"] = ras_mri_t["from"]
    head_mri_t = mne.transforms.combine_transforms(trans, ras_mri_t, trans["from"], ras_mri_t["to"])

    io.data.update(forward="mne")

    src_kwargs = dict(suffix="src")
    morph_kwargs = dict(suffix="morph", extension="h5")
    fwd_kwargs = dict(stage="forward", forward="mne", suffix="fwd")

    src = mne.setup_source_space(full_subject_id, "ico5", subjects_dir=subjects_dir, add_dist="patch")
    mne.write_source_spaces(io.data.get_filename(**src_kwargs), src, overwrite=True)

    # ico=4, default conductivities
    surfs = mne.make_bem_model(full_subject_id, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(info, head_mri_t, src, bem)
    morph = mne.compute_source_morph(fwd["src"], subjects_dir=subjects_dir)

    mne.write_bem_surfaces(io.data.get_filename(suffix="surfs"), surfs, overwrite=True)
    mne.write_forward_solution(io.data.get_filename(**fwd_kwargs), fwd, overwrite=True)
    morph.save(io.data.get_filename(**morph_kwargs), overwrite=True)

    # fids_mri = get_mri_fids(io.bids["mri"])
    # fids_mri = np.stack(f["r"] for f in fids_mri)
    # fids_mri = mne.transforms.apply_trans(ras_mri_t, fids_mri)

    """
    meg_dev = np.stack(d["loc"][:3] for d in info["chs"] if "MEG" in d["ch_name"])
    meg_head = mne.transforms.apply_trans(info["dev_head_t"], meg_head)
    meg_mri = mne.transforms.apply_trans(head_mri_t, meg_head)
    eeg_head = np.stack(d["loc"][:3] for d in info["chs"] if "EEG" in d["ch_name"])
    eeg_mri = mne.transforms.apply_trans(head_mri_t, eeg_head)

    p = pv.Plotter(notebook=False)
    p.add_mesh(pv.make_tri_mesh(surfs[0]["rr"], surfs[0]["tris"]), color="pink", opacity=0.75)
    p.add_mesh(pv.make_tri_mesh(surfs[1]["rr"], surfs[1]["tris"]), opacity=0.75)
    p.add_mesh(pv.make_tri_mesh(surfs[2]["rr"], surfs[2]["tris"]), color="black", opacity=0.75)
    p.add_mesh(pv.make_tri_mesh(src[0]["rr"], src[0]["tris"]), color="blue")
    p.add_mesh(pv.make_tri_mesh(src[1]["rr"], src[1]["tris"]), color="blue")
    p.add_mesh(pv.PolyData(meg_mri), color="green")
    p.add_mesh(pv.PolyData(eeg_mri), color="red")
    p.show()
    """


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    forward(subject_id)