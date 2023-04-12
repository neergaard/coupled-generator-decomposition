import json
import shutil

import mne
import mne_bids
import nibabel as nib
import numpy as np
import pyvista as pv

from simnibs.simulation import eeg_mne_tools

from projects.facerecognition import utils
from projects.facerecognition.config import Config

from projects.anateeg.simnibs_tools import make_skin_outer_annot


def prepare(subject_id):
    """
    Write Info
    Coregister
    Create EEG layout in MRI space
    """

    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    io.data.path.ensure_exists()

    # io.bids["meg"].update(run=mne_bids.get_entity_vals(io.bids["meg"].root, "run")[0])
    # info = mne_bids.read_raw_bids(io.bids["meg"]).pick_types(eeg=True).info
    epochs = mne.read_epochs(
        io.data.get_filename(stage="preprocess", processing="pa", suffix="epo")
    )
    info = epochs.pick_types(eeg=True).info
    mne.io.write_info(io.data.get_filename(suffix="info"), info)

    mri_head_t = coregister_get_trans(info, io.bids["mri"])
    mne.write_trans(io.data.get_filename(suffix="trans"), mri_head_t)

    io.data.update(suffix="montage", extension="csv")
    fname = io.data.get_filename()
    eeg_mne_tools.prepare_montage(fname, info, mri_head_t)
    # This way we can use the functions from anateeg directly...
    shutil.copy(fname, io.data.get_filename(forward=Config.forward.REFERENCE))

    _ = make_skin_outer_annot(io.simnibs["charm"])


def coregister_get_trans(info, bids_path_mri, refine_trans=False):

    mri_fids = get_mri_fids(bids_path_mri)
    mri_head_t = mne.coreg.coregister_fiducials(info, mri_fids)
    # Refine transformation using digitized head points
    if refine_trans:
        raise NotImplementedError
        # pos_hp = np.stack(dig['r'] for dig in info['dig'] if dig['kind'] == FIFF.FIFFV_POINT_EXTRA)
    return mne.transforms._ensure_trans(mri_head_t)


def get_mri_fids(bids_path_mri):
    # Fiducials in mri coordinates
    # The key says coordinates but it is actually voxel indices
    bids_path_mri.update(extension="json")
    mri_fids = json.load(open(bids_path_mri.fpath))["AnatomicalLandmarkCoordinates"]
    bids_path_mri.update(extension="nii.gz")
    t1w = nib.load(bids_path_mri.fpath)
    for fid in mri_fids:
        # MNE works in m
        mri_fids[fid.lower()] = (
            mri_fids.pop(fid) @ t1w.affine[:3, :3].T + t1w.affine[:3, 3]
        ) * 1e-3
    mri_fids = mne.io._digitization._make_dig_points(**mri_fids, coord_frame="mri")
    return mri_fids


def coregistration_qa(io, info, trans):

    trans = mne.transforms._ensure_trans(trans, "mri", "head")

    skin_mri = make_skin_outer_annot(io.simnibs["charm"])
    skin_meg = skin_mri.copy()
    skin_meg.points = mne.transforms.apply_trans(trans, skin_meg.points * 1e-3) * 1e3

    fids_mri = get_mri_fids(io.bids["mri"])
    fids_mri = np.stack(f["r"] for f in fids_mri)
    fids_meg = mne.transforms.apply_trans(trans, fids_mri) * 1e3
    fids_mri *= 1e3

    trans = mne.transforms._ensure_trans(trans, "head", "mri")

    digs_meg = np.stack(d["r"] for d in info["dig"])
    digs_mri = mne.transforms.apply_trans(trans, digs_meg) * 1e3
    digs_meg *= 1e3

    coreg = pv.MultiBlock()
    coreg["skin_mri"] = skin_mri
    coreg["skin_meg"] = skin_meg
    coreg["fids_mri"] = pv.PolyData(fids_mri)
    coreg["fids_meg"] = pv.PolyData(fids_meg)
    coreg["digs_mri"] = pv.PolyData(digs_mri)
    coreg["digs_meg"] = pv.PolyData(digs_meg)

    return coreg
