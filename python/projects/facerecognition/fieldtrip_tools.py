import subprocess

import mne
import scipy.io

from projects.anateeg.utils import load_module
from projects.anateeg.fieldtrip_tools import (
    export_duneuro_libs,
    format_matlab_call,
    get_ft_subject_dir,
)

from simnibs.simulation import eeg_mne_tools

from projects.mnieeg.forward import morph_forward
from projects.anateeg.mne_tools import update_source_morph

from projects.facerecognition import utils
from projects.facerecognition.config import Config


def make_headmodel(subject_id):
    """Slightly modified version of the one in
    projects.anateeg.fieldtrip_tools such that it graps the T1w image from the
    correct location.
    """

    io = utils.SubjectIO(subject_id)
    io.bids["mri"].update(extension="nii.gz")
    t1w = io.bids["mri"].fpath

    params = dict(
        path_to_matlab_function=Config.path.MATLAB_FUNCTIONS,
        subject_dir=get_ft_subject_dir(io, Config.path.FIELDTRIP),
        t1=t1w,
    )
    call = (
        "addpath({path_to_matlab_function}); fieldtrip_make_mesh({subject_dir}, {t1});"
    )
    call = " && ".join(
        [
            export_duneuro_libs(),
            load_module("matlab"),
            format_matlab_call(call, params),
        ]
    )
    subprocess.run(["bash", "-c", call])


def make_forward_solution(subject_id):
    """
    Grab files from the simnibs segmentation directory and save to fieldtrip.
    """

    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward")
    ref = Config.forward.REFERENCE
    surf_dir = io.simnibs[ref].get_path("m2m") / "surfaces"
    subject_dir = get_ft_subject_dir(io, Config.path.FIELDTRIP)

    fsavg = mne.read_source_spaces(Config.path.RESOURCES / "fsaverage_central-src.fif")

    params = dict(
        path_to_matlab_function=Config.path.MATLAB_FUNCTIONS,
        montage=io.data.get_filename(
            forward=Config.forward.REFERENCE, suffix="montage", extension="csv",
        ),
        lh=surf_dir / f"lh.central.{Config.forward.SUBSAMPLING}.gii",
        rh=surf_dir / f"rh.central.{Config.forward.SUBSAMPLING}.gii",
        subject_dir=subject_dir,
    )
    call = "addpath({path_to_matlab_function}); fieldtrip_make_forward_solution({subject_dir}, {montage}, {lh}, {rh});"
    call = " && ".join(
        [
            export_duneuro_libs(),
            load_module("matlab"),
            format_matlab_call(call, params),
        ]
    )
    subprocess.run(["bash", "-c", call])

    # pos, tris, brainstructure, brainstructurelabel, unit, inside
    is_inside = scipy.io.loadmat(subject_dir / "src.mat")["src"][0][0][5]
    is_inside = is_inside.reshape(2, -1).astype(int)

    # Convert to MNE format
    info = io.data.get_filename(suffix="info")
    trans = io.data.get_filename(suffix="trans")

    src = mne.read_source_spaces(io.data.get_filename(forward=ref, suffix="src"))
    for i, s in enumerate(src):
        s["inuse"] = is_inside[i]
        s["vertno"] = s["vertno"][s["inuse"].astype(bool)]
        s["nuse"] = s["inuse"].sum()
    morph = mne.read_source_morph(
        io.data.get_filename(forward=ref, suffix="morph", extension="h5")
    )
    update_source_morph(morph, src)

    forward = prepare_forward(subject_id)
    fwd = eeg_mne_tools.make_forward(forward, src, info, trans)

    io.data.update(forward="fieldtrip")

    src.save(io.data.get_filename(suffix="src"), overwrite=True)
    morph.save(io.data.get_filename(suffix="morph", extension="h5"), overwrite=True)
    mne.write_forward_solution(io.data.get_filename(suffix="fwd"), fwd, overwrite=True)
    morph_forward(fwd, morph, fsavg)
    mne.write_forward_solution(
        io.data.get_filename(space="fsaverage", suffix="fwd"), fwd, overwrite=True,
    )


def prepare_forward(subject_id):

    io = utils.SubjectIO(subject_id)
    subject_dir = get_ft_subject_dir(io, Config.path.FIELDTRIP)

    # chantype, chanunit, elecpos, label, m, type, chanpos
    ch_names = scipy.io.loadmat(subject_dir / "sens.mat")["sens"][0][0][3]
    ch_names = [i[0] for i in ch_names.squeeze()]

    # (channels, sources, 3 )
    lf = scipy.io.loadmat(subject_dir / "forward.mat")["forward"]
    nchan, nsrc, nori = lf.shape

    return dict(
        data=lf,
        ch_names=ch_names,
        n_channels=nchan,
        n_sources=nsrc,
        n_orientations=nori,
        subsampling=Config.forward.SUBSAMPLING,
    )

