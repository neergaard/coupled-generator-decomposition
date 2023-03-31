# import argparse
import subprocess
import sys

from projects.anateeg.utils import load_module
from projects.anateeg.fieldtrip_tools import format_matlab_call

from projects.facerecognition_dtu import utils
from projects.facerecognition_dtu.config import Config


# def parse_args(argv, prog=None, description=None):
#     """Parse arguments to a function that only takes subject ID as argument."""
#     program = dict(prog=prog, description=description,)
#     arg_subject_id = dict(
#         help="Subject id. Zero will be prepended if necessary "
#         + "(e.g., for sub-01 simply pass 1)."
#     )
#     arg_fmri_function = dict(
#         choices=["fmri_preprocess", "fmri_wavelet_despike", "fmri_glm1"],
#         help="MATLAB function to run",
#     )
#     parser = argparse.ArgumentParser(**program)
#     parser.add_argument("subject-id", **arg_subject_id)
#     parser.add_argument("fmri_function", **arg_fmri_function)
#     return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = utils.parse_args_with_mode(sys.argv)
    subject_id = getattr(args, "subject-id")
    io = utils.SubjectIO(subject_id)

    params = dict(
        FMRI_FUNCTIONS_DIR=Config.path.FMRI_FUNCTIONS,
        subject_id=subject_id,
        bids_root=io.bids["mri"].root,
        data_root=Config.path.DATA,
        fs_root=Config.path.FREESURFER,
    )

    call = (
        "addpath({FMRI_FUNCTIONS_DIR}); "
        + args.mode  # the fmri function to execute
        + "({subject_id}, {bids_root}, {data_root}, {fs_root})"
    )
    call = format_matlab_call(call, params)
    call = " && ".join([load_module("spm"), load_module("matlab"), load_module("freesurfer"), call])
    subprocess.run(["bash", "-c", call])
