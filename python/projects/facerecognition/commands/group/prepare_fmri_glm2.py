import subprocess

from projects.anateeg.utils import load_module
from projects.anateeg.fieldtrip_tools import format_matlab_call

from projects.facerecognition import utils
from projects.facerecognition.config import Config


if __name__ == "__main__":
    io = utils.GroupIO()

    params = dict(
        FMRI_FUNCTIONS_DIR=Config.path.FMRI_FUNCTIONS,
        bids_root=io.bids["mri"].root,
        data_root=Config.path.DATA,
    )

    call = "addpath({FMRI_FUNCTIONS_DIR}); fmri_glm2({bids_root}, {data_root})"
    call = format_matlab_call(call, params)
    call = " && ".join([load_module("spm"), load_module("matlab"), call])

    subprocess.run(["bash", "-c", call])
