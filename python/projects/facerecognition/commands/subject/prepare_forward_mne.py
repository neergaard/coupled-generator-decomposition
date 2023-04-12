import sys

from projects.anateeg.utils import parse_args

from projects.facerecognition import mne_tools

# sbatch segfaults when mne tries to execute some parallel code
import numba

numba.set_num_threads(1)

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    mne_tools.make_bem_surfaces(subject_id)
    mne_tools.decouple_inner_outer_skull(subject_id)
    mne_tools.make_forward_solution(subject_id)
