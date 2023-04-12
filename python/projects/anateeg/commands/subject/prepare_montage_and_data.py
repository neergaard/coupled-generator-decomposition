import sys

from projects.anateeg import mne_tools, simnibs_tools
from projects.anateeg.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    simnibs_tools.make_montage(subject_id)
    mne_tools.make_data(subject_id)
