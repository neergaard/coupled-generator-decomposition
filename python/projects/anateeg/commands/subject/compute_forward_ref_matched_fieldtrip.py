import sys


from projects.anateeg.utils import parse_args
from projects.anateeg import simnibs_tools


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")

    simnibs_tools.compute_forward_ref_matched_fieldtrip(subject_id)
    simnibs_tools.prepare_for_inverse_ref_matched_fieldtrip(subject_id)
