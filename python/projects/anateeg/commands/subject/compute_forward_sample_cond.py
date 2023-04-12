import sys


from projects.anateeg.utils import parse_args
from projects.anateeg import simnibs_tools


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")

    simnibs_tools.compute_forward_sample_cond(subject_id)
    simnibs_tools.prepare_for_inverse_sampled_cond(subject_id)
