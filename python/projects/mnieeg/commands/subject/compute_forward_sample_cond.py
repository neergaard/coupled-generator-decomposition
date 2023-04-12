import sys

from projects.mnieeg import forward
from projects.mnieeg.utils import parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    forward.compute_forward_sample_cond(subject_id)
    forward.prepare_for_inverse_sampled_cond(subject_id)
