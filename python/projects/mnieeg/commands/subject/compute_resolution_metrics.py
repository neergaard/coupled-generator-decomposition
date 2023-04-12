import sys

from projects.mnieeg import inverse
from projects.mnieeg.utils import parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    inverse.compute_resolution_metrics(subject_id)
