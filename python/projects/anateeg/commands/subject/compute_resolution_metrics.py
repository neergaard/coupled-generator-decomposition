import sys


from projects.anateeg.inverse import compute_resolution_metrics
from projects.anateeg.utils import parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")

    compute_resolution_metrics(subject_id)
