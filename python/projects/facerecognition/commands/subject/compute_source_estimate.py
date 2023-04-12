import sys

from projects.anateeg.utils import parse_args

from projects.facerecognition import inverse


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    inverse.compute_source_estimate(subject_id)
