import sys

from projects.anateeg.utils import parse_args

from projects.facerecognition import preprocess


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    preprocess.preprocess(subject_id)
