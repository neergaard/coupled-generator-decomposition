import sys

from projects.mnieeg import preprocess
from projects.mnieeg.utils import parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    preprocess.preprocess(subject_id)
