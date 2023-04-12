import sys

from projects.mnieeg import forward
from projects.mnieeg.utils import parse_args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    forward.prepare_for_inverse(subject_id)
