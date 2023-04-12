import sys

from projects.anateeg.utils import parse_args

from projects.facerecognition import forward


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    forward.prepare(subject_id)
