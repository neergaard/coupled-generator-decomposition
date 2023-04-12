import sys

from projects.anateeg.utils import parse_args
from projects.facerecognition import simnibs_tools

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    simnibs_tools.deform_template_to_subject(subject_id)
