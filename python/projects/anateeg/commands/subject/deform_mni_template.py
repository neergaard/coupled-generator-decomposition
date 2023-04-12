import sys

from projects.anateeg import simnibs_tools
from projects.anateeg.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    simnibs_tools.deform_template_to_subject(subject_id)
