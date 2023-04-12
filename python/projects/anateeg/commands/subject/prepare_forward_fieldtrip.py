import sys

from projects.anateeg import fieldtrip_tools
from projects.anateeg.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    fieldtrip_tools.make_headmodel(subject_id)
    fieldtrip_tools.make_forward_solution(subject_id)
