import sys

from projects.anateeg import utils

from projects.facerecognition import simnibs_tools

if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    simnibs_tools.make_forward_solution(subject_id)
