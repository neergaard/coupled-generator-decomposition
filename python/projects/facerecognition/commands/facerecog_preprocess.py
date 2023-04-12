import argparse
import sys

from projects.facerecognition import pipeline

def parse_args(argv):

    program = dict(
        prog = 'facerecog_preprocess',
        description = 'Preprocess EEG and/or MEG data.'
    )
    arg_subject_id = dict(
        help = 'Subject id. Zero will be prepended if necessary (e.g., for sub-01 simply pass 1).',
        type = int
    )
    arg_bids_root = dict(
        help = 'Directory containing the data organized according to the BIDS standard.'
    )
    arg_analysis_root = dict(
        help = 'Directory where analysis results will be written.'
    )

    parser = argparse.ArgumentParser(**program)
    parser.add_argument('subject-id', **arg_subject_id)
    parser.add_argument('bids-root', **arg_bids_root)
    parser.add_argument('analysis-root', **arg_analysis_root)

    args = parser.parse_args(argv)

    return args

def facerecog_preprocess(argv):
    args = parse_args(argv[1:])
    subject_id = getattr(args, 'subject-id')
    bids_root = getattr(args, 'bids-root')
    analysis_root = getattr(args, 'analysis-root')

    pipeline.preprocess(subject_id, bids_root, analysis_root)

if __name__ == '__main__':
    facerecog_preprocess(sys.argv)