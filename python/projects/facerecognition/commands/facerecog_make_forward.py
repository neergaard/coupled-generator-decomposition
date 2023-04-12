import argparse
import sys

from projects.facerecognition import pipeline

def parse_args(argv):

    program = dict(
        prog = 'facerecog_make_forward',
        description = 'Make forward models.'
    )

    args = {}
    args['subject-id'] = dict(
        help = 'Subject id. Zero will be prepended if necessary (e.g., for sub-01 simply pass 1).',
        type = int
    )
    args['bids-root'] = dict(
        help = 'Directory containing the data organized according to the BIDS standard.'
    )
    args['analysis-root'] = dict(
        help = 'Directory where analysis results will be written.'
    )

    action = {
        "metavar":"action",
        "help"   :"Action to perform."
    }

    actions = {}
    actions['prepare'] = {
        "help"   :"Prepare for forward modeling.",
        'description': """This will coregister sensors and structural MRI,
                          create the source space in MNE format, and an EEG
                          layout in MRI space (for use with FieldTrip and
                          SimNIBS).
                       """
    }
    actions['mne'] = {
        "help"   :"Make forward model using MNE.",
        'description': "Make forward model using MNE."
    }
    actions['simbio'] = {
        "help"   :"Make forward model using SimBio.",
        'description': "Make forward model using SimBio."
    }
    actions['simnibs'] = {
        "help"   :"Make forward model using SimNIBS.",
        'description': "Make forward model using SimNIBS."
    }

    # main parser
    parser = argparse.ArgumentParser(**program)
    # 'action' subparser
    subparsers = parser.add_subparsers(dest='action', **action)
    # arguments common to all subparsers
    for arg in args:
        parser.add_argument(arg, **args[arg])
    # the subparsers
    for a in actions:
        subparsers.add_parser(a, **actions[a])

    args = parser.parse_args(argv)

    return args

def facerecog_make_forward(argv):
    args = parse_args(argv[1:])

    action = args.action
    subject_id = getattr(args, 'subject-id')
    bids_root = getattr(args, 'bids-root')
    analysis_root = getattr(args, 'analysis-root')

    if action == 'prepare':
        pipeline.prepare_for_forward(subject_id, bids_root, analysis_root)
    elif action == 'mne':
        pipeline.make_forward_mne(subject_id, bids_root, analysis_root)
    elif action == 'simbio':
        pipeline.make_forward_simbio(subject_id, bids_root, analysis_root)
    elif action == 'simnibs':
        pipeline.make_forward_simnibs(subject_id, bids_root, analysis_root)

if __name__ == '__main__':
    facerecog_make_forward(sys.argv)