import argparse
import sys

import h5py
import numpy as np
from pathlib import Path

sys.addpath(r'C:\Users\jdue\Google Drive\wakeman2015_bids_scripts')
from projects import simnibs_eeg

def parse_args(argv):

    program= dict(
        description = "Export leadfield for EEG data analysis.",
        epilog = "Please use %(prog)s -h for help."
    )
    output_format = dict(
        metavar = "Output_format",
        help = "Format in which to export the leadfield."
    )
    mne = dict(
        help = """Export leadfield to MNE format. Output is a -fwd.fif file
                  containing an instance of Forward."""
    )
    fieldtrip = dict(
        help = """Export leadfield to Fieldtrip format. Output is a .mat file
                  containing a struct called fwd and, if the leadfield was
                  generated using 'interpolation', normals of the source
                  positions along with a description of which hemisphere each
                  position belongs to."""
    )
    leadfield = dict(
        help = "Filename of the leadfield (i.e., the .hdf5 file)."
    )

    # positional arguments for MNE
    info = dict(
        help = """Filename of a Raw, Epochs, Evoked or Info file with
                  measurement info. Used to populate the forward object with
                  measurement related information. Read using
                  mne.io.read_info."""
    )
    trans = dict(
        help = """Filename of a Transform file mapping between MRI and head
                  coordinates. This assumes that the forward solution generated
                  in SimNIBS is in MRI coordinates. Forward solutions in MNE
                  are in head coordinates, however, so this is used to
                  transform the solution from SimNIBS to head coordinates.
                  Read using mne.read_trans."""
    )

    parser = argparse.ArgumentParser(**program)
    subparser_format = parser.add_subparsers(dest='output_format', **output_format)

    parser_mne = subparser_format.add_parser("mne", **mne)
    parser_mne.add_argument("leadfield", **leadfield)
    parser_mne.add_argument("info", **info)
    parser_mne.add_argument("trans", **trans)

    parser_ft = subparser_format.add_parser("fieldtrip", **fieldtrip)
    parser_ft.add_argument("leadfield", **leadfield)

    args = parser.parse_args(argv[1:])
    return args

def convert_leadfield(argv):

    args = parse_args(argv)

    output_format = args.output_format
    leadfield = args.leadfield
    if output_format == 'mne':
        info = args.info
        trans = args.trans
    else:
        info = None
        trans = None

    simnibs_eeg.convert_forward_solution(output_format, leadfield, info, trans)

if __name__ == '__main__':
    convert_leadfield(sys.argv)