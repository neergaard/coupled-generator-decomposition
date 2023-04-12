import argparse
import pickle

import mne_bids
import numpy as np
import pandas as pd

from projects.base.io.organize import FileFormatter, SimnibsOrganizer
from projects.facerecognition_dtu.config import Config


def parse_args_with_mode(argv, prog=None, description=None):
    """Parse arguments to a function that takes subject ID and a `mode`
    argument."""
    program = dict(prog=prog, description=description,)
    arg_subject_id = dict(
        help="Subject id. Zero will be prepended if necessary "
        + "(e.g., for sub-01 simply pass 1)."
    )
    arg_mode = dict(help="Mode in which to execute function",)
    parser = argparse.ArgumentParser(**program)
    parser.add_argument("subject-id", **arg_subject_id)
    parser.add_argument("mode", **arg_mode)
    return parser.parse_args(argv[1:])


def init_fileformatter(root, kwargs):
    path_attrs = ("root", "subject", "session", "stage")
    file_attrs = (
        # "subject",
        # "session",
        "run",
        "task",
        "processing",
        "space",
        "condition",
        "contrast",
        "forward",
        "inverse",
        "channel",
        "split",
        "snr",
    )
    attr2abbr = dict(
        subject="sub",
        session="ses",
        run="run",
        task="task",
        processing="proc",
        condition="cond",
        contrast="contr",
        space="space",
        forward="fwd",
        inverse="inv",
        channel="ch",
        split="split",
        snr="snr",
    )
    filenamer = FileFormatter(path_attrs, file_attrs)
    filenamer.set_attr2abbr(**attr2abbr)
    filenamer.update(**kwargs, session="meg", extension="fif", root=root)
    return filenamer


class IO:
    def __init__(self, kwargs):
        self.data = init_fileformatter(Config.path.DATA, kwargs)
        self.simnibs = dict(
            charm=SimnibsOrganizer(Config.path.SIMNIBS_CHARM),
        )
        # BIDS
        self.bids = dict(
            meg=mne_bids.BIDSPath(
                root=Config.path.BIDS / "derivatives" / "meg_derivatives",
                datatype="meg",
                session="meg",
                task="facerecognition",
                processing="sss",
                extension=".fif",
            ),
            mri=mne_bids.BIDSPath(
                root=Config.path.BIDS,
                datatype="anat",
                session="mri",
                acquisition="mprage",
                suffix="T1w",
            ),
        )

    # @staticmethod
    # def read_include():
    #     with open(Config.path.DATA / "include.json", "r") as f:
    #         include = json.load(f)
    #     return include


class SubjectIO(IO):
    def __init__(self, subject_id):  # , read_session=True, raise_on_exclude=True):

        try:
            self.subject = "{:02d}".format(int(subject_id))
        except ValueError:
            self.subject = subject_id
        kwargs = dict(subject=self.subject, task="facerecognition")
        super().__init__(kwargs)

        for k in self.simnibs:
            self.simnibs[k].update(subject=self.subject)
        for k in self.bids:
            self.bids[k].update(subject=self.subject)

        # if read_session:
        #     try:
        #         self.session = self.read_include()[self.subject]
        #         self.included = True
        #     except KeyError:
        #         self.session = None
        #         self.included = False

        #     if raise_on_exclude:
        #         assert self.included, f"Subject {self.subject} not included."

        #     self.data.update(session=self.session)


class GroupIO(IO):
    def __init__(self):

        kwargs = dict(task="facerecognition")
        super().__init__(kwargs)

        # self.subjects = self.read_include()
        self.subjects = sorted(
            [p.stem.lstrip("sub-") for p in self.data.path.root.glob("sub*")]
        )

    # def write_include(self):
    #     with open(Config.path.DATA / "include.json", "w") as f:
    #         json.dump(self.subjects, f, indent=4)


def get_func_sessions(io_obj):
    return tuple(
        s.stem.lstrip("ses-")
        for s in io_obj.path.get(stage=None).glob("ses*")
        if s.stem != "ses-mri"
    )


def write_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, -1)


def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def fsaverage_as_index(n):
    """
    fsaverage       0   1   2   3   4   5   6   7
    subdivision     1   2   4   8   16  32  64  128
    """
    hemis = ("lh", "rh")
    subdiv = 2 ** n
    n_verts = 2 + 10 * subdiv ** 2
    vertnos = np.arange(n_verts)
    return pd.MultiIndex.from_product([hemis, vertnos], names=["Hemi", "Source"])
