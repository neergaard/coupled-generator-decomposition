import argparse
import json

from projects.base.io.organize import FileFormatter, SimnibsOrganizer

from projects.mnieeg.config import Config


def parse_args(argv, prog=None, description=None):
    """Parse arguments to a function that only takes subject ID as argument."""
    program = dict(prog=prog, description=description,)
    arg_subject_id = dict(
        help="Subject id. Zero will be prepended if necessary "
        + "(e.g., for sub-01 simply pass 1)."
    )
    parser = argparse.ArgumentParser(**program)
    parser.add_argument("subject-id", **arg_subject_id)
    return parser.parse_args(argv[1:])


def init_fileformatter(root, kwargs):
    path_attrs = ("root", "subject", "session", "stage")
    file_attrs = (
        # "subject",
        # "session",
        "task",
        "processing",
        "space",
        "forward",
        "inverse",
        "snr",
    )
    attr2abbr = dict(
        subject="sub",
        session="ses",
        task="task",
        processing="proc",
        space="space",
        forward="fwd",
        inverse="inv",
        snr="snr",
    )
    filenamer = FileFormatter(path_attrs, file_attrs)
    filenamer.set_attr2abbr(**attr2abbr)
    filenamer.update(**kwargs, extension="fif", root=root)
    return filenamer


class IO:
    def __init__(self, kwargs):
        self.data = init_fileformatter(Config.path.DATA, kwargs)
        self.simnibs = SimnibsOrganizer(Config.path.SIMNIBS)
        self.simnibs_template = SimnibsOrganizer(Config.path.SIMNIBS_TEMPLATE)

    @staticmethod
    def read_include():
        with open(Config.path.DATA / "include.json", "r") as f:
            include = json.load(f)
        return include


class SubjectIO(IO):
    def __init__(self, subject_id, read_session=True, raise_on_exclude=True):

        try:
            self.subject = "{:02d}".format(int(subject_id))
        except ValueError:
            self.subject = subject_id
        kwargs = dict(subject=self.subject, task=Config.data.TASK)
        super().__init__(kwargs)

        self.simnibs.update(subject=self.subject)
        self.simnibs_template.update(subject=self.subject)

        if read_session:
            try:
                self.session = self.read_include()[self.subject]
                self.included = True
            except KeyError:
                self.session = None
                self.included = False

            if raise_on_exclude:
                assert self.included, f"Subject {self.subject} not included."

            self.data.update(session=self.session)


class GroupIO(IO):
    def __init__(self):

        kwargs = dict(task=Config.data.TASK)
        super().__init__(kwargs)

        self.subjects = self.read_include()

    def write_include(self):
        with open(Config.path.DATA / "include.json", "w") as f:
            json.dump(self.subjects, f, indent=4)


def get_func_sessions(io_obj):
    return tuple(
        s.stem.lstrip("ses-")
        for s in io_obj.path.get(stage=None).glob("ses*")
        if s.stem != "ses-mri"
    )
