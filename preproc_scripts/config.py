from pathlib import Path


class PathConfig:
    BIDS = Path("/mrhome/jesperdn/INN_JESPER/DATASETS/nobackup/openneuro/ds000117")
    PROJECT = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition_dtu")
    PROJECT_ORIG = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition")
    # PROJECT_SCRATCH = Path("/mnt/scratch/INN/jesperdn/facerecognition_dtu")


    # From anateeg project
    MATLAB_FUNCTIONS = Path("/home/jesperdn/project_tools/matlab/projects/anateeg")
    FMRI_FUNCTIONS = Path(
        "/home/jesperdn/project_tools/matlab/projects/facerecognition_dtu"
    )

    DATA = PROJECT / "data"
    RESOURCES = PROJECT / "resources"
    RESULTS = PROJECT / "results"
    SIMNIBS_CHARM = PROJECT_ORIG / "simnibs_charm"
    FREESURFER = PROJECT_ORIG / "freesurfer"
    MNE = PROJECT / "mne"


class PreprocessConfig:
    MODALITIES = ["eeg", "meg"]

    # RAW
    RENAME_CHANNELS = dict(EEG061="HEOG", EEG062="VEOG", EEG063="ECG",)
    CHANNEL_TYPES = dict(HEOG="eog", VEOG="eog", ECG="ecg")
    # EEG064 is "floating noise"
    DROP_CHANNELS = ["EEG064"]

    L_FREQ = 1
    H_FREQ = 40
    NOTCH_FREQS = None

    # EPOCHS
    TMIN = -0.1
    TMAX = 0.8
    STIMULUS_DELAY = 0
    S_FREQ = 200

    COVARIANCE = dict(
        noise=dict(method="shrunk", tmin=None, tmax=0),
        data=dict(method="shrunk", tmin=0, tmax=None),
    )
    COVARIANCE_XDAWN = dict(
        noise=dict(method="diagonal_fixed", tmin=None, tmax=0),
        data=dict(method="diagonal_fixed", tmin=0, tmax=None),
    )

    # USE_FOR_CONTRAST = dict(processing="pad", space="signal")
    USE_FOR_CONTRAST = dict(processing="pa")


class AutorejectConfig:
    CONSENSUS = None
    N_INTERPOLATE = None
    N_JOBS = 4


class XdawnConfig:
    N_SPLITS = 2
    N_ROUNDS = 3


class Contrast:
    def __init__(self, name, conditions, weights):
        self.name = name
        self.conditions = conditions
        self.weights = weights


class ConditionsConfig:
    CONDITIONS = ["face", "scrambled"]
    CONTRASTS = [
        Contrast("faces vs. scrambled", ["face", "scrambled"], [1, -1])  # ,
        # Contrast('famous vs. unfamiliar', ['face/famous', 'face/unfamiliar'], [1, -1])
    ]


class ForwardConfig:
    SUBSAMPLING = 10000

    MODELS = ["charm", "fieldtrip", "mne", "template"]
    MODEL_TYPE = ["simnibs", "fieldtrip", "mne", "simnibs"]
    SOURCE_SPACE = ["charm", "charm", "charm", "template"]
    REFERENCE = "charm"  # defines the source space used with fieldtrip and mne


class InverseConfig:
    FSAVERAGE = 5  # source space for group analysis/comparison
    # FIXED = True
    METHODS = ("MNE", "dSPM", "sLORETA", "LCMV", "MUSIC")  # 'Dipole',
    METHOD_TYPE = ("mne", "mne", "mne", "beamformer", "music")  # 'dipole',
    # DEPTH = 3

    TIME_WINDOW = [0.1, 0.2]
    TIME_DELTA = 0.01
    # SOLVERS = [
    #     dict(method="dSPM", loose=None),
    #     dict(method="sLORETA", loose=None)  # ,
    #     # dict(
    #     #     method = 'lcmv',
    #     #     reg = 0.5,
    #     #     pick_ori = None
    #     # )
    # ]


class Config:
    path = PathConfig
    preprocess = PreprocessConfig
    forward = ForwardConfig
    inverse = InverseConfig
