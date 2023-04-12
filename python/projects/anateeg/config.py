from pathlib import Path


class PathConfig:
    # Original data
    SKULL_RECO = Path("/mnt/projects/skull_reco")
    CLEAN_SEGS = Path(
        "/mnt/projects/INN/oula/backup/Runs_for_whole_head_segmentation_article"
    )
    DUNEURO_LIBS = Path("/mrhome/jesperdn/Documents/duneuro_libs")
    MATLAB_FUNCTIONS = Path("/home/jesperdn/project_tools/matlab/projects/anateeg")
    PROJECT = Path("/mrhome/jesperdn/INN_JESPER/projects/anateeg")  # symlink

    DATA = PROJECT / "data"
    RESOURCES = PROJECT / "resources"
    SIMNIBS = PROJECT / "simnibs"
    SIMNIBS_REFERENCE = PROJECT / "simnibs_reference"
    SIMNIBS_TEMPLATE = PROJECT / "simnibs_template"
    FREESURFER = PROJECT / "freesurfer"
    MNE = PROJECT / "mne"
    FIELDTRIP = PROJECT / "fieldtrip"

    RESULTS = PROJECT / "results"


class DataConfig:
    TASK = "rest"


class ForwardConfig:

    MONTAGE = "easycap_BC_TMS64_X21"
    SUBSAMPLING = 10000

    MODELS = ["reference", "charm", "fieldtrip", "mne", "template"]
    MODEL_TYPE = ["simnibs", "simnibs", "fieldtrip", "mne", "simnibs"]
    SOURCE_SPACE = ["reference", "charm", "reference", "reference", "template"]
    REFERENCE = "reference"


class InverseConfig:
    FSAVERAGE = 5
    SMOOTH_SOURCE_ACTIVITY = True
    NAVE = 100
    CONE_ANGLE = 20  # normals in the inverse model are perturbed within a cone

    orientation_prior = None  # 'fixed' or None
    # DEPTH_WEIGHTING = 1

    # "For depth weighting, 0.8 is generally good for MEG, and between 2 and 5
    # is good for EEG"
    # https://doi.org/10.1016/j.neuroimage.2005.11.054

    METHODS = ("MNE", "dSPM", "sLORETA", "Dipole", "MUSIC", "LCMV")
    METHOD_TYPE = dict(
        MNE="mne",
        dSPM="mne",
        sLORETA="mne",
        MUSIC="music",
        Dipole="dipole",
        LCMV="beamformer",
    )
    # Compute cross-talk functions only for these methods
    # (CTF(mne) = CTF(dspm) = CTF(sloreta))
    METHOD_CTF = ("MNE", "Dipole", "MUSIC", "LCMV")
    SNR = (2, 4, 8)


class ResolutionConfig:
    # localization and extent is returned in cm!
    FUNCTIONS = ("psf", "ctf")
    METRICS = ("peak_err", "sd_ext")  # , "peak_amp")


class PlotConfig:

    density_point_range = [0, 12]

    limits = dict(
        forward=dict(mean=dict(RDM=[0, 0.6], lnMAG=[-0.5, 0.5])),
        inverse=dict(
            density=dict(peak_err=[0, 8], sd_ext=[0, 8], probdens=[0, 1]),
            surf=dict(
                mean=dict(peak_err=[0, 5], sd_ext=[0, 10]),
                std=dict(peak_err=[0, 5], sd_ext=[0, 10]),
            ),
        ),
    )

    names = dict(
        res_fun=dict(psf="PSF", ctf="CTF"),
        res_met=dict(peak_err="PLE", sd_ext="SD"),
        stat=dict(mean="Mean", median="Median", std="Standard Deviation"),
    )

    fwd_model_name = dict(
        reference="Manual",
        charm="SimNIBS-CHARM",
        fieldtrip="FT-SPM",
        mne="MNE-FS",
        template="MNI-Template",  # "SimNIBS-TEMP",
    )


class Config:
    path = PathConfig
    data = DataConfig
    forward = ForwardConfig
    inverse = InverseConfig
    resolution = ResolutionConfig
    plot = PlotConfig
