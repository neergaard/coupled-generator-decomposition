"""
FORWARD MODEL DESCRIPTION
-------------------------
digitized
    Affine transformation (obtained by matching fiducials in subject head space
    and subject MRI space) of digitized electrode positions (using Localite
    TMS-navigator system) defined in head space to MRI space.
custom_nonlin
    Nonlinear warp (obtained from the unified segmentation procedure) of custom
    electrode positions defined in MNI space to subject MRI space.
custom_affine_coreg
    Affine transformation (obtained by coregistration of the subject T1 with
    the template) of custom electrode positions defined in MNI space to subject
    MRI space.
custom_affine_fids

manufacturer_affine_fids
    Affine transformation (obtained by matching fiducials in subject space and
    manufacturer space) of manufacturer electrode positions defined in
    manufacturer space to subject MRI space.
manufacturer_affine_fids_template
    Affine transformation (obtained by matching fiducials in MNI space and
    manufacturer space) of manufacturer electrode positions defined in
    manufacturer space to template (MNI152) space.

NOTE To obtain 3D cartesian coordinates from a list of spherical coordinates
provided by the manufacturer, MNE-Python arranges them on a sphere of a
certain radius which can be specified by the user (the default value is 95 mm).


"""

from pathlib import Path

# 8, 10, 17, 20 have at least one channel placed in the center of the head
# 3, 10 have channels clearly in the wrong place
# 12, 33 (original subject indices!) have a number of channels whose distances
# from the group median are more than three standard deviations away (after
# having excluded the subjects with bad digitalizations).
# 12 : high in frontal and sides; relatively low in the back. Not particularly
# twisted
# 33 : cap too small? frontal ch too high, the cap twists LR going from front
# to back/tilted to the left

# -> it seems that these are positively affected by our "correction" procedure

# EXCLUDED_SUBJECTS = set((3, 8, 10, 17, 20))  # 12, 33


class Paths:
    PROJECT = Path("/mrhome/jesperdn/INN_JESPER/projects/mnieeg")  # symlink
    PHATMAGS = Path("/mnt/projects/PhaTMagS")

    DATA = PROJECT / "data"
    # ANALYSIS = PROJECT / "analysis"
    RESULTS = PROJECT / "results"
    SIMNIBS = PROJECT / "simnibs"
    SIMNIBS_TEMPLATE = PROJECT / "simnibs_template"
    RESOURCES = PROJECT / "resources"


class DataConfig:
    # EXTENSION = "vhdr"
    TASK = "rest"


class PreprocessConfig:
    # Ensure that the channels in Info matches the M10 montage by dropping the
    # EMG channels and channels 70 and 71 since these are special to the "new"
    # TMS cap (based on M10) which appear to be used in PROJECT
    # DROP_CHANNELS = ("EMGleft", "EMGright")  # "70", "71",

    # Channel names in "new" TMS cap are 1, ..., 32, 41, ..., 69[, 70, 71]
    # Channel names in M10 are 1, ..., 61
    # channel_renamer = lambda i: str(int(i) - 8) if int(i) >= 41 else i

    # Candidates for EOG artifact detection (original channel names -
    # channel_renamer will be applied!)
    VEOG_CAND = ("58", "44", "43")
    HEOG_CAND = ("59", "69", "71", "70")

    L_FREQ = 1
    H_FREQ = 100
    NOTCH_FREQS = (50, 100)
    SAMPLING_FREQ = 400


class CovarianceConfig:
    METHOD = "shrunk"
    N_DISCARD = 5  # discard the first and last few "epochs"
    EPOCH_DURATION = 0.2  # in seconds
    # METHODS: dict = {"shrunk": {}}


class InverseConfig:
    FSAVERAGE = 5  # "fsaverage5"
    SMOOTH_SOURCE_ACTIVITY = True
    NAVE = 100
    CONE_ANGLE = 20  # normals in the inverse model are perturbed within a cone

    # "For depth weighting, 0.8 is generally good for MEG, and between 2 and 5
    # is good for EEG"
    # https://doi.org/10.1016/j.neuroimage.2005.11.054

    orientation_prior = None  # 'fixed' or None
    # DEPTH_WEIGHTING = 1

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
    METHOD_CTF = {"MNE", "Dipole", "MUSIC", "LCMV"}
    SNR = (2, 4, 8)


class ResolutionConfig:
    FUNCTIONS = ("psf", "ctf")
    # localization and extent is returned in cm!
    METRICS = ("peak_err", "sd_ext")  # "cog_err", "peak_amp")


class ForwardConfig:
    # LANDMARKS = "Fiducials"
    MONTAGE_MNE = "easycap_BC_TMS64_X21"
    MONTAGE_SIMNIBS = "easycap_BC_TMS64_X21"
    # Models compared on sensor level
    ALL_MODELS = [
        "digitized",
        "custom_nonlin",
        "custom_affine_mri",  # coreg with MRI
        "custom_affine_lm",  # coreg with fiducials
        "manufacturer_affine_lm",  # coreg with fiducials
        "template_nonlin",
    ]
    # Models compared on inverse level
    MODELS = [
        "digitized",
        "custom_nonlin",
        "manufacturer_affine_lm",
        "template_nonlin",
    ]
    SOURCE_SPACE = ["reference", "reference", "reference", "template"]

    REFERENCE = MODELS[0]
    SUBSAMPLING = 10000


class StatsConfig:
    # Do F-tests for each inverse solution separately since sLORETA (having
    # zero localization error by design) would ensure an effect of forward
    # model
    EFFECTS_OF_INTEREST = ("forward", "snr", "forward:snr")
    TFCE_THRESHOLD = dict(start=0, step=20)


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
    fwd_model_name = dict(
        digitized="Digitized",
        custom_nonlin="Custom-Template",
        manufacturer_affine_lm="Man-Template",
        template_nonlin="MNI-Digitized",
    )
    # used for channel plots...
    ch_model_name = dict(
        digitized="Digitized",
        custom_nonlin="Custom (nonlinear)",
        custom_affine_mri="Custom (affine MRI)",
        custom_affine_lm="Custom (affine LM)",
        manufacturer_affine_lm="Manufacturer (affine LM)",
        template_nonlin="MNI Digitized (nonlinear)",
    )
    names = dict(
        res_fun=dict(psf="PSF", ctf="CTF"),
        res_met=dict(peak_err="PLE", sd_ext="SD"),
        stat=dict(mean="Mean", median="Median", std="Standard Deviation"),
    )


class Config:
    path = Paths
    data = DataConfig
    preprocess = PreprocessConfig
    covariance = CovarianceConfig
    forward = ForwardConfig
    inverse = InverseConfig
    resolution = ResolutionConfig
    stats = StatsConfig
    plot = PlotConfig


# class ShrinkageCovarianceConfig:
#     # To control amount of regularization, use
#     REG = 0.1
#     METHOD = 'shrinkage'
#     METHOD_PARAMS = dict(shrinkage=REG)

# class Config(Config):
#     COVARIANCE = ShrinkageCovarianceConfig
