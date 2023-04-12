import logging
import mne
import numpy as np
import json
import re
from xml.dom import minidom

import nibabel as nib

from projects.mnieeg import utils
from projects.mnieeg.config import Config
from projects.mnieeg.phatmags.bads import BadsPhatmags, BadsSubject
from projects.mnieeg.phatmags.mrifiducials import MRIFiducials

"""
NOTES
=========
T2w

It is necessary to modify mne_bids.config.ALLOWED_FILENAME_SUFFIX to include
T2w!

Transformation

Electrodes are already in orig_subject MRI RAS so we just use an identity
transformation
NOTE: orig_subject RAS coordinate frame is actually 'ras' in MNE, however, we
have to use 'mri' (which really refers to FreeSurfer's RAS coordinate system)
in order for MNE to accept it! Use

    trans = mne.Transform('mri', 'head') # identity


"""


def xml_markers_to_montage(filename):
    xml = minidom.parse(str(filename))
    elecs = xml.getElementsByTagName("EEGMarkerList")[0]

    elec_name = elecs.getElementsByTagName("Marker")
    elec_pos = elecs.getElementsByTagName("ColVec3D")

    channels = {}
    for en, ep in zip(elec_name, elec_pos):
        name = en.attributes["description"].value
        pos = np.array(ep.attributes.items())[:, 1].astype(float)
        pos *= 1e-3  # convert to m
        if name == "Cz":
            name = "1"
        channels[name] = pos

    return mne.channels.make_dig_montage(channels, coord_frame="mri")


def prepare_phatmags():

    if not Config.path.DATA.exists():
        Config.path.DATA.mkdir(parents=True)

    use_landmarks = ("nasion", "lpa", "rpa")

    log_file = Config.path.DATA / "prepare_phatmags.log"

    # Sessions to consider
    sessions = ("Alpha", "Beta", "MonoMu", "MonoMu_Radial", "PowerInformed")

    # If multiple MRIs satisfy the condition, you this one
    mri_chooser = dict(X60110="X60110_WIP_T2w_085iso_V2_501.nii")

    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = "%(asctime)s [%(levelname)-7.7s] %(message)s"
    logging.basicConfig(
        filename=log_file, format=fmt, datefmt=datefmt, level=logging.INFO
    )
    logger = logging.getLogger("prepare_phatmags")

    # Matching the (correct) T1w image
    #
    #   'X\d{5,5}_WIP_T1w_085iso.*_(?!BAD$).[.]nii$'
    #
    # This pattern will match
    # (1) X
    # (2) exactly 5 digits
    # (3) '_WIP_T1w_085iso'
    # (4) anything (.*)
    # (5) _
    # (6) not 'BAD' but anything else (.*)
    # (7) ends with '.nii$'

    patterns = dict(
        # baseline = r'Baseline01[.](eeg|vhdr|vmrk)',
        # sessions = r'Alpha|Beta|MonoMu|MonoMu_Radial|PowerInformed',
        # eegmarkers = r'EEGMarkers.*[.].xml',
        # orig_subject = r'X\d{5,5}',
        T1w=r"[xX]\d{5,5}_WIP_T1w_085iso.*(?<!BAD)[.]nii$",
        T2w=r"[xX]\d{5,5}_WIP_T2w_085iso.*(?<!BAD)[.]nii$",
    )
    compiled_pattern = {p: re.compile(v) for p, v in patterns.items()}

    filters = dict(
        not_none=lambda x: x is not None, has_open=lambda x: "open" in x.stem.lower()
    )

    log = {}

    subject_id_num = 1
    for sub_dir in Config.path.PHATMAGS.glob("X" + 5 * "[0-9]"):
        orig_subject = sub_dir.stem
        io = utils.SubjectIO(subject_id_num, read_session=False)
        # subject_id = "{:02d}".format(subject_id_num)

        logger.info(orig_subject)

        try:
            this_bads = getattr(BadsPhatmags, orig_subject)
        except AttributeError:
            # The orig_subject is not in BadsPhatmags which means that he will
            # probably be excluded for some other reason
            this_bads = BadsSubject()

        try:
            # Convert to m as mne_bids converts to mm...
            landmarks = getattr(MRIFiducials, orig_subject)
            # landmarks = {k: [1e-3 * i for i in v] for k, v in landmarks.items()}
            landmarks = {lm: [1e-3 * i for i in landmarks[lm]] for lm in use_landmarks}
            landmarks = mne.channels.make_dig_montage(**landmarks, coord_frame="mri")
        except AttributeError:
            # The subject is not in MRIFiducials which means that he will
            # probably be excluded for some reason. We will get an error later
            # if not
            landmarks = mne.channels.make_dig_montage(coord_frame="mri")

        if this_bads.exclude:
            logger.warning(f"Excluding {orig_subject} based on 'phatmags_bads'")
            logger.warning(f"Reason for exclusion: {this_bads.exclude_reason}\n")
            continue

        log[io.subject] = dict(original_subject=orig_subject)

        # Get structural data (T1, T2) if it exists
        logger.info("Structural MRIs")

        SP = sub_dir / "SP"
        files = list(SP.glob("*"))

        match = {}
        anat = ("T1w", "T2w")
        mri = {}
        for img in anat:
            match[img] = list(
                filter(
                    filters["not_none"],
                    [compiled_pattern[img].match(f.stem + f.suffix) for f in files],
                )
            )
            if match[img]:
                if len(match[img]) == 1:
                    mri[img] = SP / match[img][0].string
                if len(match[img]) > 1:
                    mri[img] = SP / mri_chooser[orig_subject]
                    logger.warning(
                        f"More than one {img} detected. Using {mri_chooser[orig_subject]}"
                    )

        if "T1w" not in mri or "T2w" not in mri:
            logger.warning(
                f"Missing MR images, skipping subject. Found only {tuple(mri.keys())}\n"
            )
            continue

        session_id_num = 1
        for session in sessions:
            session_id = "{:02d}".format(session_id_num)

            ses_dir = sub_dir / session
            if not ses_dir.exists():
                continue

            logger.info(f"Session {session}")

            # Electrode positions
            markers = list(ses_dir.glob("EEGMarkers*.xml"))
            if not markers:
                logger.warning("EEG markers not found, skipping")
                continue
            else:
                assert len(markers) == 1
                markers_fname = ses_dir / markers[0]

            # Rest (open eyes) condition
            header_fnames = ses_dir.glob("*.vhdr")
            raw_fname = list(filter(filters["has_open"], header_fnames))
            if not raw_fname:
                logger.warning("EEG data (open eyes rest) not found, skipping")
                continue
            else:
                assert len(raw_fname) == 1
                raw_fname = raw_fname[0]

            raw = mne.io.read_raw_brainvision(raw_fname, preload=True)
            # Drop EMG channels
            for ch_name in ("EMGleft", "EMGright"):
                try:
                    raw.drop_channels(ch_name)
                    logger.info(f"Dropped channel {ch_name}")
                except ValueError:
                    logger.warning(f"No channel named {ch_name}. Nothing dropped.")
            raw.info["line_freq"] = 50

            # set_montage will convert the montage to Neuromag head coordinate
            # system (the `head` coordinate system used by MNE) thus we save
            # this transformation as `mri_head-trans`
            montage = landmarks + xml_markers_to_montage(markers_fname)
            mri_head_t = mne.channels.compute_native_head_t(montage)
            raw.set_montage(montage)

            BadsSession = getattr(this_bads, session)

            # Annotate bad channels
            raw.info["bads"] = [str(ch) for ch in BadsSession]

            # Annotate bad segments (this will overwrite any existing
            # annotations!)
            # raw.set_annotations(None) # delete 'New Segment' annotation
            raw.set_annotations(getattr(this_bads.Annotations, session))

            # Write data and trans
            io.data.update(session=session_id)
            io.data.path.ensure_exists()
            raw.save(io.data.get_filename(suffix="eeg"))
            mne.write_trans(io.data.get_filename(suffix="trans"), mri_head_t)

            log[io.subject][session_id] = dict(
                original_session=session, marker=str(markers_fname), raw=str(raw_fname)
            )
            session_id_num += 1

        if session_id_num == 1:
            logger.warning("No usable sessions, skipping subject\n")
            continue

        # MRI
        log[io.subject]["mri"] = dict(original_session="SP")
        io.data.update(session="mri")
        io.data.path.ensure_exists()
        for k, v in mri.items():
            img = nib.load(v)
            img.to_filename(
                io.data.get_filename(task=None, suffix=k, extension="nii.gz")
            )
            log[io.subject]["mri"][k] = str(mri[k])

        logger.info(f"Subject included as sub-{io.subject}\n")
        subject_id_num += 1

    with open(Config.path.DATA / "prepare_phatmags.json", "w") as f:
        json.dump(log, f, indent=4)

    # Use session 01 for all subjects by default
    include = {sub: "01" for sub in log}
    with open(Config.path.DATA / "include.json", "w") as f:
        json.dump(include, f, indent=4)


if __name__ == "__main__":
    prepare_phatmags()
