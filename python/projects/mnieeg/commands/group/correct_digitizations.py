import logging

import mne
import nibabel as nib
import numpy as np
import pyvista as pv

from projects.mnieeg import utils

from simnibs.simulation import eeg


def correct_digitizations(tol=50, n_correct=1):
    """

    pre annot  : 0 = fine, 1 = futher away than `tol`
    post annot : 0 = fine, -1 = not fixed, 1 = fixed
    """

    assert n_correct == 1, "only `n_correct` = 1 is implemented as of now"

    io = utils.GroupIO()
    io.data.update(suffix="eeg")

    # Setup logging
    log_file = io.data.path.root / "correct_digitizations.log"
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = "%(asctime)s [%(name)s:%(funcName)s] [%(levelname)-7.7s] %(message)s"
    logging.basicConfig(
        format=fmt,
        datefmt=datefmt,
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    # sessions = mne_bids.get_entity_vals(io.bids.root, "session", ignore_sessions="mri")

    # Get all digitizations in MNI coordinates
    mne.set_log_level("warning")
    sub, ses, data = [], [], []
    logger.info("Loading data and transforming to MNI...")
    for subject in io.subjects:
        logger.info(f"Subject {subject}")
        io.data.update(subject=subject)
        io.simnibs.update(subject=subject)

        for session in utils.get_func_sessions(io.data):

            raw = mne.io.read_raw_fif(io.data.get_filename(session=session))
            trans = mne.read_trans(
                io.data.get_filename(session=session, suffix="trans")
            )
            itrans = mne.transforms.invert_transform(trans)

            ch_pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
            ch_pos = 1e3 * mne.transforms.apply_trans(itrans, ch_pos)

            # to MNI
            m2m = io.simnibs.get_path("m2m")
            field = nib.load(m2m / "toMNI" / "Conform2MNI_nonl.nii.gz")
            montage = eeg.Montage(None, raw.info["ch_names"], ch_pos, "Electrode")
            montage.apply_deform(field)

            sub.append(subject)
            ses.append(session)
            data.append(montage.ch_pos)

    ss = np.column_stack((sub, ses))
    data = np.array(data)

    median = np.median(data, 0)
    ss_idx, ch_idx = np.where(np.linalg.norm(data - median, axis=-1) > tol)
    flat_idx = np.ravel_multi_index((ss_idx, ch_idx), data.shape[:-1])

    montage = eeg.Montage(None, montage.ch_names, median, "Electrode")

    uniq, idx, counts = np.unique(ss_idx, return_index=True, return_counts=True, axis=0)
    correct = counts <= n_correct

    pre = pv.PolyData(data.reshape(-1, 3))
    pre["annot"] = np.zeros(len(pre.points))
    pre["annot"][flat_idx] = 1
    pre.save(io.data.path.root / "dig_mni_pre_fix.vtk")

    post = pre
    post["annot"][flat_idx] = -1

    corrected = set()
    not_corrected = set()
    for u, i, c, co in zip(uniq, idx, counts, correct):
        subject, session = ss[u]
        if not co:
            logger.info(f"Not correcting sub-{subject} ses-{session} ({c} problems)")
            not_corrected.update([(subject, session)])
            continue
        logger.info(f"Correcting sub-{subject} ses-{session}")
        corrected.update([(subject, session)])

        io.data.update(subject=subject, session=session)
        fname_raw = io.data.get_filename()
        raw = mne.io.read_raw_fif(fname_raw, preload=True)  # enable overwrite
        trans = mne.read_trans(io.data.get_filename(suffix="trans"))

        # to subject space
        io.simnibs.update(subject=subject)
        m2m = io.simnibs.get_path("m2m")
        field = nib.load(m2m / "toMNI" / "MNI2Conform_nonl.nii.gz")
        chi = ch_idx[i]
        tmp = eeg.Montage(
            None, [montage.ch_names[chi]], [montage.ch_pos[chi]], "Electrode"
        )
        tmp.apply_deform(field)

        # replace
        chs_idx = [
            chs_idx
            for chs_idx, ch in enumerate(raw.info["chs"])
            if ch["ch_name"] == tmp.ch_names[0]
        ][0]
        dig_idx = [
            dig_idx
            for dig_idx, dig in enumerate(raw.info["dig"])
            if dig["ident"] == int(tmp.ch_names[0])
        ][0]
        pos = mne.transforms.apply_trans(trans, 1e-3 * tmp.ch_pos)
        logger.info(
            f'Replacing {np.round(1e3*raw.info["chs"][chs_idx]["loc"][:3],2)} with {np.round(1e3*pos,2)}'
        )
        raw.info["chs"][chs_idx]["loc"][:3] = pos
        raw.info["dig"][dig_idx]["r"] = pos

        raw.save(fname_raw, overwrite=True)

        post.points[flat_idx[i]] = montage.ch_pos[chi]
        post["annot"][flat_idx[i]] = 1

    post.save(io.data.path.root / "dig_mni_post_fix.vtk")

    # Priority: good session > corrected session, session 01 > 02 (arbitrary)
    # If no good or corrected session exist for a subject then exclude
    subses = set(zip(sub, ses))
    include = set()
    exclude = set()

    # Check if we need to exclude any subjects
    for sub, ses in not_corrected:
        if ses == "01":
            x = (sub, "02")
            if x not in subses or x in not_corrected:
                logger.info(
                    f"No good or corrected sessions for subject {sub}, excluding"
                )
                exclude.update([sub])
        # if ses == '02' there is nothing to do as this is not used by default
        # and if ses == '01' was also in not_corrected then the subjects has
        # already been excluded...

    for sub, ses in corrected:
        # Use session 02 if it is good
        if ses == "01":
            x = (sub, "02")
            if x in subses and x not in corrected and x not in not_corrected:
                logger.info(
                    f"sub-{sub}: ses-01 is corrected, ses-02 is good; using ses-02"
                )
                include.update([x])
        # Only use a corrected session 02 if 01 was not corrected
        elif ses == "02":
            x = (sub, "01")
            if x in not_corrected:
                logger.info(
                    f"sub-{sub}: ses-01 is not corrected, ses-02 is corrected; using ses-02"
                )
                include.update([(sub, ses)])

    logger.info("Modifications to include.json")
    logger.info("Updating sessions")
    logger.info(include)
    logger.info("Excluding")
    logger.info(exclude)

    io.subjects.update({sub: ses for sub, ses in include})
    for sub in exclude:
        del io.subjects[sub]
    io.write_include()

    # logger.info("include.json does now look like this")
    # logger.info(io.subjects)


if __name__ == "__main__":
    correct_digitizations()
