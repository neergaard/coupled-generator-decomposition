import sys

import mne
import mne_bids
import numpy as np

from projects.anateeg.utils import parse_args
from projects.facerecognition import preprocess

from projects.facerecognition_dtu import utils
from projects.facerecognition_dtu.config import Config


def preprocess_raw(raw):

    if hasattr(Config.preprocess, "DROP_CHANNELS"):
        raw.drop_channels(Config.preprocess.DROP_CHANNELS)
    if hasattr(Config.preprocess, "RENAME_CHANNELS"):
        raw.rename_channels(Config.preprocess.RENAME_CHANNELS)
    if hasattr(Config.preprocess, "CHANNEL_TYPES"):
        raw.set_channel_types(Config.preprocess.CHANNEL_TYPES)

    channel_types = mne.io.pick._picks_by_type(raw.info)
    has_meg = preprocess.contains_meg(raw.info)
    if has_meg:
        try:
            # Check if data has been SSS filtered. Raises RuntimeError if it
            # has
            mne.preprocessing.maxwell._check_info(raw.info)
            is_maxfiltered = False
        except RuntimeError:
            is_maxfiltered = True
            # When the continuous HPIs are turned on, this creates a lot of
            # noise. Annotate this segment as bad.

            # Mark everything prior to first event - 1 s as bad due to cHPI being
            # turned on
            onset = raw.first_time

            duration = raw.annotations.onset[0] - 1 - onset
            description = ["bad_chpi"]
            raw.annotations.append(onset, duration, description)

    raw.load_data()

    raw.filter(l_freq=Config.preprocess.L_FREQ, h_freq=Config.preprocess.H_FREQ)
    raw.filter(
        l_freq=Config.preprocess.L_FREQ,
        h_freq=Config.preprocess.H_FREQ,
        picks=["eog", "ecg"],
    )

    raw = raw.interpolate_bads()

    if preprocess.contains_eeg(raw.info):
        raw.set_eeg_reference(projection=True)

    return raw


def prepare_epochs(raw):
    tmin = Config.preprocess.TMIN
    tmax = Config.preprocess.TMAX
    stim_delay = Config.preprocess.STIMULUS_DELAY

    events, event_id = preprocess.events_from_annotations(raw)

    # Compensate for stimulus_delay
    tmin_delay = tmin + stim_delay
    tmax_delay = tmax + stim_delay

    # # we want baseline correction to get accurate noise cov estimate
    # # this may distort the ERP/ERF though !!
    # baseline_delay = (
    #     tmin_delay,
    #     stim_delay,
    # )  # use stim_delay since we roll back the evoked axis

    # print("Stimulus delay is {:0.0f} ms".format(stim_delay * 1e3))
    # print(
    #     "Epoching from {:0.0f} ms to {:0.0f} ms".format(
    #         tmin_delay * 1e3, tmax_delay * 1e3
    #     )
    # )
    # print(
    #     "Baseline correction using {:0.0f} ms to {:0.0f} ms".format(
    #         baseline_delay[0] * 1e3, baseline_delay[1] * 1e3
    #     )
    # )

    reject = dict(eog=100e-6) # or 200e-6

    print("Epoching...")
    epochs = mne.Epochs(
        raw, events, event_id, tmin_delay, tmax_delay, baseline=None,
        preload=True, reject=reject,
    )
    epochs.pick_types(meg=True, eeg=True)

    if not np.isclose(stim_delay, 0):
        print(f"Rolling back time axis to [{tmin}, {tmax}]")
        epochs.shift_time(tmin, relative=False)
        epochs.baseline = (tmin, 0)

    # # Evoked
    # evokeds = list()
    # for condition in event_id.keys():
    #     evoked = epochs[condition].average()
    #     evokeds.append(evoked)

    # evoked_file = raw_file.parent / (raw_file.stem + '-ave' + raw_file.suffix)
    # mne.evoked.write_evokeds(evoked_file, evokeds)

    return epochs


def prepare_concatenated_epochs(epochs):
    epochs = mne.concatenate_epochs(epochs)
    epochs.resample(Config.preprocess.S_FREQ)
    return epochs

def preprocess_subject(subject_id):
    io = utils.SubjectIO(subject_id)
    runs = mne_bids.get_entity_vals(io.bids["meg"].root, "run")
    io.data.update(stage="preprocess", suffix="meg")
    io.data.path.ensure_exists()

    picks = {m: True for m in Config.preprocess.MODALITIES}
    picks.update(eog=True, ecg=True)

    print("Processing")
    all_epochs = []
    for run in runs:
        print(f"Run : {run}")
        io.bids["meg"].update(run=run)
        io.data.update(run=run)

        raw = mne_bids.read_raw_bids(bids_path=io.bids["meg"])
        raw.pick_types(**picks)
        raw = preprocess_raw(raw)
        epochs = prepare_epochs(raw)
        all_epochs.append(epochs)

    io.data.update(processing="p", run=None)

    # Concatenate runs and downsample
    concat_epochs = prepare_concatenated_epochs(all_epochs)
    concat_epochs.save(io.data.get_filename(suffix="epo"))

    rng = np.random.default_rng()
    event_id_mapper = {
        "face/famous": "famous",
        "face/unfamiliar": "unfamiliar",
        "scrambled": "scrambled"
    }

    for event_id in concat_epochs.event_id:
        n = len(concat_epochs[event_id])
        split_indices = np.split(rng.permutation(n), [n // 2])
        # compute covariance
        for i, si in enumerate(split_indices):
            condition = event_id_mapper[event_id]

            epo = concat_epochs[event_id][si]
            evoked = epo.average()
            cov_noise = mne.compute_covariance(epo, tmin=None, tmax=0, method="shrunk", rank="info")
            cov_data = mne.compute_covariance(epo, tmin=0, tmax=None, method="shrunk", rank="info")

            io.data.update(condition=condition, split=str(i))

            np.save(io.data.get_filename(suffix="idx", extension=None), si)
            mne.write_evokeds(io.data.get_filename(suffix="evo"), evoked)
            cov_noise.save(io.data.get_filename(space="noise", suffix="cov"))
            cov_data.save(io.data.get_filename(space="data", suffix="cov"))


    # print("Autoreject")
    # epochs, ar = prepare_autoreject(epochs)
    # ar.save(io.data.get_filename(suffix="ar", extension="hdf5"), overwrite=True)
    # io.data.append(processing="a")
    # epochs.save(io.data.get_filename(suffix="epo"), overwrite=True)

    # for cov_type, cov_kw in Config.preprocess.COVARIANCE.items():
    #     cov = mne.compute_covariance(epochs, rank="info", **cov_kw)
    #     cov.save(io.data.get_filename(space=cov_type, suffix="cov"))

    # print("Denoise")
    # xdawn_dict = preprocess_denoise_xdawn(
    #     epochs, n_splits=Config.xdawn.N_SPLITS, n_rounds=Config.xdawn.N_ROUNDS
    # )
    # utils.write_pickle(
    #     xdawn_dict, io.data.get_filename(suffix="xdawn", extension="pkl")
    # )
    # epochs = preprocess_denoise_xdawn_transform(epochs, xdawn_dict)
    # io.data.append(processing="d")
    # for k, v in epochs.items():
    #     v.save(io.data.get_filename(space=k, suffix="epo"), overwrite=True)

    # for cov_type, cov_kw in Config.preprocess.COVARIANCE_XDAWN.items():
    #     cov = mne.compute_covariance(epochs["signal"], **cov_kw)
    #     cov.save(io.data.get_filename(space=cov_type, suffix="cov"))

    # print("Contrasts")
    # io.data.update(**Config.preprocess.USE_FOR_CONTRAST)
    # epochs = mne.read_epochs(io.data.get_filename(suffix="epo"))
    # evokeds = prepare_contrasts(epochs)
    # mne.write_evokeds(io.data.get_filename(suffix="ave"), evokeds)


if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    preprocess_subject(subject_id)