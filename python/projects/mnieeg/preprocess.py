import mne
import numpy as np

from projects.mnieeg import utils
from projects.mnieeg.config import Config


def choose_first_not_bad(candidates, raw):
    for candidate in candidates:
        if candidate not in raw.info["bads"]:
            return candidate


def preprocess(subject_id):
    io = utils.SubjectIO(subject_id)
    # sessions = utils.get_func_sessions(io.data)
    io.data.update(stage="preprocessing", suffix="eeg")

    # for session in sessions:
    # io.data.update(session=session)

    raw = mne.io.read_raw_fif(io.data.get_filename(stage=None), preload=True)

    # One subject does not have EMG channels so this is necessary
    # raw.drop_channels(
    #     [ch for ch in Config.preprocess.DROP_CHANNELS if ch in raw.info["ch_names"]]
    # )
    # raw.info["dig"] = [
    #     d
    #     for d in raw.info["dig"]
    #     if str(d["ident"]) not in Config.preprocess.DROP_CHANNELS
    # ]
    # raw.rename_channels(Config.preprocess.channel_renamer)

    # Filter and resample
    raw.notch_filter(Config.preprocess.NOTCH_FREQS)
    raw.filter(Config.preprocess.L_FREQ, Config.preprocess.H_FREQ)
    raw.resample(Config.preprocess.SAMPLING_FREQ)

    # raw.plot(duration=20, n_channels=32);
    # raw.plot_psd();

    # Flat segments/channels
    flat_annot, flat_bads = mne.preprocessing.annotate_flat(raw)

    if len(flat_bads) > 0:
        raw.info["bads"] += flat_bads
    if len(flat_annot) > 0:
        if len(raw.annotations) > 0:
            raw.annotations.append(flat_annot)
        else:
            raw.set_annotations(flat_annot)

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=0.999999, method="picard", max_iter="auto")
    ica.fit(raw, decim=2)

    # Eyeblink detection (VEOG)
    veog_channel = choose_first_not_bad(Config.preprocess.VEOG_CAND, raw)
    # veog_channel = Config.preprocess.channel_renamer(veog_channel)
    veog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=veog_channel)
    veog_inds, _ = ica.find_bads_eog(veog_epochs, ch_name=veog_channel)
    ica.apply(raw, exclude=veog_inds)

    # Horizontal eye movement detection (HEOG)
    heog_channel = choose_first_not_bad(Config.preprocess.HEOG_CAND, raw)
    # heog_channel = Config.preprocess.channel_renamer(heog_channel)
    heog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=heog_channel)
    heog_inds, _ = ica.find_bads_eog(heog_epochs, ch_name=heog_channel)
    ica.apply(raw, exclude=heog_inds)

    raw.interpolate_bads()
    raw.set_eeg_reference(projection=True)

    io.data.path.ensure_exists()
    ica.save(io.data.get_filename(suffix="ica"))
    raw.save(io.data.get_filename(suffix="eeg"))


def make_epochs(raw, seed=None):

    n_epochs = Config.inverse.NAVE
    n_discard = Config.covariance.N_DISCARD
    epoch_duration = Config.covariance.EPOCH_DURATION

    sfreq = raw.info["sfreq"]
    n_samples = int(epoch_duration * sfreq)
    onsets = np.arange(raw.first_samp, raw.last_samp, n_samples, dtype=int)
    onsets = onsets[n_discard:-n_discard]
    n_events = len(onsets)
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = onsets
    events[:, 2] = 999

    rng = np.random.default_rng(seed=seed)
    p = rng.permutation(n_events)
    event1, event2 = p[:n_epochs], p[n_epochs : 2 * n_epochs]
    events[event1, 2] = 1
    events[event2, 2] = 2
    event_id = dict(forward=1, inverse=2)

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=epoch_duration,
        baseline=(None, epoch_duration),
    )


def compute_covariance(subject_id):

    io = utils.SubjectIO(subject_id)
    # sessions = utils.get_func_sessions(io.data)
    io.data.update(stage="preprocessing")

    # for session in sessions:
    # io.data.update(session=session)

    raw = mne.io.read_raw(io.data.get_filename(suffix="eeg"))
    # Compute the covariance on the full data
    cov = mne.compute_raw_covariance(raw, method=Config.covariance.METHOD)
    cov.save(io.data.get_filename(suffix="cov"))

    # Compute the covariance on "epochs"
    epochs = make_epochs(raw, seed=int(subject_id))
    for event in epochs.event_id:
        cov = mne.compute_covariance(epochs[event], method=Config.covariance.METHOD)
        cov.save(io.data.get_filename(space=event, suffix="cov"))
