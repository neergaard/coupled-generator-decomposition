import json

import autoreject
import mne
import mne_bids
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from projects import base
from projects.facerecognition import utils
from projects.facerecognition.config import Config


def contains_meg(info):
    return mne.io.pick._contains_ch_type(info, "mag") or mne.io.pick._contains_ch_type(
        info, "grad"
    )


def contains_eeg(info):
    return mne.io.pick._contains_ch_type(info, "eeg")


def contains_eog(info):
    return mne.io.pick._contains_ch_type(info, "eog")


def contains_ecg(info):
    return mne.io.pick._contains_ch_type(info, "ecg")


def preprocess(subject_id):
    """Run all preprocessing steps from raw to evoked response.

    - Filtering (low, high, notch)
    - HEOG/VEOG/ECG epoching
    - Physiological artifact correction (ICA, RLS)
    - Epoching, concatenation of runs, and downsampling
    - AutoReject (repair or reject bad trials)
    - Denoise using XDAWN
    - Evoked contrasts

    Also computes noise and data covariance matrices.

    Processing annotations

        p : h/l/n filtering, physiological noise correction
        a : autoreject
        d : xdawn denoise

    """

    io = utils.SubjectIO(subject_id)
    runs = mne_bids.get_entity_vals(io.bids["meg"].root, "run")
    io.data.update(stage="preprocess", suffix="meg")
    io.data.path.ensure_exists()

    picks = {m: True for m in Config.preprocess.MODALITIES}
    picks.update(eog=True, ecg=True)

    print("Processing")
    epochs = []
    for run in runs:
        print(f"Run : {run}")
        io.bids["meg"].update(run=run)
        io.data.update(run=run)

        raw = mne_bids.read_raw_bids(bids_path=io.bids["meg"])
        raw.pick_types(**picks)
        raw, artifact_epochs, icas = preprocess_raw(raw)
        # for k, v in artifact_epochs.items():
        #     v.save(io.data.get_filename(processing=k, suffix="epo"), overwrite=True)
        # for k, v in icas:
        #     v.save(io.data.get_filename(processing=k, suffix="ica"))
        epochs.append(prepare_epochs(raw))

    io.data.update(processing="p", run=None)

    # Concatenate runs and downsample
    epochs = prepare_concatenated_epochs(epochs)
    epochs.save(io.data.get_filename(suffix="epo"), overwrite=True)

    print("Autoreject")
    epochs, ar = prepare_autoreject(epochs)
    ar.save(io.data.get_filename(suffix="ar", extension="hdf5"), overwrite=True)
    io.data.append(processing="a")
    epochs.save(io.data.get_filename(suffix="epo"), overwrite=True)

    for cov_type, cov_kw in Config.preprocess.COVARIANCE.items():
        cov = mne.compute_covariance(epochs, rank="info", **cov_kw)
        cov.save(io.data.get_filename(space=cov_type, suffix="cov"))

    print("Denoise")
    xdawn_dict = preprocess_denoise_xdawn(
        epochs, n_splits=Config.xdawn.N_SPLITS, n_rounds=Config.xdawn.N_ROUNDS
    )
    utils.write_pickle(
        xdawn_dict, io.data.get_filename(suffix="xdawn", extension="pkl")
    )
    epochs = preprocess_denoise_xdawn_transform(epochs, xdawn_dict)
    io.data.append(processing="d")
    for k, v in epochs.items():
        v.save(io.data.get_filename(space=k, suffix="epo"), overwrite=True)

    for cov_type, cov_kw in Config.preprocess.COVARIANCE_XDAWN.items():
        cov = mne.compute_covariance(epochs["signal"], **cov_kw)
        cov.save(io.data.get_filename(space=cov_type, suffix="cov"))

    # print("Contrasts")
    # io.data.update(**Config.preprocess.USE_FOR_CONTRAST)
    # epochs = mne.read_epochs(io.data.get_filename(suffix="epo"))
    # evokeds = prepare_contrasts(epochs)
    # mne.write_evokeds(io.data.get_filename(suffix="ave"), evokeds)


def preprocess_raw(raw):
    """

    * set bad channels
    * rename channels

    * detect (squid) jumps
    * detect eyeblinks
    * detect heartbeats

    * filter (hp/lp/notch/chpi)

    * Rereference EEG to average

    * Calculate noise covariance

    * physiological noise correction
    ** ICA and regression of eyeblinks and heartbeats


    fname_raw :
    channels :
    filt :
    noise_cov :
    stim_delay :
    phc :
    interactive : bool

    outdir : if None, use path of input file...

    """

    if hasattr(Config.preprocess, "DROP_CHANNELS"):
        raw.drop_channels(Config.preprocess.DROP_CHANNELS)
    if hasattr(Config.preprocess, "RENAME_CHANNELS"):
        raw.rename_channels(Config.preprocess.RENAME_CHANNELS)
    if hasattr(Config.preprocess, "CHANNEL_TYPES"):
        raw.set_channel_types(Config.preprocess.CHANNEL_TYPES)

    #     # Set bad channels
    #     if 'bads' in channels:
    #         raw.info["bads"] += channels['bads']
    #         if any(raw.info["bads"]):
    #             print("The following channels has been marked as bad:")
    #             print(raw.info["bads"])

    channel_types = mne.io.pick._picks_by_type(raw.info)
    has_meg = contains_meg(raw.info)

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

            # chpi_channel = channels['chpi_indicator']
            # chpi = mne.find_events(raw, chpi_channel, min_duration=10, verbose=False)
            # if chpi.any():
            #     print("Annotating pre-cHPI samples as bad")
            #     onset = [raw.first_samp/raw.info["sfreq"]]
            #     # add 0.25 which is the approximate duration of the cHPI SSS transition noise
            #     chpi_duration = [(chpi[0][0]-raw.first_samp)/raw.info["sfreq"] + 2]
            #     description = ["bad_cHPI_off"]
            #     raw.annotations.append(onset, chpi_duration, description)

            # print("Detecting squid jumps / high frequency stuff...")
            # na = len(raw.annotations)
            # compute_misc.detect_squid_jumps(raw)
            # try:
            #     print("Annotated {} segments as bad".format(len(raw.annotations)-na))
            # except TypeError:
            #     # no annotations
            #     print("Annotated 0 segments as bad")

    # Uses [x.max()-x.min()] / 4 as threshold
    artifact_epochs = dict().fromkeys(("VEOG", "HEOG", "ECG"))
    for k in artifact_epochs:
        if "EOG" in k:
            artifact_epochs[k] = mne.preprocessing.create_eog_epochs(
                raw, k, baseline=(None, None)
            )
        elif "ECG" in k:
            artifact_epochs[k] = mne.preprocessing.create_ecg_epochs(
                raw, k, baseline=(None, None)
            )
        else:
            raise ValueError

    # print("Detecting Eyeblinks")
    # artifact_epochs['veog'] = mne.preprocessing.create_eog_epochs(
    #    raw, "VEOG", baseline=(None,None))
    # artifact_epochs['heog'] = mne.preprocessing.create_eog_epochs(
    #    raw, "HEOG", baseline=(None,None))
    # print("Detecting Heartbeats")
    # artifact_epochs['ecg'] = mne.preprocessing.create_ecg_epochs(
    #    raw, "ECG", baseline=(None,None))

    #### Interactively annotate/mark data ####
    """
    if interactive:

        # Manual detection of miscellaneous artifacts
        # Plot raw measurements with eyeblink events

        print("Vizualizing data for manual detection of miscellaneous artifacts")
        raw.plot(events=eyeblinks, duration=50, block=True)

        # Manual detection of squid jumps
        print("Visualizing data for manual squid jump detection")
        raw_sqj = raw.copy().pick_types(meg=True).filter(l_freq=1, h_freq=7)
        raw_sqj.plot(events=eyeblinks, duration=50, block=True)

        raw.info["bads"] += raw_sqj.info["bads"]
        raw.info["bads"] = np.unique(raw.info["bads"]).tolist()

        #raw.annotations.append( raw_sqj.annotations )

        plt.close("all")
    """

    print("Filtering")
    # Also filter EOG and ECG channels such that frequencies are not
    # reintroduced by RLS regression

    # if has_meg:
    #     try:
    #         chpi_freqs = [hpi["coil_freq"] for hpi in raw.info["hpi_meas"][0]["hpi_coils"]]
    #     except IndexError:
    #         # no cHPI info
    #         chpi_freqs = []

    #     if any(chpi_freqs):
    #         print('Removing noise associated with HPI coils')
    #         raw.notch_filter(chpi_freqs)
    #         raw.notch_filter(chpi_freqs, picks=['eog', 'ecg'])

    raw.load_data()

    if Config.preprocess.NOTCH_FREQS is not None:
        raw.notch_filter(Config.preprocess.NOTCH_FREQS)
        raw.notch_filter(Config.preprocess.NOTCH_FREQS, picks=["eog", "ecg"])

    raw.filter(l_freq=Config.preprocess.L_FREQ, h_freq=Config.preprocess.H_FREQ)
    raw.filter(
        l_freq=Config.preprocess.L_FREQ,
        h_freq=Config.preprocess.H_FREQ,
        picks=["eog", "ecg"],
    )

    if contains_eeg(raw.info):
        print("Rereferencing EEG data to average reference")
        print("(not applying)")
        raw.set_eeg_reference(projection=True)

    # Physiological noise correction
    # =========================================================================
    print("Correcting for physiological artifacts")
    print("Using ICA and RLS")
    ranks = mne.compute_rank(raw, rank="info")
    icas = {}
    for channel_type in ranks:
        # Is maxwell filtering has been applied then mag and grad will be
        # combined as 'meg' and processed together
        picks = mne.pick_types(raw.info, **{channel_type: True})

        print(f"Processing {channel_type.upper()} channels")
        # print(f"Using {ranks[channel_type]} components for ICA decomposition")

        # Fit ICA
        # n_components=ranks[channel_type]
        ica = mne.preprocessing.ICA(n_components=0.999999, method="picard")
        ica.fit(raw, picks=channel_type, decim=5)
        icas[channel_type] = ica

        print("Identifying artifact components...")
        bad_idx = []
        for k in artifact_epochs:
            if "EOG" in k:
                inds, scores = ica.find_bads_eog(artifact_epochs[k])
            elif "ECG" in k:
                inds, scores = ica.find_bads_ecg(artifact_epochs[k])
            else:
                raise ValueError
            if any(inds):
                print(f"Found the following {k} related components: {inds}")
                bad_idx += inds
            else:
                print(f"Found no {k} related components!")
        bad_idx = list(set(bad_idx))

        if bad_idx:
            # Clean artifact ICA components using RLS and reconstruct data from ICA
            # ica.apply() only support zeroing out components completely so we have
            # to do it this way
            print("Constructing sources")
            sources = base.ica.unmix(ica, raw.get_data(channel_type))

            print("Regressing bad components")
            signal = sources[bad_idx]
            reference = raw.get_data(["eog", "ecg"])
            signal = base.filters.recursive_least_squares(signal, reference)
            sources[bad_idx] = signal

            print("Reconstructing data")
            raw._data[picks] = base.ica.mix(ica, sources)

    # preprocessed_raw_file = raw_file.parent / (prefix + raw_file.stem + raw_file.suffix)
    # #raw.save(preprocessed_raw_file, overwrite=True)

    # results = dict(
    #     raw=raw,
    #     eyeblink_epochs=eyeblink_epochs,
    #     heartbeat_epochs=heartbeat_epochs,
    #     icas=icas
    #     )

    # Remove EOG and ECG channels
    # names = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, eog=True, ecg=True)]
    # raw.drop_channels(names)

    for k in artifact_epochs:
        artifact_epochs[k].resample(sfreq=100)

    return raw, artifact_epochs, icas


def associated_emptyroom(bids_path):
    """Retrieve empty-room measurement associated with this subject.
    """
    datatype = "meg"
    sidecar_fname = mne_bids.read._find_matching_sidecar(
        bids_path, suffix=datatype, extension=".json", on_error="warn"
    )
    with open(sidecar_fname, "r", encoding="utf-8-sig") as fin:
        sidecar_json = json.load(fin)
        emptyroom = sidecar_json.get("AssociatedEmptyRoom")
    return bids_path.root / emptyroom


def events_from_annotations(raw):
    """MNE-BIDS reads and stores events as annotations. Reformat those to
    an 'events' and 'event_id' expected by mne.Epoch.
    """

    raw.annotations.description = raw.annotations.description.astype("<U15")
    raw.annotations.description[raw.annotations.description == "Famous"] = "face/famous"
    raw.annotations.description[
        raw.annotations.description == "Unfamiliar"
    ] = "face/unfamiliar"
    raw.annotations.description[
        raw.annotations.description == "Scrambled"
    ] = "scrambled"

    not_bad = [not desc.startswith("bad") for desc in raw.annotations.description]
    onset_ix = raw.time_as_index(raw.annotations.onset[not_bad])
    event_id = {
        e: i for i, e in enumerate(np.unique(raw.annotations.description[not_bad]))
    }
    events = np.zeros((sum(not_bad), 3), dtype=int)
    events[:, 0] = onset_ix
    events[:, 2] = [event_id[e] for e in raw.annotations.description[not_bad]]
    return events, event_id


def prepare_epochs(raw):
    """

    raw : MNE raw object


    * Find events
    * Epoch
    ** optionally calculate noise cov from prestim baseline
    ** optionally calculate signal cov from entire epoch


    """
    # assert raw_file.is_file()
    # figure_dir = raw_file.parent / 'figures'
    # if not figure_dir.exists():
    #     os.mkdir(figure_dir)

    # raw = mne.io.read_raw_fif(raw_file)

    # uV, fT/cm, fT
    # scale = dict(eeg=1e6, grad=1e13, mag=1e15) # the 'scalings' dict from evoked.plot()

    # print("Finding events")
    # events = mne.find_events(raw, stim_channel, consecutive=False)
    # valid_events = tuple(val for vals in event_codes.values() for val in vals)
    # events = events[np.isin(events[:, -1], valid_events)]
    # event_id = {}
    # for i, k in enumerate(event_codes):
    #     mask = np.isin(events[:, -1], event_codes[k])
    #     event_id[k] = i
    #     events[mask, -1] = i
    #     print(f"{k} : mapped {mask.sum()} events with code(s) {event_codes[k]} to {i}")
    # print(f'{len(events)} events kept')

    tmin = Config.preprocess.TMIN
    tmax = Config.preprocess.TMAX
    stim_delay = Config.preprocess.STIMULUS_DELAY

    events, event_id = events_from_annotations(raw)

    # Compensate for stimulus_delay
    tmin_delay = tmin + stim_delay
    tmax_delay = tmax + stim_delay

    # we want baseline correction to get accurate noise cov estimate
    # this may distort the ERP/ERF though !!
    baseline_delay = (
        tmin_delay,
        stim_delay,
    )  # use stim_delay since we roll back the evoked axis

    print("Stimulus delay is {:0.0f} ms".format(stim_delay * 1e3))
    print(
        "Epoching from {:0.0f} ms to {:0.0f} ms".format(
            tmin_delay * 1e3, tmax_delay * 1e3
        )
    )
    print(
        "Baseline correction using {:0.0f} ms to {:0.0f} ms".format(
            baseline_delay[0] * 1e3, baseline_delay[1] * 1e3
        )
    )

    print("Epoching...")
    epochs = mne.Epochs(
        raw, events, event_id, tmin_delay, tmax_delay, baseline_delay, preload=True
    )
    if not np.isclose(stim_delay, 0):
        print(f"Rolling back time axis to [{tmin}, {tmax}]")
        epochs.shift_time(tmin, relative=False)
        epochs.baseline = (tmin, 0)

    # # Evoked
    # evokeds = list()
    # for condition in event_id.keys():
    #     evoked = epochs[condition].average()
    #     evokeds.append(evoked)

    #     fig = evoked.plot(spatial_colors=True, exclude='bads', window_title=condition, show=False)
    #     fig.savefig(figure_dir / (raw_file.stem + f'_Evoked_{condition}.png'))

    # plt.close('all')

    # evoked_file = raw_file.parent / (raw_file.stem + '-ave' + raw_file.suffix)
    # mne.evoked.write_evokeds(evoked_file, evokeds)

    return epochs


def prepare_concatenated_epochs(epochs):
    """Concatenate epoched data across runs.

    Concatenate list of epochs from filenames.
    """
    print("Concatenating epochs")
    epochs = mne.concatenate_epochs(epochs)
    # According to the Nyquist theorem down-sampling to 2x the highest
    # remaining frequency should be alright, however, due to filter
    # imperfections, Michel (2019) recommends down-sampling to no less than 4x
    # the highest remaining frequency instead.
    # We downsample *after* epoching to avoid jittering of the event timings.
    epochs.resample(Config.preprocess.S_FREQ)

    return epochs


def prepare_autoreject(epochs):
    """

    Parameter space to use for cross validation

    n_interpolate (rho)
        The maxmimum number of "bad" channels to interpolate
    consensus (kappa)
        Fraction of channels which has to be deemed bad for an epoch to be
        dropped.
    """

    print(f"Autoreject running {Config.autoreject.N_JOBS} jobs")

    # Fit (local) AutoReject using CV
    # verbose=False # doesn't seem to do anything?
    ar = autoreject.AutoReject(
        consensus=Config.autoreject.CONSENSUS,
        n_interpolate=Config.autoreject.N_INTERPOLATE,
        n_jobs=Config.autoreject.N_JOBS,
    )
    epochs_clean = ar.fit_transform(epochs)

    # autoreject.get_rejection_threshold(epochs)

    # ar.get_reject_log(epochs)
    # labels = 0 (good), 1 (not interpolated), 2 (interpolated)

    return epochs_clean, ar


def preprocess_denoise_xdawn_transform(epochs, xdawn_dict):
    old_level = mne.set_log_level("warning", return_old_level=True)

    epochs_signal = epochs.copy()
    epochs_noise = epochs.copy()
    for ch_type in xdawn_dict:
        picks = mne.pick_types(epochs.info, **{ch_type: True})
        epochs_picks = epochs.copy().pick(picks)
        is_pooled_events = xdawn_dict[ch_type]["is_pooled_events"]
        for event in xdawn_dict[ch_type]:
            if event == "is_pooled_events":
                continue
            xdawn = xdawn_dict[ch_type][event]["xdawn"]
            signal_components = xdawn_dict[ch_type][event]["info"]["signal_components"]
            noise_components = xdawn_dict[ch_type][event]["info"]["noise_components"]

            print(f"Transforming {ch_type.upper()} {event}")
            # Check if events were pooled. If so,
            if is_pooled_events:
                sel = np.arange(len(epochs))
                epochs_picks.events[:, -1] = 0
                epochs_picks.event_id = {event: 0}
            else:
                sel = np.where(epochs.events[:, 2] == epochs.event_id[event])[0]

            epochs_picks_event = xdawn.apply(
                epochs_picks[event], include=list(signal_components)
            )[event]
            epochs_signal._data[sel[:, None], picks[None]] = epochs_picks_event._data
            epochs_picks_event = xdawn.apply(
                epochs_picks[event], include=list(noise_components)
            )[event]
            epochs_noise._data[sel[:, None], picks[None]] = epochs_picks_event._data

    mne.set_log_level(old_level)
    return dict(signal=epochs_signal, noise=epochs_noise)


def preprocess_denoise_xdawn(epochs, pool_events=True, n_splits=2, n_rounds=3):
    # xDAWN prints a lot of stuff
    old_level = mne.set_log_level("warning", return_old_level=True)

    if pool_events:
        original_events = epochs.events[:, -1]
        event = "pooled events"
        event_id = {event: 0}
    else:
        original_events = None

    ranks = mne.compute_rank(epochs, rank="info")
    xdawn_dict = {}
    for ch_type in ranks:
        xdawn_dict[ch_type] = {}
        print(f"Channels : {ch_type.upper()}")
        print("--------------")
        picks = mne.pick_types(epochs.info, **{ch_type: True})
        epochs_picks = epochs.copy().pick(picks)

        if pool_events:
            epochs_picks.events[:, -1] = event_id[event]
            epochs_picks.event_id = event_id

        n_channels = len(picks)
        n_components = ranks[ch_type]

        for event in epochs_picks.event_id:
            xdawn_dict[ch_type][event] = {}
            # signal_cov as estimated from all data?
            # Keep all components
            print(f'Fitting xDAWN to event type "{event}"')
            xdawn = mne.preprocessing.Xdawn(n_channels, reg="shrunk")
            xdawn.fit(epochs_picks[event])

            print("Determining signal space using CV")
            fit_info = xdawn_signal_space(
                xdawn,
                epochs_picks,
                event,
                n_components,
                original_events,
                n_splits,
                n_rounds,
            )
            sig_comp, noise_comp = (
                fit_info["signal_components"],
                fit_info["noise_components"],
            )
            filters = xdawn.filters_[event]
            patterns = xdawn.patterns_[event]
            # apply as projs['signal'] @ data for example
            projs = dict(
                signal=patterns[sig_comp].T @ filters[sig_comp],
                noise=patterns[noise_comp].T @ filters[noise_comp],
            )
            xdawn_dict[ch_type][event]["xdawn"] = xdawn
            xdawn_dict[ch_type][event]["info"] = fit_info
            xdawn_dict[ch_type][event]["projs"] = projs
            xdawn_dict[ch_type]["is_pooled_events"] = pool_events
    mne.set_log_level(old_level)

    return xdawn_dict


def xdawn_signal_space(
    xdawn,
    epochs_picks,
    event,
    n_components,
    original_events=None,
    n_splits=2,
    n_rounds=3,
    max_iter=5,
):

    # Discard null space
    # Initial sorting of the components. Also discard the null space.
    # sources = xdawn.filters_[event] @ epochs_picks[event].get_data()
    sources = xdawn.transform(epochs_picks[event])
    var = sources.mean(0).var(1)
    components_ = var.argsort()[::-1][:n_components]

    # Initial sorting of components based on how well they predict data
    # individually
    errors = xdawn_cv(
        xdawn,
        epochs_picks,
        event,
        components_,
        "each",
        original_events,
        n_splits,
        n_rounds,
    )
    errors_ = errors.mean((1, 2))
    components_sorter = errors_.argsort()
    components_ = components_[components_sorter]

    error_prev = np.inf
    rel_error = np.inf
    for _ in range(max_iter):
        # Variables ending with _ are associated with the configuration of
        # components currently being tested (components_)
        rmse_ = xdawn_cv(
            xdawn,
            epochs_picks,
            event,
            components_,
            original_events=original_events,
            n_splits=n_splits,
            n_rounds=n_rounds,
        )
        # We assume that the first component is signal (!) and find the
        # subsequent components which, when added, decreased the error
        mean_ = rmse_.mean((1, 2))
        diff_ = np.r_[-np.inf, np.diff(mean_)]
        improve_ = diff_ <= 0
        signal_components_ = components_[improve_]
        noise_components_ = components_[~improve_]

        # Compute the error using the components which seemed to decrease error
        # in this round of CV
        error_current = xdawn_cv(
            xdawn,
            epochs_picks,
            event,
            signal_components_,
            "all",
            original_events,
            n_splits,
            n_rounds,
        ).mean()
        rel_error = (error_prev - error_current) / error_current
        print(f"Previous error : {error_prev:.3f}")
        print(f"Current error  : {error_current:.3f}")
        print(f"Relative error : {rel_error * 100:.3f} %")

        if rel_error < 0.05:
            # Exit if performance either decreased (0 < 0.05) by less than 5 %
            # or increased (< 0)
            print(f"Relative error decreased/increased by {rel_error}. Using previous")
            break
        # If performance improved, save the info (i.e., drop the _)
        rmse = rmse_
        signal_components = signal_components_
        noise_components = noise_components_
        components = components_

        # And prepare for next iteration
        error_prev = error_current
        # Not really any good reason for this other than it should give a
        # nicer looking error curve next round
        components_ = components_[diff_.argsort()]

    return dict(
        rmse=rmse,
        signal_components=signal_components,
        noise_components=noise_components,
        # rmse[4] is the result of using components[:4] etc.
        components=components,
    )


def xdawn_cv(
    xdawn,
    epochs_picks,
    event,
    component_order,
    cv_type="incremental",
    original_events=None,
    n_splits=2,
    n_rounds=3,
):

    n_components = len(component_order)
    if cv_type in ("each", "incremental"):
        first = 1
    elif cv_type == "all":
        first = len(component_order)
    else:
        raise ValueError

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    rmse = np.zeros((n_components - first + 1, n_splits, n_rounds))
    for i in range(n_rounds):
        for j, (train, val) in enumerate(
            kf.split(range(len(epochs_picks[event])), original_events)
        ):
            scaler = StandardScaler()

            epochs_train = epochs_picks[event].copy().drop(val)
            epochs_val = epochs_picks[event].copy().drop(train)
            evoked_val = epochs_val.average().data

            scaler.fit(epochs_train.average().data)
            evoked_val = scaler.transform(evoked_val)

            for k in range(first, n_components + 1):
                if cv_type in ("incremental", "all"):
                    include = component_order[:k]
                elif cv_type == "each":
                    include = (component_order[k - 1],)
                epochs_train_proj = xdawn.apply(epochs_train, include=list(include))[
                    event
                ]
                evoked_train_proj = epochs_train_proj.average().data
                evoked_train_proj = scaler.transform(evoked_train_proj)
                error = evoked_val - evoked_train_proj
                rmse[k - first, j, i] = np.sqrt(np.mean(error ** 2))

    return rmse


# def covariance_emptyroom(raw, filter_freqs):
#     """

#     Use the same filt dict used to process the
#     functional raw data.

#     """

#     assert raw_file.is_file()
#     if not outdir.exists():
#         os.mkdir(outdir)
#     figure_dir = outdir / 'figures'
#     if not figure_dir.exists():
#         os.mkdir(figure_dir)

#     raw = mne.io.read_raw_fif(raw_file, preload=True)

#     print("Filtering emptyroom data as functional data")
#     print(f"Removing line noise at {filt['fnotch']} Hz")
#     #notch_freqs = [f for f in filt['fnotch'] if f<=raw.info['sfreq']/2]
#     raw.notch_filter(filt['fnotch'])

#     print(f"Highpass filtering at {filt['fmin']} Hz")
#     print(f"Lowpass filtering at {filt['fmax']} Hz")
#     raw.filter(filt['fmin'], filt['fmax'])

#     print("Computing covariance matrix")
#     noise_cov = mne.compute_raw_covariance(raw, method="shrunk")
#     stem = 'emptyroom_noise-cov'
#     emptyroom_cov_file = outdir / (stem + raw_file.suffix)
#     noise_cov.save(emptyroom_cov_file)

#     fig_cov, fig_eig = noise_cov.plot(raw.info, show=False)
#     fig_cov.savefig(figure_dir / (stem + "_covariance.pdf"))
#     fig_eig.savefig(figure_dir / (stem + "_eigenvalues.pdf"))
#     plt.close('all')

#     return emptyroom_cov_file


def combine_evoked(epochs, conditions, weights):
    if isinstance(conditions, str):
        conditions = [conditions]
    all_evoked = [epochs[c].average() for c in conditions]
    return mne.combine_evoked(all_evoked, weights)


def prepare_contrasts(epochs):
    evokeds = []
    for contrast in Config.conditions.CONTRASTS:
        conditions = contrast.conditions
        weights = contrast.weights
        combined_evoked = combine_evoked(epochs, conditions, weights)
        combined_evoked.comment = contrast.name
        evokeds.append(combined_evoked)
    return evokeds
