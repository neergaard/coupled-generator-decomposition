from mne.simulation import simulate_evoked, simulate_sparse_stc

def prepare_simulated(sd, config, i=None):

    if i is None:
        i = 0

    d = getd(sd)
    raw = getf(d['runs'][0], '{}*_raw.fif'.format(config['prefix']))[0]
    noise_cov = getf(d['cov'], '*_raw_noise-cov.fif')[0] # general noise cov
    fwdd = d['fwds'][i]
    fwdb = op.basename(fwdd)
    fwd = getf(op.join(fwdd, 'gain'), '*-fwd.fif')[0]
    outdir = op.join(d['contrasts'], 'Simulation_{}'.format(fwdb))

    print('Simulating evoked data using {:s}'.format(fwdb))
    evoked, stc = simulate_evoked_data(raw, fwd, noise_cov, config['nave'],
                                  outdir=outdir)
    return evoked, stc

def simulate_evoked_data(raw, fwd, noise_cov, nave=30, outdir=None):
    """

    raw : instance of Raw
        Instance of raw with information corresponding to that of the forward
        solution. This is used as a template for the simulated data.
    fwd : mne.forward.Forward
        Instance of fwd used to simulate the measured data.
    noise_cov : mne.cov.Covariance

    nave : int
        Number of averages in evoked data. This determines the SNR as noise is
        reduced by a factor of sqrt(nave).

    nave = (1 / 10 ** ((actual_snr - snr)) / 20) ** 2



    """

    # Output directories
    if outdir is None:
        outdir = op.dirname(raw)
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)

    base = get_output_names(raw)[0]
    base += '-ave'

    #### raw as template
    raw = mne.io.read_raw_fif(raw)
    info = raw.info
    # 'remove' SSS projector
    info['proc_history'] = []
    info['proj_id'] = None
    info['proj_name'] = None

    fwd = mne.read_forward_solution(fwd, force_fixed=True, surf_ori=True)
    noise_cov = mne.read_cov(noise_cov)

    # Make autocorrelations in the noise using an AR model of order n
    # (get the denominator coefficients only)
    iir_filter = mne.time_frequency.fit_iir_model_raw(raw, order=5, tmin=30, tmax=30+60*4)[1]
    iir_filter[1:] /= np.ceil(np.abs(iir_filter).max()) # unstable filter..?
    #iir_filter[np.abs(iir_filter) > 1] /= np.ceil(np.abs(iir_filter[np.abs(iir_filter) > 1]))
    iir_filter=None

    #rng = np.random.RandomState(42)

    # Time axis
    start, stop = -0.2, 0.5
    sfreq = raw.info['sfreq']
    times = np.linspace(start, stop, np.round((stop-start)*sfreq).astype(int))


    # Source time course
    np.random.seed(42)
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times,
                              random_state=42, data_fun=sim_er)

    # Noisy, evoked data

    #chs = mne.io.pick.channel_indices_by_type(info)

    # Pick MEG and EEG channels
    meg = [info['ch_names'][i] for i in mne.pick_types(info, meg=True, ref_meg=False)]
    eeg = [info['ch_names'][i] for i in mne.pick_types(info, meg=False, eeg=True, ref_meg=False)]


    # Simulate evoked data
    # simulate MEG and EEG data separately, otherwise the whitening is messed
    # up, then merge
    #noise_cov2 = noise_cov.copy()
    #noise_cov2.update(dict(data=noise_cov.data + np.random.random(noise_cov.data.shape) * noise_cov.data))
    evoked = simulate_evoked(mne.pick_channels_forward(fwd, meg), stc, info,
                             mne.pick_channels_cov(noise_cov, meg),
                             nave=nave, iir_filter=iir_filter)
    evoked_eeg = simulate_evoked(mne.pick_channels_forward(fwd, eeg), stc,
                                 info, mne.pick_channels_cov(noise_cov, eeg),
                                 nave=nave, iir_filter=iir_filter)
    evoked.add_channels([evoked_eeg])
    evoked.set_eeg_reference(projection=True)#.apply_proj()

    #evoked.data = mne.filter.filter_data(evoked.data, sfreq, None, 80, fir_design='firwin')
    evoked.comment = 'Simulation'
    #evoked.crop(-0.2, 0.5)
    #stc.crop(-0.2, 0.5)

    evoked.save(op.join(outdir, base + '.fif'))

    #picks = mne.pick_types(evoked.info, meg=False, eeg=True)
    fig = evoked.plot(spatial_colors=True, show=False)
    fig.savefig(op.join(figdir, base + '.png'))
    fig = evoked.plot_white(noise_cov, show=False)
    fig.savefig(op.join(figdir, base + '_whitened.png'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stc.times*1e3, stc.data.T*1e9)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (nAm)')
    ax.set_title('Simulated Sources')
    fig.savefig(op.join(figdir, 'simulated_sources.png'))

    plt.close('all')

    return evoked, stc

def sim_er(times):
    """Function to generate random source time courses"""

    sine = 50e-9 * np.sin(30. * times + np.random.randn(1)*200)

    peak = 0.2
    peakshift = 0.05
    duration = 0.01 # standard deviation of gaussian
    gaussian = np.exp(-(times - peak + peakshift * np.random.randn(1)) ** 2 / duration)
    return sine * gaussian

