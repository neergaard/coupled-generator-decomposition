

# Check the event-related potentials/fields
import matplotlib.pyplot as plt
plt.style.use('dark_background')

bids_root = r'C:\Users\jdue\Documents\phd_data\openneuro\ds000117'
analysis_root = r'C:\Users\jdue\Documents\phd_data\analysis\ds000117'

config, org, root, subject_id = pipeline.initialize(1, bids_root, analysis_root)
fnamer = org['output']
fnamer.update(stage='preprocessing', processing='contrasts', space='signal', suffix='ave')

for i in range(1,3):
    subject_id = '{:02d}'.format(i)
    fnamer.update(subject=subject_id)

    kw = dict(spatial_colors=True, window_title=subject_id)

    evoked = mne.read_evokeds(fnamer.get_filename(), 'faces', verbose=False)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,10))
    for ax in axes:
        ax.grid(True, alpha=0.5)
    evoked.plot(axes=axes, **kw);

    #evoked = mne.read_evokeds(fnamer.get_filename(), 'faces vs. scrambled', verbose=False)
    #evoked.plot(**kw);

    input('Press any key to continue...')

print('Done')



#### PREPROCESSING

psd_kwargs = dict(tmin=0, tmax=np.inf, average=False,
                    reject_by_annotation=False,
                    line_alpha=0.5, show=False, verbose=False)

# PSDs before filtering
fig = raw.plot_psd(**psd_kwargs)
fig.savefig(figure_dir / "PSD_0_NoFilter_0-nyquist.png")
fig = raw.plot_psd(fmin=0, fmax=50, **psd_kwargs)
fig.savefig(figure_dir / "PSD_0_NoFilter_0-50.png")

fig = raw.plot_psd(**psd_kwargs)
fig.savefig(figure_dir / "PSD_1_Notch_0-nyquist.png")

fig = raw.plot_psd(fmin=0, fmax=50, **psd_kwargs)
fig.savefig(figure_dir / "PSD_2_HiLow_0-50.png")

fig = raw.plot_psd(**psd_kwargs)
fig.savefig(figure_dir / "PSD_3_HiLow_0-nyquist.png")

plt.close("all")


#### ICA

eog_inds, eog_scores = ica.find_bads_eog(eyeblinks_epoch)
if any(eog_inds):
    print(f"Found the following EOG related components: {eog_inds}")
    bad_idx += eog_inds

    # Plot correlation scores
    fig = visualize_misc.plot_ica_artifacts(ica, eog_inds, eog_scores, "eog")
    fig.savefig(figure_dir / f"ICA_EOG_{channel_type}_components", bbox_inches="tight")

    # Plot how components relate to eyeblinks epochs
    figs = ica.plot_properties(eyeblinks_epoch, picks=eog_inds,
                                psd_args={"fmax": 30}, show=False)
    try:
        figs = iter(figs)
    except TypeError:
        figs = iter([figs])
    for i,fig in enumerate(figs):
        fig.savefig(figure_dir / f"ICA_EOG_{channel_type}_component_{i}")
else:
    print("Found no EOG related components!")

ecg_inds, ecg_scores = ica.find_bads_ecg(heartbeats_epoch)
if any(ecg_inds):
    print("Found the following ECG related components: {}".format(ecg_inds))

    # Plot correlation scores
    fig = visualize_misc.plot_ica_artifacts(ica, ecg_inds, ecg_scores, "ecg")
    fig.savefig(figure_dir / f"ICA_ECG_{channel_type}_components", bbox_inches="tight")

    # Plot how components relate to heartbeat epochs
    figs = ica.plot_properties(heartbeats_epoch, picks=ecg_inds,
                                psd_args={"fmax": 80}, show=False)
    try:
        figs = iter(figs)
    except TypeError:
        figs = iter([figs])
    for i,fig in enumerate(figs):
        fig.savefig(figure_dir / f"ICA_ECG_{channel_type}_component_{i}")
else:
    print("Found no ECG related components!")

plt.close("all")

#### AUTOREJECT

q = mne.combine_evoked([epochs.average(), epochs_clean.average()], [1, -1])
q.plot()

# Evoked response after AR
evokeds = []
evokeds_clean = []
for event in epochs.event_id:
    evokeds.append(epochs[event].average())
    evokeds_clean.append(epochs_clean[event].average())

evoked_file = epochs_file.parent / (prefix + epochs_file.stem.replace('-epo', '-ave') + epochs_file.suffix)
mne.evoked.write_evokeds(evoked_file, evokeds_clean)

# Visualize results of autoreject
# ===============================
# Bad segments
fig_bad, fig_frac = visualize_misc.viz_ar_bads(ar, epochs)
fig_bad.savefig(figure_dir / "Epochs_bad_segments.pdf", bbox_inches="tight")
fig_frac.savefig(figure_dir / "Epochs_bad_fractions.pdf", bbox_inches="tight")

# Check for bad channels
bad_cutoff = 0.5
reject_log = ar.get_reject_log(epochs)
bad_frac = (np.nan_to_num(reject_log.labels) > 0).mean(0)
#bad_frac = ar.bad_segments.mean(0)
possible_bads = [epochs.ch_names[bad] for bad in np.where(bad_frac>bad_cutoff)[0]]

for evoked, clean_evoked in zip(evokeds, evokeds_clean):
    condition = clean_evoked.comment

    fig = clean_evoked.plot(spatial_colors=True, exclude='bads',
                            window_title=f'Evoked Response [{condition}]',
                            show=False)
    evoked_file = figure_dir / (prefix + epochs_file.stem + '_' + condition)
    fig.savefig(str(evoked_file) + '.png')

    if any(possible_bads):
        # Plot the bad channels before and after AR
        evoked.info["bads"] = possible_bads
        clean_evoked.info["bads"] = possible_bads

        kw_plot = dict(spatial_colors=False, exclude=[], show=False)

        title = f'Evoked Response [{condition}] before AutoReject'
        fig = evoked.plot(window_title=title, **kw_plot)
        fig.savefig(str(evoked_file) + "-bads-before.png")

        title = f'Evoked Reponse [{condition}] after AutoReject'
        fig = clean_evoked.plot(window_title=title, **kw_plot)
        fig.savefig(str(evoked_file) + "-bads-after.png")

    plt.close('all')

#### COVARIANCE
fig_cov, fig_eig = cov.plot(epochs.info, show=False)
fig_cov.savefig(figure_dir / (stem + "_covariance.pdf"))
fig_eig.savefig(figure_dir / (stem + "_eigenvalues.pdf"))
plt.close('all')


#### DENOISE

# denoise xDAWN ...

ch_type = 'meg'
event = 'face/unfamiliar'

xdawn = xdawn_dict[ch_type][event]['xdawn']
info = xdawn_dict[ch_type][event]['info']

mean = info['rmse'].mean((1, 2))
std = info['rmse'].std((1, 2))
diff = np.r_[-np.inf, np.diff(mean)]

plt.plot(mean)
plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color='gray', alpha=0.3)
plt.xlabel('Number of components') # says nothing about WHICH components


epochs_signal, epochs_noise = preprocess_denoise_xdawn_transform(epochs, xdawn_dict)

evoked_signal = epochs_signal.average()
evoked_noise = epochs_noise.average()
evoked_all = epochs_picks[event].average()

evoked_all.plot(spatial_colors=True);
evoked_signal.plot(spatial_colors=True);
evoked_noise.plot(spatial_colors=True);

rank_signal = mne.compute_rank(epochs_signal)
rank_noise = mne.compute_rank(epochs_noise)
print(f'Rank of signal subspace is {rank_signal}')
print(f'Rank of noise subspace is  {rank_noise}')

# explained variance
EV = np.sum(evoked_signal.data**2) / np.sum(evoked_all.data**2)
print(EV)

# Compute the noise covariance on the entire length of the noise epochs but
# using the signal space rank
noisecov = mne.compute_covariance(epochs_noise, method='shrunk', rank={'eeg':27, 'meg':27})
noisecov = mne.compute_covariance(epochs_noise, method='shrunk', rank=rank_signal)

evoked_signal.plot_white(noisecov);
evoked_noise.plot_white(noisecov);
noisecov_all = mne.compute_covariance(epochs, method='shrunk', tmax=0, rank='info')
evoked_all.plot_white(noisecov_all);


def preprocess_denoise_plot(filename):
    assert filename.is_file()
    figure_dir = filename.parent / 'figures'
    if not figure_dir.exists():
        os.mkdir(figure_dir)

    f = h5py.File(filename, 'r')
    n_ch_types = len(f)
    n_event_types = len(f['eeg'])

    fig, axes = plt.subplots(n_event_types, n_ch_types,
                             sharex='col', sharey=True, figsize=(10, 10))
    for ch_type, col in zip(f, axes.T):
        for event_type, ax in zip(f[ch_type], col):
            g = f[ch_type][event_type]

            mean = g['mean'][:]
            std = g['std'][:]
            ss = g['signal_space'][:]
            ns = g['noise_space'][:]
            n_components = len(mean)
            component_number = np.arange(1, n_components+1)

            ax.scatter(component_number[ss], mean[ss])
            ax.scatter(component_number[ns], mean[ns])
            ax.fill_between(component_number, mean-std, mean+std, alpha=0.3, facecolor='gray')

    fig.suptitle('Root Mean Squared Error')
    # Column titles
    for ax, title in zip(axes[0], tuple(c for c in f)):
        ax.set_title(title.upper())
    # Row titles
    for ax, title in zip(axes[:, 0], tuple(e for e in f[ch_type])):
        ax.set_ylabel(title)
    # Row titles
    for ax in axes[-1]:
        ax.set_xlabel('Component Number')
    fig.tight_layout()
    fig.savefig(figure_dir / 'denoise_decomp.pdf')
    f.close()

############



vtk = pv.PolyData(mri_fids_head*1e3)
vtk.save(coregdir / 'headcoord_'+mri_fids_base)

########


# forward solutions...

fwd_simnibs = mne.read_forward_solution(r'C:\Users\jdue\Documents\phd_data\wakeman2015_bids_analysis\sub-01\ses-meg\forward\sub-01_ses-meg_proc-simnibs_fwd.fif')
fwd_simbio = mne.read_forward_solution(r'C:\Users\jdue\Documents\phd_data\wakeman2015_bids_analysis\sub-01\ses-meg\forward\simbio-fwd.fif')


from mpl_toolkits.axes_grid1 import ImageGrid

i = 6426

# Set up figure and image grid
fig = plt.figure(figsize=(8, 4))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,2),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.15,
                )

# Add data to image grid
for ax, fwd in zip(grid, (fwd_simnibs['sol']['data'][:, i], fwd_simbio['sol']['data'][:, i]*1e6)):
    im, _ = mne.viz.plot_topomap(fwd, info, axes=ax, show=False)
    # vmin=0, vmax=1,

# Colorbar
ax.cax.colorbar(im)

axes[0].set_title('SimNIBS')
axes[1].set_title('Simbio')