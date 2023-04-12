import mne_bids


# datalad status
# git reset --hard (remove modifications of files)
# git clean
# -n (show what would be removed)
# -f (remove files)
# -d (remove directories)


bids = mne_bids.BIDSPath(
    root=root['bids'],
    datatype='meg',
    session='meg',
    task='facerecognition'
)

bids_meg = mne_bids.BIDSPath(
    root=root['bids_derivative'],
    datatype='meg',
    session='meg',
    task='facerecognition',
    processing='sss'
)


create_derivative_symlinks(bids, bids_meg)


def create_derivative_symlinks(bids_path, bids_path_derivative):

    kw_channels = dict(suffix='channels', extension='.tsv')
    kw_events = dict(suffix='events', extension='.tsv')
    kw_scans = dict(suffix='scans', extension='.tsv')
    kw_headshape = dict(suffix='headshape', extension='.pos')
    kw_coordsystem = dict(suffix='coordsystem', extension='.json')

    for subject in mne_bids.get_entity_vals(bids_path_derivative.root, 'subject'):
        bids_path.update(subject=subject)
        dst = mne_bids.BIDSPath(
            root=bids_path_derivative.root,
                datatype='meg',
                session='meg',
                task='facerecognition',
                processing='sss',
        )
        dst.update(subject=subject)


        # DO NOT THE ORDER WITH WHICH SYMLINKS ARE CREATED!

        # Events
        for run in mne_bids.get_entity_vals(bids_path_derivative.root, 'run'):
            bids_path.update(run=run)
            src = mne_bids.read._find_matching_sidecar(bids_path, **kw_events, on_error='warn')
            dst.update(**kw_events, run=run)
            os.symlink(src, dst.fpath)
        dst.update(run=None)

        # Coordsystem
        src = mne_bids.read._find_matching_sidecar(bids_path, **kw_coordsystem, on_error='warn')
        dst.update(**kw_coordsystem)
        os.symlink(src, dst.fpath)

        # Headshape
        src = mne_bids.read._find_matching_sidecar(bids_path, **kw_headshape, on_error='warn')
        dst.update(**kw_headshape, task=None)
        os.symlink(src, dst.fpath)

        # Channels
        src = mne_bids.read._find_matching_sidecar(bids_path, **kw_channels, on_error='warn')
        dst.update(**kw_channels, datatype=None, task=bids_path.task)
        os.symlink(src, dst.fpath)


def remove_chpi_from_channels_sidecar(bids_path):
    """The CHPI channels are only present in the channel sidecar file and not
    in the raw files. If the channels do not match exactly, this will cause
    MNE-BIDS to throw an error. Thus, remove the CHPIs from the channel sidecar
    file.
    """
    # CHPI001 ... CHPI009
    chpi_channels = tuple('CHPI{:03d}'.format(i) for i in range(1, 10))
    channels_fname = mne_bids.read._find_matching_sidecar(bids_path,
                                                          suffix='channels',
                                                          extension='.tsv',
                                                          on_error='warn')
    channels_tsv = mne_bids.read._from_tsv(channels_fname)
    channels_tsv = mne_bids.tsv_handler._drop(channels_tsv, chpi_channels, 'name')
    mne_bids.tsv_handler._to_tsv(channels_tsv, channels_fname)

bids_root = r'C:\Users\jdue\Documents\phd_data\openneuro\ds000117'

n_subjects = 16
n_runs = 6
subjects = tuple('{:02d}'.format(i+1) for i in range(n_subjects))
runs = tuple('{:02d}'.format(i+1) for i in range(n_runs))

kwargs = dict(
    root = bids_root,
    datatype = 'meg',
    session = 'meg',
    task = 'facerecognition',
    suffix = 'meg'
)
bids_path = mne_bids.BIDSPath(**kwargs)

for subject in subjects:
    bids_path.update(subject=subject)
    remove_chpi_from_channels_sidecar(bids_path)