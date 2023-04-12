import mne_bids

from projects.facerecognition.config import Config

# datalad status
# git reset --hard (remove modifications of files)
# git clean
# -n (show what would be removed)
# -f (remove files)
# -d (remove directories)


def create_derivative_symlinks(bids_path, bids_path_derivative):
    """Create symlinks of sidecar files from original data to derivatives."""

    kw_channels = dict(suffix="channels", extension=".tsv")
    kw_events = dict(suffix="events", extension=".tsv")
    kw_headshape = dict(suffix="headshape", extension=".pos")
    kw_coordsystem = dict(suffix="coordsystem", extension=".json")
    # kw_scans = dict(suffix='scans', extension='tsv')

    dst = bids_path_derivative

    for subject in mne_bids.get_entity_vals(bids_path_derivative.root, "subject"):
        bids_path.update(subject=subject, datatype="meg")
        dst.update(subject=subject, datatype="meg")

        # DO NOT CHANGE THE ORDER WITH WHICH SYMLINKS ARE CREATED!

        # Events
        for run in mne_bids.get_entity_vals(bids_path_derivative.root, "run"):
            bids_path.update(run=run)
            src = mne_bids.read._find_matching_sidecar(
                bids_path, **kw_events, on_error="warn"
            )
            dst.update(**kw_events, run=run)
            dst.fpath.symlink_to(src)
        bids_path.update(run=None)
        dst.update(run=None)

        # Coordsystem
        src = mne_bids.read._find_matching_sidecar(
            bids_path, **kw_coordsystem, on_error="warn"
        )
        dst.update(**kw_coordsystem)
        dst.fpath.symlink_to(src)

        # Headshape
        src = mne_bids.read._find_matching_sidecar(
            bids_path, **kw_headshape, on_error="warn"
        )
        dst.update(**kw_headshape, task=None)
        dst.fpath.symlink_to(src)

        # Channels
        bids_path.update(datatype=None)
        src = mne_bids.read._find_matching_sidecar(
            bids_path, **kw_channels, on_error="warn"
        )
        dst.update(**kw_channels, datatype=None, task=bids_path.task)
        dst.fpath.symlink_to(src)

        # remove_chpi_from_channels_sidecar(bids_path)


def remove_chpi_from_channels_sidecar(bids_path):
    """The CHPI channels are only present in the channel sidecar file and not
    in the raw files. If the channels do not match exactly, this will cause
    MNE-BIDS to throw an error. Thus, remove the CHPIs from the channel sidecar
    file.
    """
    # CHPI001 ... CHPI009
    chpi_channels = tuple("CHPI{:03d}".format(i) for i in range(1, 10))
    channels_fname = mne_bids.read._find_matching_sidecar(
        bids_path, suffix="channels", extension=".tsv", on_error="warn"
    )
    channels_tsv = mne_bids.read._from_tsv(channels_fname)
    channels_tsv = mne_bids.tsv_handler._drop(channels_tsv, chpi_channels, "name")
    mne_bids.tsv_handler._to_tsv(channels_tsv, channels_fname)


if __name__ == "__main__":

    bids = mne_bids.BIDSPath(
        root=Config.path.BIDS, datatype="meg", session="meg", task="facerecognition"
    )
    bids_deriv = mne_bids.BIDSPath(
        root=Config.path.BIDS / "derivatives" / "meg_derivatives",
        datatype="meg",
        session="meg",
        task="facerecognition",
        processing="sss",
    )
    create_derivative_symlinks(bids, bids_deriv)
