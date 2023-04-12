import mne
import scipy.io
import subprocess

def enquote(path):
    return '\'' + str(path) + '\''

def make_forward(fieldtrip_subject_dir, path2script):
    """
    Grab files from the simnibs segmentation directory and save to fieldtrip.

    path2script :
        Path to fieldtrip_make_forward
    """
    fieldtrip_subject_dir_ = enquote(fieldtrip_subject_dir)
    path2script_ = enquote(path2script)

    #call = f'addpath({path2script}); fieldtrip_make_headmodel({fieldtrip_subject_dir});'
    #subprocess.run(['matlab', '-batch', call])

    call = f'addpath({path2script_}); fieldtrip_make_forward({fieldtrip_subject_dir_});'
    subprocess.run(['matlab', '-batch', call])

    return fieldtrip_subject_dir / 'forward.mat'

def convert_forward(forward, info, trans, src):
    """
    Taken from simnibs_eeg.utils.convert_forward
    """
    # (channels, sources, 3 )
    fwd = scipy.io.loadmat(forward)['forward']
    nchan, nsrc, nori = fwd.shape

    kwargs = dict(
        src=src,
        mri_head_t=trans,
        info=info,
        bem=None,
        mindist=0,
        n_jobs=1,
        meg=False,
        ignore_ref=True,
        allow_bem_none=True
    )
    _, _, _, _, _, eegnames, _, info, update_kwargs, _ \
        = mne.forward._make_forward._prepare_for_forward(**kwargs)

    cmd = ''
    info['command_line'] = cmd
    update_kwargs['info']['command_line'] = cmd

    # Convert forward solution from MRI coordinates to head coordinates
    # fwd = mne.transforms.apply_trans(trans, fwd, move=False)
    fwd = fwd @ trans['trans'][:3, :3].T
    fwd = fwd.reshape((nchan, nsrc*nori))
    # leadfield should be shape (3 * n_dipoles, n_sensors) so transpose
    fwd = mne.forward._make_forward._to_forward_dict(fwd.T, eegnames)
    fwd.update(**update_kwargs)

    filename = forward.parent / f'{forward.stem}-fwd.fif'
    mne.write_forward_solution(filename, fwd, overwrite=True)

    return filename