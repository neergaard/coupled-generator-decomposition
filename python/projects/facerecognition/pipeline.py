

import json
import mne
import mne_bids
import nibabel as nib
import os

from projects.base.io import geometry, organize
from projects.base.sourcespace import sourcespace_from_charm
from projects.facerecognition import analysis, interpolation, fieldtrip
from projects.facerecognition.config import Config
from projects import simnibseeg

def associated_emptyroom(bids_path):
    """Retrieve empty-room measurement associated with this subject.
    """
    datatype = 'meg'
    sidecar_fname = mne_bids.read._find_matching_sidecar(bids_path,
                                                         suffix=datatype,
                                                         extension='.json',
                                                         on_error='warn')
    with open(sidecar_fname, 'r', encoding='utf-8-sig') as fin:
        sidecar_json = json.load(fin)
        emptyroom = sidecar_json.get("AssociatedEmptyRoom")
    return bids_path.root / emptyroom

def preprocess(subject_id):

    io = IOInit(subject_id)

    runs = mne_bids.get_entity_vals(io.bids_meg.root, 'run')

    print('Processing')

    io.filenamer.update(stage='preprocessing', suffix='meg')
    io.filenamer.path.ensure_exists()

    # Filter, correct for physiological artifacts, and epoch
    epochs = []
    for run in runs:
        print(f'Run : {run}')
        io.bids_meg.update(run=run)
        io.filenamer.update(run=run)

        raw = mne_bids.read_raw_bids(bids_path=io.bids_meg)
        picks = {m:True for m in Config.PREPROCESS.MODALITIES}
        picks.update(eog=True, ecg=True)
        raw.pick_types(**picks)
        raw = analysis.prepare_raw(raw, io.filenamer)
        epochs.append(analysis.prepare_epochs(raw))

    io.filenamer.update(processing='p', run=None)

    # Concatenate runs and downsample
    epochs = analysis.prepare_concatenated_epochs(epochs, io.filenamer)

    print('Autoreject')
    ar = analysis.prepare_autoreject(epochs, io.filenamer)
    io.filenamer.append(processing='a')
    epochs = analysis.prepare_autoreject_transform(epochs, ar, io.filenamer)

    analysis.prepare_covariance(epochs, 'noise', io.filenamer)
    analysis.prepare_covariance(epochs, 'data', io.filenamer)

    print('Denoise')
    xdawn_dict = analysis.prepare_denoise_xdawn(epochs, io.filenamer)
    io.filenamer.append(processing='d')
    epochs_signal, epochs_noise = analysis.prepare_denoise_xdawn_transform(
        epochs, xdawn_dict, io.filenamer
        )

    io.filenamer.update(processing='pad')
    analysis.prepare_covariance(epochs_signal, 'noise', io.filenamer)
    analysis.prepare_covariance(epochs_signal, 'data', io.filenamer)

    print('Contrasts')
    io.filenamer.update(**Config.USE)
    analysis.prepare_contrasts(epochs_signal, io.filenamer)

def prepare_for_forward(subject_id):
    """
    Coregister
    Create source space
    Create EEG layout in MRI space
    """

    io = IOInit(subject_id)

    runs = mne_bids.get_entity_vals(io.bids_meg.root, 'run')
    io.filenamer.update(stage='forward')
    io.filenamer.path.ensure_exists()

    # We just need the info object
    io.bids_meg.update(run=runs[0])
    info = mne_bids.read_raw_bids(io.bids_meg).pick_types(eeg=True).info

    # Get info, trans, and src
    mne.io.write_info(io.filenamer.get_filename(suffix='info'), info)
    trans = analysis.coregister_mri_head(info, io.bids_mri, io.filenamer)
    src = sourcespace_from_charm(io.simnibs, io.filenamer, Config.FORWARD.N_SOURCES)

    simnibseeg.utils.prepare_eeg_layout(info, trans, io.simnibs)
    analysis.coregistration_qa(io.simnibs, io.bids_mri, io.filenamer, info, trans)

    return info, src, trans

def make_forward_mne(subject_id):

    io = IOInit(subject_id)

    io.filenamer.update(stage='forward')

    # Get info, trans, and src
    info = mne.io.read_info(io.filenamer.get_filename(suffix='info'))
    trans = mne.read_trans(io.filenamer.get_filename(suffix='trans'))
    src = mne.read_source_spaces(io.filenamer.get_filename(suffix='src'))

    fs_subject_dir = Config.PATH.FREESURFER / subject_id
    forward = analysis.prepare_forward_mne(info, trans, src, fs_subject_dir, io.filenamer)

    return forward

def make_forward_simnibs(subject_id):

    io = IOInit(subject_id)

    io.filenamer.update(stage='forward')

    # Make forward
    forward = simnibseeg.utils.make_forward(io.simnibs)

    # Get info and trans
    # SimNIBS uses files in /surfaces as source space which should be the ones
    # used to create src!
    info = mne.io.read_info(io.filenamer.get_filename(suffix='info'))
    trans = mne.read_trans(io.filenamer.get_filename(suffix='trans'))
    forward = simnibseeg.utils.convert_forward('mne', forward, info, trans)

    # Move
    io.filenamer.update(forward='simnibs')
    forward = move_forward_solution(forward, io.filenamer)

    return forward

def make_forward_fieldtrip(subject_id):

    io = IOInit(subject_id)

    io.filenamer.update(stage='forward')

    # Make forward solution
    # (the segmentation and head model creation is also done here)
    fieldtrip_subject_dir = Config.PATH.SIMBIO / subject_id
    if not fieldtrip_subject_dir.exists():
        fieldtrip_subject_dir.mkdir(parents=True)
    # Create symlinks to stuff in fieldtrip directory
    sym_lh = fieldtrip_subject_dir / 'lh.gii'
    sym_rh = fieldtrip_subject_dir / 'rh.gii'
    sym_layout = fieldtrip_subject_dir / 'electrode_layout.csv'
    if not sym_lh.exists():
        os.symlink(io.simnibs.match('surfaces/lh.central.*.gii'), sym_lh)
    if not sym_rh.exists():
        os.symlink(io.simnibs.match('surfaces/rh.central.*.gii'), sym_rh)
    if not sym_layout.exists():
        os.symlink(io.simnibs.match('electrode_layout.csv', 'subject'), sym_layout)
    forward = fieldtrip.make_forward(fieldtrip_subject_dir, Config.PATH.SCRIPTS)

    # Get info, trans, and src
    info = mne.io.read_info(io.filenamer.get_filename(suffix='info'))
    trans = mne.read_trans(io.filenamer.get_filename(suffix='trans'))
    src = mne.read_source_spaces(io.filenamer.get_filename(suffix='src'))
    forward = fieldtrip.convert_forward(forward, info, trans, src)

    # Move
    io.filenamer.update(forward='fieldtrip')
    forward = move_forward_solution(forward, io.filenamer)

    return forward

def make_inverse(subject_id):

    io = IOInit(subject_id)

    conditions = ('famous', 'scrambled', 'unfamiliar', 'face')

    # Get data and covariance
    io.filenamer.update(stage='preprocessing', suffix='epo')
    io.filenamer.update(processing=Config.USE['processing'])
    epochs = mne.read_epochs(io.filenamer.get_filename(**Config.USE))
    evokeds = [epochs[c].average() for c in epochs.event_id.keys()]

    # filename = io.filenamer.get_filename(space='data', suffix='cov')
    # data_cov = mne.read_cov(filename)
    filename = io.filenamer.get_filename(space='noise', suffix='cov')
    noise_cov = mne.read_cov(filename)

    io.filenamer.update(processing=None) # reset
    io.filenamer.update(stage='inverse')
    io.filenamer.path.ensure_exists()

    # Compute evokeds
    evokeds_conditions = [epochs[c].average() for c in conditions]
    filename = io.filenamer.get_filename(suffix='ave')
    mne.write_evokeds(filename, evokeds_conditions)

    # Read forward models
    forwards = read_forward_solutions(Config.FORWARD.MODELS, io.filenamer)

    # Minimum norm estimate
    # =================================
    for inverse in Config.INVERSE.SOLVERS:
        inv_name = inverse['method']

        io.filenamer.update(inverse=inv_name, suffix='inv')
        inv_ops = {}
        lambdas2 = {}
        stcs = {}

        # Prepare inverse
        kwargs = dict(
            evokeds = evokeds,
            noise_cov = noise_cov,
            method = inv_name
        )
        for fwd_name, fwd in forwards.items():
            inv_op, lambda2 = analysis.mne_prepare_inverse(forward=fwd, **kwargs)
            inv_ops[(fwd_name, inv_name)] = inv_op
            lambdas2[(fwd_name, inv_name)] = lambda2

            filename = io.filenamer.get_filename(forward=fwd_name)
            mne.minimum_norm.write_inverse_operator(filename, inv_op)

        # Apply inverse
        for condition, evoked in zip(conditions, evokeds_conditions):
            io.filenamer.update(condition=condition)
            for (forward, inverse), inv_op in inv_ops.items():
                stc = analysis.mne_apply_inverse(evoked, inv_op, inv_name)
                stcs[(condition, forward, inverse)] = stc

        # Form contrasts and add
        stcs.update(make_contrasts(stcs, Config.CONTRAST.CONTRASTS))

        # Write STCs
        io.filenamer.update(suffix='source', extension=None)
        for (condition, forward, inverse), stc in stcs.items():
            filename = io.filenamer.get_filename(condition=condition,
                                                 forward=forward,
                                                 inverse=inverse)
            stc.save(filename)
        io.filenamer.update(condition=None, inverse=None, extension='fif')

        # Save all STCs to VTK
        # The basic source space should be the same
        mb = geometry.src_to_multiblock(forwards['simnibs']['src'])
        for k in stcs:
            geometry.add_stc_to_multiblock(mb, stcs[k], Config.INVERSE.TOI, '/'.join(k))
        filename = io.filenamer.get_filename(suffix='source', extension='vtm')
        mb.save(filename)

    """
    # Beamformer
    # ===================================
    config_beamformer = config['inverse']['models']['beamformer']

    beamformer_method = config_beamformer['method']
    assert beamformer_method.lower() == 'lcmv'

    filenamer.update(inverse=beamformer_method, suffix='filters')
    beamformer_filters = {}
    beamformer_regs = {}
    beamformer_stcs = {}

    # Prepare filters
    kwargs = dict(
        info = evoked.info,
        data_cov = data_cov,
        reg = config_beamformer['reg'],
        noise_cov = noise_cov,
        pick_ori = config_beamformer['pick_ori']
    )
    for fwd_name, fwd in fwds.items():
        filters = beamformer_prepare_filters(forward=fwd, **kwargs)
        filters = mne.beamformer.make_lcmv()
        beamformer_filers[fwd_name] = filters

        filename = filenamer.get_filename(forward=fwd_name)
        filters.save(filename, overwrite=True)

    # Apply filters
    filenamer.update(suffix='source', extension=None)
    for condition, evoked in zip(conditions, evokeds):
        filenamer.update(condition=condition)
        for fwd_name, filters in beamformer_filters.items():
            stc = beamformer_apply_filters(evoked, filters)
            beamformer_stcs[(condition, fwd_name)] = stc

            filename = filenamer.get_filename(condition=condition, forward=fwd_name)
            stc.save(filename)
    filenamer.update(condition=None)

    # Form contrasts
    for contrast in contrasts:
        beamformer_stcs = form_contrast_stc(contrast, beamformer_stcs, forward_models, filenamer)
    """

# nilearn.plotting.plot_surf_stat_map(
#     surf_mesh = [mb['lh'].points, mb['lh'].faces.reshape(-1, 4)[:, 1:]],
#     stat_map = mb['lh']['face/simnibs/dSPM'],
#     threshold = 10,
#     view = 'posterior'
#     )

# lh = mb['lh']
# lh.set_active_scalars('face/simnibs/dSPM')
# tlh = lh.threshold(10)

# pv.plot(
#     tlh
# )

def interpolate_to_mni_vol(subject_id, bids_root, analysis_root):
    """
    Interpolate an inverse solution from subject lh/rh surfaces to MNI
    volume.
    """

    config, org, root, subject_id = initialize(subject_id, bids_root, analysis_root)
    #config = config['interpolate_mni']

    conditions = config['visualize']['conditions']
    forward_models = config['visualize']['forward']
    inverse_models = config['visualize']['inverse']

    filename = org['output'].get_filename(stage='forward', suffix='trans')
    trans = mne.read_trans(filename)

    forwards = read_forward_solutions(forward_models, org['output'])
    # we only need the source space
    srcs = {k:v['src'] for k, v in forwards.items()}
    stcs = read_source_estimates(forward_models, inverse_models, conditions, org['output'])

    mni_map = org['simnibs'].get_path('m2m') / 'toMNI'
    kw = dict(
        subject_to_mni = nib.load(mni_map / 'Conform2MNI_nonl.nii.gz'),
        mni_to_subject = nib.load(mni_map / 'MNI2Conform_nonl.nii.gz'),
        trans = trans,
        time = config['visualize']['time'],
        mask = None
        )
    for (condition, forward, inverse), stc in stcs.items():
        interp_img = interpolation.interp_to_mni_vol(stc, srcs[forward], **kw)

        filename = org['output'].get_filename(
            stage='inverse',
            condition=condition,
            forward=forward,
            inverse=inverse,
            space='mni',
            extension='nii.gz'
            )
        interp_img.to_filename(filename)


# imgs = [nib.load(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\analysis\sub-01\ses-meg\stage-inverse\sub-01_ses-meg_space-mni_cond-facescrambled_inv-dSPM_fwd-{}.nii.gz'.format(i)) for i in forward_models]
# for img in imgs:
#     nilearn.plotting.plot_glass_brain(img, threshold=4, cmap='jet', colorbar=True, plot_abs=False)


def move_forward_solution(forward, file_organizer):
    dst = file_organizer.get_filename(suffix='fwd')
    os.rename(forward, dst)
    return dst

def read_forward_solutions(forward_models, filenamer, force_fixed=False):
    """Load forward solutions into a dictionary."""
    fwds = {}
    for fwd in Config.FORWARD.MODELS:
        filename = filenamer.get_filename(stage='forward', forward=fwd, suffix='fwd')
        fwds[fwd] = mne.read_forward_solution(filename)
        if force_fixed:
            fwds[fwd] = mne.convert_forward_solution(fwds[fwd], force_fixed=True)
    return fwds

def read_inverse_operators(inverse_models, forward_models, conditions, filenamer):
    filenamer.update(stage='inverse', suffix='inv')
    invs = {}
    for cond in conditions:
        filenamer.update(condition=cond)
        for fwd in forward_models:
            filename = filenamer.get_filename(forward=fwd)
            for inv in inverse_models:
                filenamer.update(inverse=inv)
                invs[(cond, fwd, inv)] = mne.minimum_norm.read_inverse_operator(filename)
    filenamer.update(stage=None, condition=None, inverse=None, suffix=None)
    return invs

def read_source_estimates(forward_models, inverse_models, conditions, filenamer):
    """Load source estimates into a dictionary indexed by a tuple of the form
    (condition, inverse, forward).

    If is_contrast is True then consider 'conditions' as contrasts instead.
    """
    filenamer.update(stage='inverse', suffix='source', extension=None)
    stcs = {}
    for condition in conditions:
        filenamer.update(condition=condition)
        for forward in forward_models:
            filename = filenamer.update(forward=forward)
            for inverse in inverse_models:
                filename = filenamer.get_filename(inverse=inverse)
                stcs[(condition, forward, inverse)] = mne.read_source_estimate(filename)
    filenamer.update(stage=None, condition=None, inverse=None, suffix=None, extension='fif')

    return stcs

def make_contrasts(stcs, contrasts):
    """
    # Apply np.abs before forming the contrast!
    """
    stcs_contrast = {}
    for contrast in Config.CONTRAST.CONTRASTS:
        for forward in Config.FORWARD.MODELS:
            for inverse in Config.INVERSE.SOLVERS:
                stc = sum(stcs[c, forward, inverse]*w for c, w in \
                          zip(contrast.conditions, contrast.weights))
                stcs_contrast[(contrast.name, forward, inverse)] = stc
    return stcs_contrast


