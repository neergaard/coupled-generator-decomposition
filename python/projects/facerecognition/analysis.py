#%%
import autoreject
import json
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pyvista as pv

from projects import base
from projects.facerecognition.config import Config

# Inverse
def prepare_source_estimate_mne(evoked, forward, noise_cov, config,
                                file_organizer):
    kwargs = config['inverse']['mne']
    kwargs['snr_window'] = config['inverse']['snr_window']

    stc, inv, lambda2 = source_estimate_mne(evoked, forward, noise_cov, **kwargs)
    inv['info']['meas_file'] = ''
    inv['info']['mri_file'] = ''

    # filename = file_organizer.get_filename(
    #     inverse=kwargs['method'].lower(),
    #     suffix='inv'
    # )
    # mne.minimum_norm.write_inverse_operator(filename, inv)

    # filename = file_organizer.get_filename(
    #     inverse=kwargs['method'].lower(),
    #     suffix='source',
    #     extension=None # mne applies '.stc'
    # )
    # stc.save(filename)

    return stc, inv, lambda2

def prepare_source_estimate_beamformer(evoked, forward, data_cov, noise_cov,
                                       config, file_organizer):
    kwargs = {k:v for k,v in config['inverse']['beamformer'].items() if k not in ['method']}
    method = config['inverse']['beamformer']['method']

    stc, filters = source_estimate_beamformer(evoked, forward, data_cov, noise_cov, **kwargs)

    filename = file_organizer.get_filename(
        inverse=method.lower(),
        suffix='filters'
    )
    filters.save(filename, overwrite=True)

    filename = file_organizer.get_filename(
        inverse=method.lower(),
        suffix='source',
        extension=None # mne applies '-lh/rh.stc'
    )
    stc.save(filename)

    return stc

def prepare_source_estimate_music(evoked, forward, noise_cov, config, file_organizer):
    kwargs = config['inverse']['music']

    dipoles = source_estimate_music(evoked, forward, signal_cov, **kwargs)

    #file_organizer.update(suffix='residual')
    #inv.save(file_organizer.get_filename())

    filename = file_organizer.get_filename(
        processing=config['method'],
        suffix='dip'
    )
    dipoles.save(file_organizer.get_filename())

    return dipoles

# LOW LEVEL FUNCTIONS
# =============================================================================




########################
# COREGISTRATION

import vtk


def remap_points_tris(points, tris):
    u = np.unique(tris)
    remap = np.zeros(u.max() + 1, dtype=np.int)
    remap[u] = np.arange(len(u))
    tris = remap[tris]
    points = points[u]
    return points, tris

def get_skin_surface_from_mesh(simnibs_organizer):

    tissue = 'skin'
    tag = 1005

    mesh = pv.read_meshio(simnibs_organizer.match('*.msh'))
    m2m = simnibs_organizer.get_path('m2m')

    is_tissue = mesh['gmsh:geometrical'] == tag
    tris = mesh.cells_dict[vtk.VTK_TRIANGLE]

    skin = mesh.extract_cells(mesh['gmsh:geometrical'] == tag)

    return pv.PolyData(skin.points, skin.cells)

def prepare_forward_mne(info, trans, src, freesurfer_subject_dir,
                        filenamer):

    # config with conductivities..?

    # default conductivities = [0.3, 0.006, 0.3]
    subject = freesurfer_subject_dir.stem
    fs_subjects_dir = freesurfer_subject_dir.parent

    surfs = mne.make_bem_model(subject, subjects_dir=fs_subjects_dir)

    # Convert surfaces to original MRI space
    # vox : MRI (voxel) voxel indices
    # mri : MRI (surface RAS) freesurfer coordinates
    # ras : RAS (non-zero origin) real world coordinates (scanner coordinates but in RAS)
    # (mri_ras_t is tkr-RAS to scanner-RAS)
    # outputs: vox_ras_t, vox_mri_t, mri_ras_t, dims, zooms
    _, _, mri_ras_t, _, _ = \
         mne.source_space._read_mri_info(freesurfer_subject_dir / 'mri' / 'orig.mgz')
    # The BEM surfaces are in FreeSurfer's MRI coordinate system so convert to
    # world coordinates
    surfs = [mne.transform_surface_to(surf, mri_ras_t['to'], mri_ras_t) for surf in surfs]
    # Hack otherwise make_forward_solution will complain that the BEM model is
    # not in MRI (surface RAS) space, however, we work in scanner space and not
    # FreeSurfer space so just pretend like we are in MRI (surface RAS)
    for surf in surfs:
        surf['coord_frame'] = mri_ras_t['from']

    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(info, trans, src, bem)

    filename = filenamer.get_filename(forward='mne', suffix='fwd')
    mne.write_forward_solution(filename, fwd, overwrite=True)

    return fwd

def mne_prepare_inverse(evokeds, forward, noise_cov, method):
    """
    evokeds : list of mne.EvokedArray
        Evoked response of each condition.

    """

    # Setup the inverse operator
    kwargs = dict(
        info = evokeds[0].info,
        forward = forward,
        noise_cov = noise_cov
        #loose = 0 # fixed orientation
        #loose = 'auto', # default is 0.2 for surface source spaces
        #depth = 0.8,
    )
    inv = mne.minimum_norm.make_inverse_operator(**kwargs)

    # Estimate regularization from the average SNR over conditions
    #snr, snr_est = mne.minimum_norm.estimate_snr(evoked, inv)
    snr = np.stack(mne.minimum_norm.estimate_snr(evoked, inv)[0] for evoked in evokeds).mean(0)
    snr_estimate = snr[evokeds[0].time_as_index(Config.INVERSE.TOI)].mean()
    lambda2 = 1/snr_estimate**2
    print(f'lambda2 = {lambda2}')
    #lambda2 = 1e-7
    #print(lambda2)

    # Compute the inverse operator
    kwargs = dict(
        orig = inv,
        nave = np.round(np.mean([evoked.nave for evoked in evokeds])).astype(int),
        lambda2 = lambda2,
        method = method
    )
    inv = mne.minimum_norm.prepare_inverse_operator(**kwargs)
    inv['info']['meas_file'] = ''
    inv['info']['mri_file'] = ''

    return inv, lambda2

def mne_apply_inverse(evoked, inv, method):
    # Source time course
    kwargs = dict(
        evoked = evoked,
        inverse_operator = inv,
        method = method,
        #pick_ori = 'normal',
        return_residual=True,
        prepared = True
    )
    stc, residual = mne.minimum_norm.apply_inverse(**kwargs)
    # if loose = 0 or pick_ori = 'normal' then the result if signed
    #stc.transform(np.abs)

    #cost, we, jp = compute_MNE_cost(evoked, inv, lambda2, twin, return_parts=True)

    return stc#, residual


def beamformer_prepare_filters(evoked, forward, data_cov, noise_cov, pick_ori=None):
    kwargs = dict(
        info = evoked.info,
        forward = forward,
        data_cov = data_cov,
        reg = 0.5,
        noise_cov = noise_cov,
        pick_ori = pick_ori
    )
    filters = mne.beamformer.make_lcmv(**kwargs)
    return filters

def beamformer_apply_filters(evoked, filters):
    stc = mne.beamformer.apply_lcmv(evoked, filters)
    return stc

def source_estimate_beamformer(evoked, forward, data_cov, noise_cov,
                               pick_ori=None):


    stc = mne.beamformer.apply_lcmv(evoked, filters)

    return stc, filters

def source_estimate_music(evoked, forward, noise_cov, method, n_dipoles):

    if method == 'rap_music':

        kwargs = dict(
            evoked = evoked,
            forward = forward,
            noise_cov = noise_cov,
            n_dipoles = n_dipoles
        )
        dipoles = mne.beamformer.rap_music(**kwargs)

    elif method == 'trap_music':
        raise NotImplementedError
        trap_music()
        dipoles = mne.beamformer._rap_music._make_dipoles()

    return dipoles

#%%

def fit_dipole(evoked, noise_cov, bem, trans, times):
    """



    """
    # I/O
    basename, outdir, figdir = get_output_names(evoked)
    evoked = mne.read_evokeds(evoked)

    if isinstance(noise_cov, str):
        noise_cov = mne.read_cov(noise_cov)
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    if isinstance(trans, str):
        trans = mne.read_trans(trans)
    if isinstance(times, float):
        times = [times, times]
    assert isinstance(times, list) and len(times) == 2



    dips = list()
    for evo in evoked:
        print('Fitting dipole to condition {}'.format(evo.comment))
        #print('Fitting to time {} ms'.format([int(np.round(t*1e3)) for t in time]))
        evoc = evo.copy()
        evoc.crop(*times)
        dip, res = mne.fit_dipole(evoc, noise_cov, bem, trans)
        dips.append(dip)

        # Find best fit (goodness of fit)
        best_idx = np.argmax(dip.gof)
        best_time = dip.times[best_idx]
        best_time_ms = best_time*1e3

        # Crop evoked and dip
        evoc.crop(best_time, best_time)
        dip.crop(best_time, best_time)

        print('Found best fitting dipole (max GOF) at {:0.0f} ms'.format(best_time_ms))
        print('Estimating time course by fixing position and orientation')

        # Time course of the dipole with highest GOF
        # dip_fixed.data[0] : dipole time course
        # dip_fixed.data[1] : dipole GOF (how much of the total variance is explained by this dipole)
        dip_fixed = mne.fit_dipole(evo, noise_cov, bem, trans,
                           pos=dip.pos[best_idx], ori=dip.ori[best_idx])[0]
        fig = dip_fixed.plot(show=False)
        fig.suptitle('Dipole with max GOF at {:0.0f} ms'.format(best_time_ms))
        fig.set_size_inches(8,10)
        fig.show()

        # Plot residual field from dipole
        # Make forward solution for dipole
        fwd, stc = mne.make_forward_dipole(dip, bem, evo.info, trans)
        # Project to sensors
        pred_evo = mne.simulation.simulate_evoked(fwd, stc, evo.info, cov=noise_cov)
        # Calculate residual
        res_evo = mne.combine_evoked([evoc, -pred_evo], weights='equal')

        # Get min/max for plotting
        ch_type = mne.channels._get_ch_type(evo,None)
        scale = mne.defaults._handle_default('scalings')[ch_type]
        evo_cat = np.concatenate((evoc.data, pred_evo.data, res_evo.data))
        vmin = evo_cat.min()*scale
        vmax = evo_cat.max()*scale

        # Plot topomaps
        fig, axes = plt.subplots(1,4)
        plot_params = dict(vmin=vmin, vmax=vmax, times=best_time, colorbar=False)
        evoc.plot_topomap(time_format='Observed', axes=axes[0], **plot_params)
        pred_evo.plot_topomap(time_format='Predicted',axes=axes[1], **plot_params)
        plot_params['colorbar'] = True
        res_evo.plot_topomap(time_format='Residual',axes=axes[2], **plot_params)
        fig.suptitle('Residual field from dipole at {:.0f} ms'.format(best_time_ms))
        fig.set_size_inches(10,4)
        fig.show() # savefig()

        # Plot on MRI

        # Dipole positions and vectors are given in head coordinates
        dip_mri = mne.transforms.apply(np.linalg.inv(trans['trans'], ))

        dip.plot_amplitudes()
        dip.plot()

        # check out code in
        # mne/viz/_3d.py - _plot_dipole_mri_orthoview

    return

def compute_whitener(inv, return_inverse=False):
    """
    Use mne.cov.compute_whitener to compute W which whitens AND transforms the
    data back to the original space
    """
    # Whitener
    eig = inv['noise_cov']['eig']
    nonzero = eig > 0
    ieig = np.zeros_like(eig)
    ieig[nonzero] = 1/np.sqrt(eig[nonzero])
    ieig = np.diag(ieig)
    eigvec = inv['noise_cov']['eigvec']
    W = ieig @ eigvec
    #W = eigvec.T @ ieig @ eigvec # what whiten_evoked does; projects it back to original space

    iW = eigvec.T @ np.diag(np.sqrt(eig))

    # such that iW @ W == I (approx.)

    if return_inverse:
        return W, iW
    else:
        return W

def compute_MNE_cost(evoked, inv, lambda2, twin, return_parts=False):
    """Compute the cost function of a minimum norm estimate. The cost function
    is given by

        S = E.T @ E + lambda2 * J.T @ R**-1 @ J

    where E = X - Xe, i.e., the error between the actual data, X, and the data
    as estimated (predicted) from the inverse solution, lambda2 is the
    regularization coefficient, J is the (full) current estimate (i.e., the
    inverse solution), and R is the source covariance which depends on the
    particular MNE solver.

    MNE employs data whitening and SVD of the gain matrix.



    See also

        https://martinos.org/mne/stable/manual/source_localization/inverse.html


    inv : *Prepared* inverse operator

    """
    if twin is not None:
        assert len(twin) == 2
        tidx = evoked.time_as_index(twin)
        X = evoked.data[:,slice(*tidx)]
    else:
        X = evoked.data
    #nave = evoked.nave

    # SVD gain matrix
    U = inv['eigen_fields']['data'].T # eigenfields
    V = inv['eigen_leads']['data']    # eigenleads
    S = inv['sing']                   # singular values


    #gamma_reg = S / (S**2 + lambda2)  # Inverse, regularized singular values
    gamma_reg = inv['reginv']
    Pi = S * gamma_reg                # In the case of lambda2 == 0, Pi = I

    # Source covariance
    R = inv['source_cov']['data'][:,None]
    iR = 1/R

    # Whitening operator
    W, iW = compute_whitener(inv, return_inverse=True)

    # Inverse operator
    M = np.sqrt(R) * V * gamma_reg[None,:] @ U.T


    # Current estimate (the inverse solution)
    J = M @ W @ X

    # Estimated (predicted) data from the inverse solution. If lambda2 == 0,
    # then this equals the actual data. If lambda2 > 0, there will be some
    # deviation (error) from this.
    Xe = iW @ U * Pi[None,:] @ U.T @ W @ X

    """
    # ALTERNATIVE
    # Calculating the estimated (predicted) data from the (full, i.e., all
    # components, x, y, z) inverse solution is done like so:

    # Get the solution ('sol') from mne.minimum_norm.inverse line 795. Don't
    # apply noise_norm to sol. Then undo source covariance (depth) weighting.
    # MNE also weighs by the effective number of averages ('nave') such that
    # R = R / nave that thus iR = 1/R * nave, however, since we are using a
    # *prepared* inverse operator, R has already been scaled by nave so no need
    # to undo this manually.
    J /= np.sqrt(R)

    # Calculate the whitened (and weighted) gain matrix
    # Note that the eigenfields are transposed, such that U is in fact
    # inv['eigen_fields']['data'].T. We get
    G = U * S[None,:] @ V.T

    # To be clear,
    #
    #   inv['eigen_fields'] - stores U.T
    #   inv['eigen_leads']  - stores V
    #
    # such that
    #
    #     gain = USV.T

    # Finally, the estimated (recolored) data is obtained from
    Xe = iW @ G @ J
    """
    #
    # Cost function
    #
    # The prediction term
    E = X - Xe    # Error
    WE = W @ E    # White error
    #WE = np.sum(WE**2)
    #JP = lambda2 * np.sum(J**2 * iR) # current norm penalty

    WE = np.sum(WE**2)
    JP = lambda2 * np.sum(J**2 * iR)

    # Cost in 'twin' or over all time points
    cost = WE + JP

    """
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(X.T*1e6)
    plt.subplot(3,1,2)
    plt.plot(Xe.T*1e6)
    plt.subplot(3,1,3)
    plt.plot(E.T*1e6)
    """
    if return_parts:
        return cost, WE, JP
    return cost
