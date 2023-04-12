#%%

# from meeg_tools.pipelines import wakeman2015 as wm

import mne
import mne_bids

import wakeman.initialize




#%%

# Inputs
subject = '01'
path_to_data = r'C:\Users\jdue\Documents\phd_data\openneuro\ds000117' # bids_root
path_to_analysis = r'C:\Users\jdue\Documents\phd_data\analysis\ds000117'

#%%

# Evaluation of inverse solutions
# ===============================

def read_source_estimates(inverse_models, forward_models, conditions, filenamer, is_contrast=True):
    """Load source estimates into a dictionary indexed by a tuple of the form
    (condition, inverse, forward).

    If is_contrast is True then consider 'conditions' as contrasts instead.
    """
    filenamer.update(stage='inverse', suffix='source', extension=None)
    stcs = {}
    for cond in conditions:
        attr = 'contrast' if is_contrast else 'condition'
        val = cond.replace(' ', '') if is_contrast else cond
        filenamer.update(**{attr: val})
        for inv in inverse_models:
            filenamer.update(inverse=inv)
            for fwd in forward_models:
                filename = filenamer.get_filename(forward=fwd)
                stcs[(val, inv, fwd)] = mne.read_source_estimate(filename)
    filenamer.update(stage=None, inverse=None, suffix=None, extension='fif')
    filenamer.update(**{attr: None})
    return stcs

def read_inverse_operators(inverse_models, forward_models, conditions, filenamer):
    filenamer.update(stage='inverse', suffix='inv')
    invs = {}
    for cond in conditions:
        filenamer.update(condition=cond)
        for inv in inverse_models:
            filenamer.update(inverse=inv)
            for fwd in forward_models:
                filename = filenamer.get_filename(forward=fwd)
                invs[(cond, inv, fwd)] = mne.minimum_norm.read_inverse_operator(filename)
    filenamer.update(stage=None, condition=None, inverse=None, suffix=None)
    return invs


contrasts = [c['name'] for c in config['inverse']['contrasts']]
read_source_estimates(inverse_models, forward_models, contrasts, filenamer)


stc = stc_contrasts[contrast]['simnibs']['mne']
src = inv_conditions[condition]['simnibs']['mne']['src']

interpolate_surf_stc_to_mni_vol('dSPM', 'mne', 0.165, filenamer, simnibs_organizer)

import scipy.ndimage



def interpolate_surf_stc_to_mni_vol(inverse, forward, time, filenamer, simnibs_organizer):
    """

    src :
        Source space in head coordinates from forward or inverse operator
    """

    kwargs = dict(inverse=inverse, forward=forward)
    inv = filenamer.get_filename(**kwargs, suffix='inv')
    stc = filenamer.get_filename(**kwargs, suffix='source', extension=None)

    # Transformations
    to_mni = simnibs_organizer.get_path('m2m') / 'toMNI'
    subject_to_mni = nib.load(to_mni / 'Conform2MNI_nonl.nii')
    mni_to_subject = nib.load(to_mni / 'MNI2Conform_nonl.nii')

    # needs to be transformed to MNI space
    mask = simnibs_organizer.get_path('m2m') / 'surfaces' / 'cereb_mask.nii.gz'
    mask = nib.load(mask).get_fdata()
    mask = (mask == 1) | (mask == 2)

    return

def interpolate_surf_stc_to_mni_vol(stc, src, subject_to_mni, mni_to_subject, time, mask=None):
    """

    """
    # Get some stuff
    in_use = np.concatenate([s['inuse'] for s in src]).astype(np.bool)
    subject_vox_to_mni_coords = mni_to_subject.get_fdata()
    func = stc.data[:, stc.time_as_index(time)].squeeze()

    # Source space is assumed to be in head coordinates
    # Head to MRI (subject space)
    points = np.concatenate([s['rr'] for s in src])
    points = points[in_use]
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
        points = mne.transforms.apply_trans(head_mri_t, points)
    else:
        raise RuntimeError('Source space should be in head coordinates')

    # Convert to m
    points *= 1e3
    # MRI to voxels (subject space)
    # (mni_to_subject.affine = affine of original T1)
    mri_vox_t = np.linalg.inv(mni_to_subject.affine)
    points = mne.transforms.apply_trans(mri_vox_t, points)

    # Voxels (subject space) to coordinates (MNI)
    ndim = points.shape[1]
    points = np.stack(scipy.ndimage.map_coordinates(subject_vox_to_mni_coords[..., i], points.T) for i in range(ndim))
    points = points.T

    # Coordinates (MNI) to voxels (MNI)
    vox_mri_t_mni = subject_to_mni.affine
    mri_vox_t_mni = np.linalg.inv(vox_mri_t_mni)
    points = mne.transforms.apply_trans(mri_vox_t_mni, points)

    # Interpolate point cloud to grid
    # rbfi = scipy.interpolate.Rbf(*points.T, func)
    # grid = np.asarray([np.arange(i) for i in subject_to_mni.shape[:3]], dtype=np.object)
    # grid = np.asarray(np.meshgrid(*grid))
    # di = rbfi(*grid)

    points = pv.PolyData(points)
    points['func'] = func

    grid = pv.UniformGrid()
    grid.dimensions = subject_to_mni.shape[:3]
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid_interp = grid.interpolate(points, radius=3)

    data = grid_interp['func'].reshape(grid.dimensions, order='F')
    if mask is not None:
        data[~mask] = 0
    img = nib.Nifti1Image(data, vox_mri_t_mni)

    return img

# ================================


mne.minimum_norm.get_point_spread()
mne.minimum_norm.get_cross_talk()

functions = config['evaluation']['functions']
metrics = config['evaluation']['metrics']
forward_comparisons = config['evaluation']['forward_comparisons']
inverse_solutions = config['evaluation']['inverse']

res_metrics = dict()
res_metrics_diff = dict()

filenamer.update(stage='evaluation')

info = None #...

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)

for inv_name in inverse_solutions:
    filenamer.update(inverse=inv_name)
    for fwd_name in forward_models:

        filename = filenamer.get_filename(forward=fwd_name)

        if inv_name.lower() in ('mne', 'dspm', 'sloreta', 'eloreta'):
            inv = mne.minimum_norm.read_inverse_operator(filename)
            lambda2 = 1e05 # ...
            res_matrix = mne.minimum_norm.make_inverse_resolution_matrix(fwd, inv, inv_name, lambda2)
            src = inv['src']
        elif inv_name.lower() in ('lcmv', ):
            filters = mne.beamformer.read_beamformer(filename)
            res_matrix = mne.beamformer.resolution_matrix.make_lcmv_resolution_matrix(filters, fwd, info)
            src = filters['src']
        else:
            raise ValueError

        for function in functions:
            for metric in metrics:
                res_metric = mne.minimum_norm.resolution_metrics(res_matrix, src, function, metric)
                # Combine xyz per location
                res_metrics[(inv_name, fwd_name, function, metric)] = mne.minimum_norm.inverse.combine_xyz(res_metric)
        del res_matrix

    # Comparisons/differences
    for fwds in forward_comparisons:
        for function in functions:
            for metric in metrics:
                res_metrics_diff[(inv, fwds, function, metric)] = \
                    res_metrics[inv, fwds[0], function, metric] - \
                    res_metrics[inv, fwds[1], function, metric]

    # pickle...
    filename = filenamer.get_filename(stage='evaluation')
    write_object(res_metrics, filename)


# formats
# res_metric[inverse_solution][forward_solution][function][metric]
# res_metric_diff[inverse_solution][forward_comparison][function][metric]
# save res_metric, res_metric_diff

n_functions = len(functions)
n_metrics = len(metrics)

hist_kwargs = dict(bins='auto', alpha=0.5)

# Histograms of resolutions metrics (for each forward solution)

for inv in inverse_solutions:
    # Plot
    fig, axes = plt.subplots(n_functions, n_metrics, sharex='col', sharey='row')
    for (axes_row, function) in zip(axes, functions):
        for ax, metric in zip(axes_row, metrics):
            for fwd in forward_solutions:
                data = res_metric[inv, fwd, function, metric].data
                ax.hist(data, **hist_kwargs)
    # Label
    for ax, function in zip(axes[:, 0], functions):
        ax.set_ylabel(function)
    for ax, metric in zip(axes[-1], metrics):
        ax.set_xlabel(metric)
    axes[0,0].legend(forward_solutions)

    fig.sup_title(inv)
    fig.tight_layout()
    fig.savefig()

# Histogram of resolutions metrics (differences between forward solutions)

for inv in inverse_solutions:
    # Plot
    fig, axes = plt.subplots(n_functions, n_metrics, sharex='col', sharey='row')
    for (axes_row, function) in zip(axes, functions):
        for ax, metric in zip(axes_row, metrics):
            for fwds in forward_comparisons:
                data = res_metric_diff[inv, fwds, function, metric].data
                ax.hist(data, **hist_kwargs)
    # Label
    for ax, function in zip(axes[:, 0], functions):
        ax.set_ylabel(function)
    for ax, metric in zip(axes[-1], metrics):
        ax.set_xlabel(metric)
    axes[0,0].legend([f'{f1} - {f2}' for f1, f2 in forward_comparisons])

    fig.sup_title(inv)
    fig.tight_layout()
    fig.savefig()

# Do differences in forward solutions translate to differences in resolution metrics?

# Correlation coefficients between RDM/lnMAG and (absolute) resolution metric differences

fig, ax = plt.subplots(1, 1)

rho = {k:_dict3d for k in inverse_solutions}

forward_metrics = np.stack((rdm, lnmag))
for inv in inverse_solutions:
    for fwds in forward_comparisons:
        for function in functions:
            for metric in metrics:

                resolution_metric = res_metric_diff[inv][fwds][function][metric]
                r = np.corrcoef(forward_metrics, resolution_metric, rowvar=True)
                rho[inv][fwds][function][metric] = r[]
                ax.bar(r)

df = pd.Dataframe.from_dict(res_metric_diff)


# Scatter plot of RDM vs. resolution metric difference

fig, ax = plt.subplots(1, 1)
ax.scatter(rdm, res_metric_diff[inv][fwds][function][metric])




for function in functions:
    write_source_estimates(src, res_metrics[function], file_organizer)


# Plot forward solution on sensors
info = mne.pick_info(evoked.info, mne.pick_types(evoked.info, eeg=True))

i = 100
source_topography = forward['sol']['data'][:, i]
source_topography2 = forward2['sol']['data'][:, i]

fwd_simbio = scipy.io.loadmat(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\simbio\sub-01\forward.mat')['fwd']

source_topography = fwd_simnibs[:, 123, 0]
source_topography2 = fwd_simbio[:, 123, 0]

mne.viz.plot_topomap(source_topography, info, vmax=20)
mne.viz.plot_topomap(source_topography2, info)

mne.viz.plot_topomap(fwd_mne['sol']['data'][:, 2], info)
mne.viz.plot_topomap(fwd[:, 0, 2], info)
mne.viz.plot_topomap((fwd[:, 0, :] @ inv_trans['trans'][:3, :3])[:, 2], info)
mne.viz.plot_topomap(fwd_simnibs['sol']['data'][:, 2], info)

mne.viz.plot_topomap(fwd[:, 0, i], info)
mne.viz.plot_topomap(fwd_simbio[:, 0, i], info)

vertno = fwd_mne['src'][0]['vertno']

k = 1000
i = 1
mne.viz.plot_topomap(fwd_mne['sol']['data'][:, k * 3 + i], info)
mne.viz.plot_topomap(fwd_simnibs['sol']['data'][:, vertno[k] * 3 + i], info)


mne.viz.plot_topomap(fwd_simbio['sol']['data'][:, 100], info)





mb = src_to_multiblock(forward['src'])
mb['lh']['fwd'] = fwd_simnibs[40, :10000, 1]
mb['rh']['fwd'] = fwd_simnibs[40, 10000:, 1]

plotter = pv.Plotter()
plotter.add_mesh(mb)
plotter.view_yz(True)
plotter.show(use_ipyvtk=True)

# ================================



def write_source_estimates(src, stcs, file_organizer):
    """
    src
    stcs : dict
        Dictionary of stc objects.
    """

    # Plot source estimates on source space
    mb = pv.MultiBlock()
    for hemisphere in src:
        if hemisphere['id'] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
            name = 'lh'
        elif hemisphere['id'] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
            name = 'rh'
        else:
            raise ValueError
        points = hemisphere['rr'].astype(np.float32)
        triangles = hemisphere['tris']
        triangles = np.column_stack((np.full(triangles.shape[0], 3),
                                    triangles)).ravel()
        hemi = pv.PolyData(points, triangles)

        for k in stcs:
            if name == 'lh':
                data = stcs[k].lh_data
            elif name == 'rh':
                data = stcs[k].rh_data
            hemi[k] = data
        mb[name] = hemi

    filename = file_organizer.get_filename(
        suffix='hello1',
        extension='vtm')
    mb.save(filename)


dip = mne.beamformer.rap_music(evoked, forward, noise_cov, 5)

def write_source_estimate_music(src, subcorrs, file_organizer):
    """
    src
    stcs : dict
        Dictionary of stc objects.
    """

    subcorrs = [q,w]

    # Plot source estimates on source space
    mb = pv.MultiBlock()
    for hemisphere in src:
        if hemisphere['id'] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
            name = 'lh'
        elif hemisphere['id'] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
            name = 'rh'
        else:
            raise ValueError
        points = hemisphere['rr'].astype(np.float32)
        triangles = hemisphere['tris']
        triangles = np.column_stack((np.full(triangles.shape[0], 3),
                                    triangles)).ravel()
        hemi = pv.PolyData(points, triangles)

        for i, subcorr in enumerate(subcorrs):
            if name == 'lh':
                data = subcorr[:src[0]['rr'].shape[0]]
            elif name == 'rh':
                data = subcorr[src[0]['rr'].shape[0]:]
            hemi[f'Recursion {i}'] = data
        mb[name] = hemi

    filename = file_organizer.get_filename(
        suffix='music',
        extension='vtm')
    mb.save(filename)



x = pv.PolyData(src[0]['rr'], np.column_stack((np.full(src[0]['tris'].shape[0], 3),
                                    src[0]['tris'])).ravel())
x['r1']=subcorrs[0,:10000]#subcorr[:10000]
x['r2']=subcorrs[1,:10000]#subcorr[:10000]
#x.plot()
x.save(r'C:\Users\jdue\Documents\phd_data\wakeman2015_bids_analysis\sub-01\ses-meg\testvector2.vtk')

# A = adjacency (sparse) matrix
# G = (normalized) leadfield matrix
row_ind =
col_ind =
data = np.sum(G[A.indices] * G[A.indptr])
affinity = scipy.sparse.csr_matrix((data, (A.indptr, A.indices)), shape=A.shape)

clustering = sklearn.cluster.SpectralClustering(n_clusters, affinity='precomputed')
clustering.fit(affinity)
clusters = clustering.labels_ # clustering
