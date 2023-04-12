
import mne
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi

def interp_to_mni_vol(stc, src, subject_to_mni, mni_to_subject, trans, time, mask=None):
    """

    mask = nib.load(mask).get_fdata()
    mask = (mask == 1) | (mask == 2)

    subject_to_mni : Nifti1Image
        Maps subject voxels to MNI coordinates. The affine is that of the
        original T1.
    mni_to_subject : Nifti1Image
        Maps .

        The affine maps voxels in mni_to_subject.shape to MNI
        coordinates.

    nilearn.plotting.plot_glass_brain(img, threshold=25, cmap='jet', colorbar=True, plot_abs=False)
    nilearn.plotting.plot_stat_map(img) #, threshold=25, cmap='jet', colorbar=True, plot_abs=False)


    nilearn.plotting.plot_glass_brain(interp_img, threshold=3, cmap='jet', colorbar=True, plot_abs=False)
    nilearn.plotting.plot_stat_map(interp_img)

    """
    # Get some stuff
    inuse = np.concatenate([s['inuse'] for s in src]).astype(bool)
    #subject_vox_to_mni_coords = mni_to_subject.get_fdata()
    subject_vox_to_mni_coords = subject_to_mni.get_fdata()
    if isinstance(stc, mne.SourceEstimate):
        func = stc.data[:, stc.time_as_index(time)].mean(1)#.squeeze()
    elif isinstance(stc, np.ndarray):
        # vector of length n_points
        # should correspond to the positions in src!
        func = stc

    # Source space is assumed to be in head coordinates
    # Head to MRI (subject space)
    points = np.concatenate([s['rr'] for s in src])
    points = points[inuse]
    if src[0]['coord_frame'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD:
        head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
        points = mne.transforms.apply_trans(head_mri_t, points)
    else:
        raise RuntimeError('Source space should be in head coordinates')

    # Convert to m
    points *= 1e3
    # MRI to voxels (subject space)
    # (subject_to_mni.affine = affine of original T1)
    mri_vox_t = np.linalg.inv(subject_to_mni.affine)
    points = mne.transforms.apply_trans(mri_vox_t, points)

    # Voxels (subject space) to coordinates (MNI)
    ndim = points.shape[1]
    points = np.stack(ndi.map_coordinates(subject_vox_to_mni_coords[..., i], points.T) for i in range(ndim))
    points = points.T

    # Coordinates (MNI) to voxels (MNI)
    vox_mri_t_mni = mni_to_subject.affine
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
    grid.dimensions = mni_to_subject.shape[:3]
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid_interp = grid.interpolate(points, radius=3)

    data = grid_interp['func'].reshape(grid.dimensions, order='F')
    if mask is not None:
        data[~mask] = 0
    interp_img = nib.Nifti1Image(data, vox_mri_t_mni)

    return interp_img