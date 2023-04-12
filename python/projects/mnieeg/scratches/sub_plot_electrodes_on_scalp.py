from pathlib import Path
import sys

import mne
import nibabel as nib
import numpy as np
import skimage.measure
import pyvista as pv

def plot_electrodes_on_scalp(subject_id: int):

    subject = f'sub-{subject_id:02d}'
    session = 'ses-01'
    root = Path('/mnt/projects/PhaTMagS/jesper/')
    fname_t1w = root / 'data' / subject / 'ses-mri' / 'anat' / f'{subject}_ses-mri_T1w.nii.gz'
    fname_raw = root / 'analysis' / subject / session / 'stage-preprocessing' / f'{subject}_{session}_task-rest_eeg.fif'

    # Skin surface
    t1w = nib.load(fname_t1w)
    trans = mne.transforms.Transform('unknown','unknown',t1w.affine)
    v,f,_,_ = skimage.measure.marching_cubes(t1w.get_fdata(), level=500, step_size=2, allow_degenerate=False)
    vv = mne.transforms.apply_trans(trans, v)
    # nib.freesurfer.write_geometry('/mrhome/jesperdn/temp/sub01surf',vv,f)
    skin = pv.helpers.make_tri_mesh(vv,f)

    # Electrode positions
    raw = mne.io.read_raw_fif(fname_raw)
    points = np.array([i['loc'][:3] for i in raw.info['chs'][:63]])
    electrodes = pv.PolyData(points*1e3)

    p = pv.Plotter(shape=(2,3))
    p.subplot(0, 0)
    p.add_mesh(skin)
    p.add_mesh(electrodes, color='r')
    p.show_axes()
    p.view_xy()

    p.subplot(0, 1)
    p.add_mesh(skin)
    p.add_mesh(electrodes, color='r')
    p.show_axes()
    p.view_xz()

    p.subplot(0, 2)
    p.add_mesh(skin)
    p.add_mesh(electrodes, color='r')
    p.show_axes()
    p.view_xz(True)

    p.subplot(1, 0)
    p.add_mesh(skin)
    p.add_mesh(electrodes, color='r')
    p.show_axes()
    p.view_yz()

    p.subplot(1, 1)
    p.add_mesh(skin)
    p.add_mesh(electrodes, color='r')
    p.show_axes()
    p.view_yz(True)

    p.show()

if __name__ == '__main__':
    plot_electrodes_on_scalp(int(sys.argv[1]))