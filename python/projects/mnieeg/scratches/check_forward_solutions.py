import sys

import matplotlib.pyplot as plt
import mne
import mne.channels._standard_montage_utils
import numpy as np
import pyvista as pv

from projects.base.forward_evaluation import rdm, lnmag
from projects.mnieeg import utils
from projects.mnieeg.config import Config

def check_forward(subject_id, source=True, channel=False, force_fixed=False):

    io = utils.SubjectIO(subject_id)
    io.data.update(stage='forward', suffix='fwd')

    fname = Config.path.RESOURCES / (Config.forward.MONTAGE_MNE + ".txt")
    montage = mne.channels._standard_montage_utils._read_theta_phi_in_degrees(
        fname, mne.defaults.HEAD_SIZE_DEFAULT, add_fiducials=True
    )
    info = mne.create_info(montage.ch_names, 100, ch_types="eeg")
    info.set_montage(montage)


    force_fixed = False

    data = {}
    #infos = {}
    for model in Config.forward.MODELS:
        data[model] = {}

        # fname = r"C:\Users\jdue\Desktop\data\sub-01_ses-01_task-rest_fwd-digitized_fwd.fif"

        fwd = mne.read_forward_solution(io.data.get_filename(forward=model))
        n_chan, n_src = fwd['nchan'], fwd['nsource']
        src = fwd['src']
        if force_fixed:
            fwd = mne.convert_forward_solution(fwd, force_fixed=True)
            tmp = fwd['sol']['data'].reshape(n_chan, n_src, 1)
        else:
            tmp = fwd['sol']['data'].reshape(n_chan, n_src, 3)
        data[model]['lh'] = np.ascontiguousarray(tmp[:,:n_src//2])
        data[model]['rh'] = np.ascontiguousarray(tmp[:,n_src//2:])
        #infos[model] = fwd['info']

    #plot_forward_topomap(data, info, 3913)

    chan = 60

    mb = pv.MultiBlock()
    for s in src:
        h = 'lh' if s['id'] == 101 else 'rh'
        mb[h] = pv.make_tri_mesh(s['rr'], s['tris'])

    for h in mb.keys():
        for model in Config.forward.MODELS:
            if force_fixed:
                mb[h][f'n_{model}'] = data[model][h][chan]
            else:
                mb[h][f"x_{model}"] = data[model][h][chan, :, 0]
                mb[h][f"y_{model}"] = data[model][h][chan, :, 1]
                mb[h][f"z_{model}"] = data[model][h][chan, :, 2]

    #mb['lh'].set_active_scalars('x')
    #mb['lh'].plot()

    mb.save(f'/home/jesperdn/tmp_sub-{subject_id}_chan-{chan}.vtm')


src = eeg.FsAverage(int(Config.inverse.SOURCE_SPACE[-1]))
surf = src.get_central_surface()


def plot_forward_topomap(data,info, src_idx, hemi='lh'):

    n_forward = len(data)
    n_chan = 63
    n_src = 10242
    n_ori = data['digitized'][hemi].shape[-1]
    models = tuple(data.keys())
    one_info = isinstance(info, mne.Info)


    mn = min(data[model][hemi][:, src_idx].min() for model in models)
    mx = max(data[model][hemi][:, src_idx].max() for model in models)
    kwargs = dict(vmin=mn,vmax=mx,show=False)

    fig, axes = plt.subplots(n_ori, n_forward, figsize=(10,10*n_ori/n_forward), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i,(ori, row) in enumerate(zip(range(n_ori), axes)):
        for j,(model,ax) in enumerate(zip(models, row)):
            info_ = info if one_info else info[model]
            d = data[model][hemi][:, src_idx, ori]
            im, _ = mne.viz.plot_topomap(d, info_, axes=ax, **kwargs)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(f'ori = {ori}')
            if model != 'digitized':
                d_ref = data['digitized'][hemi][:, src_idx, ori]
                ax.set_xlabel(f'{rdm(d, d_ref):0.2f} / {lnmag(d, d_ref):0.2f}')
        cbar = fig.colorbar(im, ax=row, shrink=1, pad=0.025)
        cbar.set_label("uV")  # , rotation=-90)



fname = r'C:\Users\jdue\Desktop\forward\sub-01_ses-01_task-rest_fwd-digitized_fwd.fif'
fwd = mne.read_forward_solution(fname)

    pd.set_active_scalars("z")
    pd.plot()


#%%
models = Config.forward.MODELS

data = {}
data1 = {}
srcs = {}
infos = {}
for model in models:
    fname = r'C:\Users\jdue\Desktop\forward\sub-01_ses-01_task-rest_fwd-{}_fwd.fif'.format(model)
    fwd = mne.read_forward_solution(fname)
    data1[model] = fwd['sol']['data']
    x = fwd['sol']['data'][:, src*3:src*3+3]
    #print(np.round(x, 2)[:5])
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    infos[model] = fwd['info']
    y = fwd['sol']['data'][:, src]
    #print(np.round(y,2)[:5])
    data[model] = fwd['sol']['data']
    srcs[model] = fwd['src']


# print(rdm(data1['digitized'][:,src*3:src*3+3], data1['template_nonlin'][:, src*3:src*3+3]))
# print(rdm(data['digitized'][:,src], data['template_nonlin'][:, src]))

# x = rdm(data['digitized'], data['template_nonlin'])
# y = rdm(data1['digitized'], data1['template_nonlin'])

#%%

src = 5395
src = 3912
src = 6042
src = 345
src = 3611

print(1e3*fwd['src'][0]['rr'][src])

#mn,mx = -300,300
mn = min(data1[model][:, src*3:src*3+3].min() for model in models)
mx = max(data1[model][:, src*3:src*3+3].max() for model in models)


#%%
if __name__ == '__main__':
    sys.argv

    check_forward(subject_id)
