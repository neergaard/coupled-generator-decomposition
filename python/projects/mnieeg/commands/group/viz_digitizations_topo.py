from pathlib import Path

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pyvista as pv

from simnibs.simulation import eeg

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.mnieeg import utils
from projects.mnieeg.config import Config
from projects.mnieeg.evaluation_viz_topo import create_info, choose_to_contour_levels

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])

if __name__ == "__main__":
    project_on_surface = True

    output_dir = Config.path.RESULTS / "figure_channel"
    if not output_dir.exists():
        output_dir.mkdir()

    io = utils.GroupIO()
    io.data.update(suffix="eeg")

    pv_surf = pv.read(Config.path.RESOURCES / "mni152_surface.vtk")
    surf = {"points": pv_surf.points, "tris": pv_surf.faces.reshape(-1, 4)[:, 1:]}

    # Get all digitizations in MNI coordinates
    fname = io.data.path.root / "dig_mni_proj.vtk"
    if fname.exists():
        print("Reading existing data")
        m = pv.read(fname)
        data = m.points.reshape(len(io.subjects), -1, 3)
    else:
        mne.set_log_level("warning")
        data = []
        print("Collecting data for all subjects")
        for subject, session in io.subjects.items():
            io.data.update(subject=subject, session=session)
            io.simnibs.update(subject=subject)

            raw = mne.io.read_raw_fif(io.data.get_filename(session=session))
            trans = mne.read_trans(
                io.data.get_filename(session=session, suffix="trans")
            )
            itrans = mne.transforms.invert_transform(trans)

            ch_pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
            ch_pos = 1e3 * mne.transforms.apply_trans(itrans, ch_pos)

            # to MNI
            m2m = io.simnibs.get_path("m2m")
            field = nib.load(m2m / "toMNI" / "Conform2MNI_nonl.nii.gz")
            montage = eeg.Montage(None, raw.info["ch_names"], ch_pos, "Electrode")
            montage.apply_deform(field)
            if project_on_surface:
                montage.project_to_surface(surf)
            data.append(montage.ch_pos)
        data = np.array(data)

        ch_label = np.tile(np.arange(data.shape[1]), data.shape[0])
        x = pv.PolyData(data.reshape(-1, 3))
        x["ch_label"] = ch_label
        x.save(io.data.path.root / "dig_mni_proj.vtk")

    dist_std = np.linalg.norm(data - data.mean(0), axis=2).std(0)
    xyz_std = np.abs(data - data.mean(0)).std(0)
    std = np.column_stack((dist_std, xyz_std))

    cap = eeg.make_montage("easycap_BC_TMS64_X21")
    cap_pos = cap.ch_pos[1:]

    dist_bias = np.linalg.norm(data.mean(0) - cap_pos, axis=-1)
    xyz_bias = np.abs(data - cap_pos).mean(0)
    bias = np.column_stack((dist_bias, xyz_bias))

    info = create_info()

    w = figure_sizing.fig_width["inch"]["double"]
    h = w / std.shape[1]
    fig, axes = plt.subplots(1, std.shape[1], figsize=(w, h), constrained_layout=True)
    cs_levels = choose_to_contour_levels(std.max()) # contour levels
    for ax, dat, title in zip(axes, std.T, ("Euclidean Distance", "x", "y", "z")):
        im, cs = mne.viz.plot_topomap(dat, info, axes=ax, cmap='viridis', contours=cs_levels, show=False)
        ax.clabel(cs, cs.levels, fontsize=6)
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=axes, shrink=1, pad=0.025)
    cbar.set_label("mm")  # , rotation=-90)
    fig.savefig(output_dir / "dig_mni_std.pdf")

    fig, axes = plt.subplots(1, bias.shape[1], figsize=(w, h), constrained_layout=True)
    cs_levels = choose_to_contour_levels(bias.max()) # contour levels
    for ax, dat, title in zip(axes, bias.T, ("Euclidean Distance", "x", "y", "z")):
        im, cs = mne.viz.plot_topomap(dat, info, axes=ax, cmap='viridis', contours=cs_levels, show=False)
        ax.clabel(cs, cs.levels, fontsize=6)
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=axes, shrink=1, pad=0.025)
    cbar.set_label("mm")  # , rotation=-90)
    fig.savefig(output_dir / "dig_custom_bias.pdf")
