from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from projects.base import mpl_styles
from projects.base.colors import get_simnibs_cmap
from projects.mnieeg import utils
from projects.mnieeg.config import Config
from projects.facerecognition.evaluation_viz_surf import make_image_grid

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])
pv.set_plot_theme("document")


def plot_slice(m, focus_point, cmap=None, parallel_scale=None):
    s = m.slice("y", focus_point)

    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.add_mesh(s, cmap=cmap, clim=[0, cmap.N - 1])
    p.view_xz()
    p.set_focus(focus_point)
    # p.set_focus((origin[0], origin[1], origin[2] + 30))
    p.enable_parallel_projection()
    p.parallel_scale /= parallel_scale or np.sqrt(2)
    p.remove_scalar_bar()
    return p.screenshot(transparent_background=True)


def plot_surf(m, cmap=None):
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.add_mesh(m, cmap=cmap, clim=[0, cmap.N - 1])
    p.camera.zoom(np.sqrt(2))
    p.remove_scalar_bar()
    return p.screenshot(transparent_background=True)


def prepare_mesh(m):
    m1 = m.remove_cells(m.celltypes == 5, inplace=False)  # tets
    m2 = m.remove_cells(m.celltypes == 10, inplace=False)  # tris
    tissue = m1["gmsh:physical"] - 1
    m1.clear_cell_data()
    m1["tissue"] = tissue
    tissue = m2["gmsh:physical"] - 1001
    is_gm = tissue == 1
    m2.remove_cells(~is_gm, inplace=True)  # keep gm only
    m2.clear_cell_data()
    m2["tissue"] = tissue[is_gm]
    return m1, m2


if __name__ == "__main__":

    focus_point = (-30,0,60)
    focus_scale = 2 * np.sqrt(2)
    titles = ["SimNIBS-CHARM", "MNI-Digitized"]

    output_dir = Config.path.RESULTS / "figure_forward"

    subject_id = 1

    io = utils.SubjectIO(subject_id)

    fname = io.simnibs_template.get_path("m2m") / f"sub-{io.subject}.msh"
    template = pv.read_meshio(fname)
    template, template_gm = prepare_mesh(template)

    fname = io.simnibs.get_path("m2m") / f"sub-{io.subject}.msh"
    charm = pv.read_meshio(fname)
    charm, charm_gm = prepare_mesh(charm)

    simnibs_cmap = get_simnibs_cmap()

    imgs = {
        "Slice":{
            'SimNIBS-CHARM':plot_slice(charm, focus_point, simnibs_cmap, focus_scale),
            'MNI-Digitized':plot_slice(template, focus_point, simnibs_cmap, focus_scale),
        },
        "GM":{
            'SimNIBS-CHARM':plot_surf(charm_gm, simnibs_cmap),
            'MNI-Digitized':plot_surf(template_gm, simnibs_cmap),
        },
    }

    fig = make_image_grid(imgs, 'all not equal', 255, width='single')
    for ax in fig.axes:
        ax.set_ylabel('')
    fig.savefig(output_dir / "headmodel_charm_vs_template.png")

    #fig = make_image_grid_from_flat(imgs, (1, 4), "all not equal", 255)
    #fig.savefig(output_dir / "headmodel_charm_vs_template1.png")
