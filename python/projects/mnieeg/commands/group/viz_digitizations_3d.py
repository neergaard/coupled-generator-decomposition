from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv

from simnibs.simulation import eeg

from projects.base import mpl_styles
from projects.base.colors import get_random_cmap
from projects.facerecognition.evaluation_viz_surf import make_image_grid_from_flat
from projects.mnieeg import utils
from projects.mnieeg.config import Config

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])
pv.set_plot_theme("document")


if __name__ == "__main__":
    io = utils.GroupIO()

    output_dir = Config.path.RESULTS / "figure_channel"
    if not output_dir.exists():
        output_dir.mkdir()

    pv_surf = pv.read(Config.path.RESOURCES / "mni152_surface.vtk")

    cap = eeg.make_montage("easycap_BC_TMS64_X21")
    cap_pos = cap.ch_pos[1:]
    pv_cap = pv.PolyData(cap_pos)

    pv_dig = pv.read(io.data.path.root / "dig_mni_proj.vtk")

    cmap = get_random_cmap(cap.n_channels)

    zoom_factor = 1.3
    view_vector = [(0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
    view_up = [(0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]

    res = 1000
    point_size = res * 1e-2
    imgs = []
    for vv, vu in zip(view_vector, view_up):
        p = pv.Plotter(off_screen=True, window_size=(res, res))
        p.add_mesh(pv_surf, color="#FFE0BD")
        p.add_mesh(
            pv_dig,
            cmap=cmap,
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        p.add_mesh(
            pv_cap,
            color="white",
            point_size=2 * point_size,
            render_points_as_spheres=True,
        )
        p.view_vector(vv)
        p.set_viewup(vu)
        p.camera.zoom(zoom_factor)
        p.enable_parallel_projection()
        imgs.append(p.screenshot(transparent_background=True))

    fig = make_image_grid_from_flat(
        imgs, (1, 5), mask_how="all not equal", mask_val=255, imgrid_kwargs=None,
    )
    fig.savefig(output_dir / "dig_mni_custom_3d.png")
