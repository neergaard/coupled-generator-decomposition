from pathlib import Path

import numpy as np
import pyvista as pv

from projects.base.colors import get_simnibs_cmap
from projects.facerecognition.evaluation_viz_surf import make_image_grid
from projects.simval.config import Config

focus_point = (-50,-20,70)
parallel_scale = 4 * np.sqrt(2)

data_dir = Path('/mrhome/jesperdn/INN_JESPER/projects/simnibs_validations/simeeg/validation/head/fem_as_ref/models')
out_dir = Path('/mrhome/jesperdn/INN_JESPER/projects/simval/head/fem_ref/results')

fem = {'FEM 0.5': 'FEM_05', 'FEM 1.0': 'FEM_1'}
res = {'Original': '', 'Refined': '_refined'}
files = {r:{f:data_dir / fem[f] / f'almi5{res[r]}.msh' for f in fem} for r in res}

cmap = get_simnibs_cmap()

imgs = {r:{} for r in res}
for r in res:
    for f in fem:
        print((f,r))
        m = pv.read_meshio(files[r][f])

        s = m.slice("y", focus_point)
        s.remove_cells(s['gmsh:physical'] > 1000, inplace=True)
        tissue = s["gmsh:physical"] - 1
        s.clear_cell_data()
        s["tissue"] = tissue

        p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
        p.add_mesh(s, cmap=cmap, clim=[0, cmap.N - 1], show_edges=True)
        p.view_xz()
        p.set_focus(focus_point)
        p.enable_parallel_projection()
        p.parallel_scale /= parallel_scale or np.sqrt(2)
        p.remove_scalar_bar()
        imgs[r][f] = p.screenshot(transparent_background=True)

fig = make_image_grid(imgs, 'all not equal', 255, width='single')
fig.savefig(out_dir / 'fem_models_refined.png')
