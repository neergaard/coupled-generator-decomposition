from pathlib import Path

# from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.io
from vtk import VTK_HEXAHEDRON

from simnibs.mesh_tools import mesh_io

from projects.base import mpl_styles
from projects.base.mpl_styles import figure_sizing
from projects.base.colors import get_simnibs_cmap, tissue_to_simnibs_index
from projects.anateeg.config import Config
from projects.anateeg.utils import GroupIO, SubjectIO
from projects.facerecognition.evaluation_viz_surf import _crop_imgs_from_dict

pub_style = Path(mpl_styles.__path__[0]) / "publication.mplstyle"
plt.style.use(["default", "seaborn-paper", pub_style])
pv.set_plot_theme("document")


def apply_affine(points, affine):
    return points @ affine[:3, :3].T + affine[:3, 3]


def get_t1_vol(io):
    # fname = io.data.get_filename(
    #     session="mri", task=None, prefix="t1w", extension="nii.gz"
    # )
    fname = io.simnibs["charm"].get_path("m2m") / "segmentation" / "T1_denoised.nii.gz"
    t1 = nib.load(fname)

    ras_vox_t = np.linalg.inv(t1.affine)

    nii = pv.UniformGrid()
    nii.dimensions = np.array(t1.shape) + 1
    nii.origin = (-0.5, -0.5, -0.5)  # t1.affine[:3,3]
    nii.spacing = (1, 1, 1)  # (0.8506944, 0.8506944, 0.85)
    nii.cell_data["data"] = t1.get_fdata().ravel(order="F")

    return nii, ras_vox_t


def get_simnibs_surfs(io, key):
    m2m = io.simnibs[key].get_path("m2m")
    mesh = mesh_io.read(m2m / f"sub-{io.subject}.msh")
    mesh = mesh.remove_from_mesh(elm_type=4)  # keep surface elements only
    s = pv.make_tri_mesh(mesh.nodes.node_coord, mesh.elm.node_number_list[:, :3] - 1)
    s["tissue"] = mesh.elm.tag1 - 1001
    return s


def get_fieldtrip_surfs(io):

    headmodel = scipy.io.loadmat(
        Config.path.FIELDTRIP / f"sub-{io.subject}" / "headmodel_mesh.mat"
    )["mesh"][0][0]
    points, hexahedra = headmodel[1], headmodel[0] - 1
    points_tissue = headmodel[2].squeeze() - 1
    tissue_labels = np.array([i[0] for i in headmodel[3].squeeze()])
    # 4=mm, 5=ras, 6=cfg

    ft_to_simnibs_tissue = dict(
        white="wm", gray="gm", csf="csf", skull="bone", scalp="skin"
    )

    # Remap tissues
    remapper = np.array(
        [tissue_to_simnibs_index[ft_to_simnibs_tissue[t]] for t in tissue_labels]
    )
    tissue_labels = [ft_to_simnibs_tissue[t] for t in tissue_labels]
    tissue_labels = [tissue_labels[i] for i in remapper]
    points_tissue = remapper[points_tissue]
    tissue_order = np.sort(remapper)

    faces, counts = [], []
    for i, label in enumerate(tissue_labels):

        this_hexahedra = hexahedra[np.isin(points_tissue, tissue_order[: i + 1])]

        # uh = np.unique(hexahedra)
        # points = points_orig[uh]
        # z = np.zeros(hexahedra.max() + 1, dtype=hexahedra.dtype)
        # z[uh] = np.arange(len(uh))
        # hexahedra = z[hexahedra]

        cell_types = np.full(this_hexahedra.shape[0], VTK_HEXAHEDRON)
        cells = np.column_stack(
            (np.full(*this_hexahedra.shape), this_hexahedra)
        ).ravel()
        hex_mesh = pv.UnstructuredGrid(cells, cell_types, points)

        s = hex_mesh.extract_surface(pass_pointid=True)
        pids = s["vtkOriginalPointIds"]
        s.faces = np.column_stack(
            (np.full(s.n_cells, 4), pids[s.faces.reshape(-1, 5)[:, 1:]])
        ).ravel()

        faces.append(s.faces)
        counts.append(s.n_cells)

    faces = np.concatenate(faces)
    ccounts = [0] + [
        sum(counts[: i + 1]) for i in range(len(counts))
    ]  # cumulative counts

    s = pv.PolyData(points, faces)
    s["tissue"] = np.zeros(ccounts[-1], dtype=int)
    for i, label in enumerate(tissue_order):
        s["tissue"][ccounts[i] : ccounts[i + 1]] = label

    return s


def get_mne_surfs(io):

    orig = nib.load(Config.path.FREESURFER / f"sub-{io.subject}" / "mri" / "orig.mgz")
    # FreeSurfer voxel space to FreeSurfer RAS
    t_orig = orig.header.get_vox2ras_tkr()
    # FreeSurfer RAS to scanner RAS
    mri_ras_t = orig.affine @ np.linalg.inv(t_orig)

    surf_names = dict(inner_skull="csf", outer_skull="bone", outer_skin="skin")
    s = pv.MultiBlock()
    for name in surf_names:
        fname = Config.path.FREESURFER / f"sub-{io.subject}" / "bem" / f"{name}.surf"
        s[name] = pv.make_tri_mesh(*nib.freesurfer.read_geometry(fname))
        i = tissue_to_simnibs_index[surf_names[name]]
        s[name]["tissue"] = np.full(s[name].n_points, i)
    s = s.combine()
    s.points = apply_affine(s.points, mri_ras_t)
    return s


def get_slices(
    bg_img,
    surf,
    origin,
    axis,
    cmap,
    focus_point=None,
    parallel_scale=None,
    line_width=2,
):

    view_vector = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
    view_up = dict(x=(0, 0, 1), y=(0, 0, 1), z=(0, 1, 0))

    bg_slice = bg_img.slice(axis, origin)

    imgs = {}
    for m in surf:
        p = pv.Plotter(off_screen=True, window_size=(500, 500))
        p.add_mesh(bg_slice, cmap="gray")
        p.add_mesh(
            surf[m].slice(axis, origin),
            cmap=cmap,
            clim=[0, cmap.N - 1],
            line_width=line_width,
            # scalar_bar_args=dict(color="w"),
        )
        p.view_vector(view_vector[axis], view_up[axis])
        if focus_point is not None:
            p.set_focus(focus_point)
        p.enable_parallel_projection()
        p.parallel_scale /= parallel_scale or np.sqrt(2)
        p.remove_scalar_bar("data")
        p.remove_scalar_bar("tissue")

        imgs[m] = p.screenshot()
        p.close()

    return imgs


def plot_head_models():
    # origin = (150, 110, 200)
    origin = (10, 0, 0)  # in world RAS
    axis = "x"  # but this is voxel space...
    focus_point = origin[0], -55, 55
    focus_scale = 3 * np.sqrt(2)
    focus_line_width = focus_scale

    output_dir = Config.path.RESULTS / "figure_forward"
    tissue_cmap = get_simnibs_cmap()

    models = Config.forward.MODELS[1:]
    model_types = Config.forward.MODEL_TYPE[1:]
    final_model_names = [Config.plot.model_name[m] for m in models]

    io = GroupIO()
    for subject in io.subjects:
        print(f"Subject {subject}")

        iosub = SubjectIO(subject)

        nii, ras_vox_t = get_t1_vol(iosub)
        # round to avoid round-off errors causing issues with mesh not showing
        origin_vox = np.round(apply_affine(origin, ras_vox_t))
        focal_vox = np.round(apply_affine(focus_point, ras_vox_t))

        surf = {}
        for t, m, n in zip(model_types, models, final_model_names):
            if m == Config.forward.REFERENCE:
                continue
            if t == "simnibs":
                s = get_simnibs_surfs(iosub, m)
            elif t == "fieldtrip":
                s = get_fieldtrip_surfs(iosub)
            elif t == "mne":
                s = get_mne_surfs(iosub)
            s.points = apply_affine(s.points, ras_vox_t)
            surf[n] = s

        imgs = get_slices(nii, surf, origin_vox, axis, tissue_cmap, line_width=2.5)
        imgs_zoom = get_slices(
            nii,
            surf,
            origin_vox,
            axis,
            tissue_cmap,
            focal_vox,
            focus_scale,
            focus_line_width*2.5/1.5,
        )

        fig = make_image_grid_from_flat_custom(imgs, imgs_zoom)
        fig.savefig(output_dir / f"seg_models_sub-{iosub.subject}.png")

        plt.close("all")


def make_image_grid_from_flat_custom(imgs1, imgs2, width="double"):
    """
    imgs : dict of ndarrays

    """

    rows, cols, aspect1 = _crop_imgs_from_dict(imgs1, "all greater than", 3)
    imgs1 = {k: v[rows, cols] for k, v in imgs1.items()}
    rows, cols, aspect2 = _crop_imgs_from_dict(imgs2, "all greater than", 3)
    imgs2 = {k: v[rows, cols] for k, v in imgs2.items()}

    nrows, ncols = (2, len(imgs1))
    w = figure_sizing.fig_width["inch"][width]
    figsize = (
        w,
        w * nrows / ncols / ((aspect1 + aspect2) / 2) + 0.2,  # add space for titles
    )  # width, height

    fig = plt.figure(figsize=figsize)
    heights = (1 / aspect1, 1 / aspect2)
    spec = fig.add_gridspec(nrows, ncols, height_ratios=heights)
    for col, label in enumerate(imgs1):
        ax = fig.add_subplot(spec[0, col])
        ax.imshow(imgs1[label])
        ax.set_title(label)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    for col, label in enumerate(imgs2):
        ax = fig.add_subplot(spec[1, col])
        ax.imshow(imgs2[label])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig


# plotlines = qq.points[lines, 1:]

# plt.imshow(t1.dataobj[..., 120])
# plt.plot(plotlines[:, :, 0], plotlines[:, :, 1])


# which_slice = 100
# origin = (100, 100, 100)
# qq = q.slice("x", origin)
# lines = qq.lines.reshape(-1, 3)[:, 1:]

# xx = x.slice("x", origin)
# lines = xx.lines.reshape(-1, 3)[:, 1:]

# fig, ax = plt.subplots(figsize=(20, 20))
# ax.imshow(t1.dataobj[100].T, cmap="gray")
# lc = LineCollection(qq.points[lines][..., [1, 2]], colors=colors[qq["tissue"]])
# ax.add_collection(lc)
# ax.autoscale()


# lines = qq.lines.reshape(-1, 3)[:, 1:]
# qq.points[lines][:, :2]

# plt.figure()
# plt.plot(
#     qq.points[:, 1:][lines][:, :, 0].ravel(), qq.points[:, 1:][lines][:, :, 1].ravel()
# )

# qq.points[:, 1:][lines[qq["tissue"] == 6]]
# plt.Line2D()

# plt.tricontour(
#     x=qq.points[:, 1],
#     y=qq.points[:, 2],
#     triangles=qq.lines.reshape(-1, 3)[:, 1:],
#     Z=qq["tissue"],
# )
