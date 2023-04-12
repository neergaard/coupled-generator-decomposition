import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from pathlib import Path
from skimage import measure
import subprocess


def apply_trans(trans, x):
    return (trans[:3, :3] @ x.T + trans[:3, 3][:, None]).T


path = Path(r"C:\Users\jdue\googledrive\mnihead_ext")

# Distance to move nodes inwards (opposite point normal)
dist = 1
# The radius around each target point to find search for neighboring points to
# move as well
radius = 0
# The points around which to move nodes
targets = np.array(
    [
        [0, 82.9, -43],  # [0, 82.21, -41.98],  # Nasion
        [0, -116.2, -30.5],  # inion
        [79.37, -11.49, -50.82],  # RPA
        [-79.37, -8.73, -53.57],  # LPA
        [-19.29, -109.60, -45.85],
        [29.21, -111.65, -23.22],
        [49.05, -85.16, 55.70],
        [-54.57, -82.40, 54.60],
        [68.35, -48.73, 60.67],
        [-67.24, -55.36, 60.67],
        [-49.61, 3.70, 73.36],
        [43.10, -18.10, 83.34],
        [45.90, 48.68, 50.57],
        [-47.29, 56.86, 37.49],
    ]
)

box = pv.read(path / "nose_box.vtk")
img = nib.load(path / "labeling.nii.gz")
# bounding boxes for closing ear canals
right_coo = np.array([[71, -29, -67], [70, -17, -47]])
left_coo = np.array([[-68, -29, -65], [-69, -17, -48]])
# z coordinate from which to pad downwards
z_pad = np.array([[0, 0, -149]])

data = img.get_fdata()
data = ~np.isin(data, (0, 517))  # border and air?
trans = img.affine
inv_trans = np.linalg.inv(trans)

# Fill ear canals
right_vox = np.round(apply_trans(inv_trans, right_coo)).astype(int)
left_vox = np.round(apply_trans(inv_trans, left_coo)).astype(int)
data[tuple(slice(*v) for v in right_vox.T)] = True
data[tuple(slice(*v) for v in left_vox.T)] = True
data = ndi.binary_fill_holes(data)

# Pad downwards to create a stable base
z_pad_vox = np.round(apply_trans(inv_trans, z_pad)).astype(int)
z_pad_vox = z_pad_vox[0, -1]
data[..., 1:z_pad_vox] = data[..., z_pad_vox][..., None]
data[:, 110:140, 1:11] = False

# Add nose box
box.points[:, 0][box.points[:, 0] == -9] = -15
box.points[:, 0][box.points[:, 0] == 11] = 15
bounds = np.array(box.bounds).reshape(3, 2)
points = np.stack(
    np.meshgrid(*[np.arange(np.floor(b[0]), np.ceil(b[1]) + 1) for b in bounds])
)
points = pv.PolyData(points.reshape(3, -1).T)
select = points.select_enclosed_points(box, check_surface=False)[
    "SelectedPoints"
].astype(bool)
vox = np.round(apply_trans(inv_trans, points.points[select])).astype(int)
data[[v for v in vox.T]] = True

# Extract surface and transform to MNI coordinates
verts, faces, normals, _ = measure.marching_cubes(
    data, level=0.5, step_size=2, allow_degenerate=False
)
verts = apply_trans(trans, verts)
surf = pv.make_tri_mesh(verts, faces)
surf.smooth(1000, inplace=True)
surf.compute_normals(inplace=True)

# Find target points
tree = cKDTree(surf.points)
_, idx = tree.query(targets)
targets_refined = surf.points[idx]
results = tree.query_ball_point(targets_refined, r=radius)
results = np.concatenate(results)

scale = 57 / 59.5
surf.points *= scale

pv.save_meshio(path / "mni152_surface_57cm.off", surf)

# Move target points inwards
targets_refined = surf.points[idx]
surf.points[results] -= dist * surf.point_arrays["Normals"][results]
targets_refined_moved = targets_refined - dist * surf.point_arrays["Normals"][idx]

pv.save_meshio(path / "mni152_surface_57cm_with_targets.off", surf)

# Original target points as identified on the MNI152 template
np.savetxt(path / "targets.txt", targets)
pv.PolyData(targets).save(path / "targets.vtk")
# Closest node on the mesh to the original target points
np.savetxt(path / "targets_refined.txt", targets_refined)
pv.PolyData(targets_refined).save(path / "targets_refined.vtk")
# Closest node on the mesh after it has been moved. These are the actual MNI
# coordinates the measurements correspond to
np.savetxt(path / "targets_refined_moved.txt", targets_refined)
pv.PolyData(targets_refined_moved).save(path / "targets_refined_moved.vtk")

meshfix = r"C:\Users\jdue\Documents\git_repos\simnibs\main\simnibs\external\bin\win\meshfix.exe"
for s in (
    str(path / "mni152_surface_57cm"),
    str(path / "mni152_surface_57cm_with_targets"),
):
    subprocess.run(f"{meshfix} {s}.off".split())
    subprocess.run(
        f"{meshfix} {s}_fixed.off  --no-clean --fsmesh -o {s}_fixed.fsmesh".split()
    )
    subprocess.run(f"meshio-convert {s}_fixed.off {s}_fixed.stl")
    subprocess.run(f"meshio-binary {s}_fixed.stl")
    subprocess.run(f"meshio-convert {s}_fixed.off {s}_fixed.vtk")

surf = pv.read(path / "mni152_surface_57cm_fixed.vtk")
surf.points /= scale
surf.save(path / "mni152_surface_fixed.vtk")


# meshfix postprocessing and conversion

# meshfix mni152_surface.off
# meshfix mni152_surface_fixed.off --no-clean --fsmesh -o mni152_surface_fixed.fsmesh
# meshio-convert mni152_surface_fixed.off mni152_surface_fixed.stl
# meshio-binary mni152_surface_fixed.stl


# plotter = pv.Plotter()
# plotter.add_mesh(surf)
# plotter.add_mesh(pv.PolyData(targets_refined), color='r')
# plotter.show()

#%%

# from simnibs.mesh_tools import mesh_io

# def remap_points_tris(points, tris):
#     u = np.unique(tris)
#     remap = np.zeros(len(points), dtype=np.int)
#     remap[u] = np.arange(len(u))
#     tris = remap[tris]
#     points = points[u]
#     return points, tris

# def get_skin_surface_from_mesh(fname):
#     tag = 1005 # skin surface
#     mesh = pv.read_meshio(fname)
#     is_tissue = mesh['gmsh:geometrical'] == tag
#     tris = mesh.cells_dict[vtk.VTK_TRIANGLE]
#     skin = mesh.extract_cells(mesh['gmsh:geometrical'] == tag)
#     return pv.PolyData(skin.points, skin.cells)

# m = mesh_io.read_msh(path / 'mnihead.msh')
# m = m.crop_mesh(tags=1005)
# nn = m.nodes_normals().value

# tree = cKDTree(m.nodes.node_coord)
# _, idx = tree.query(targets)
# targets_refined = m.nodes.node_coord[idx]
# results = tree.query_ball_point(targets_refined, r=radius)
# results = np.concatenate(results)
# m.nodes.node_coord[results] -= dist*nn[results]

# mesh_io.write_stl(m, path / 'mnihead.stl')
# np.savetxt(path / 'targets.txt', targets)
# np.savetxt(path / 'targets_refined.txt', targets_refined)
