import sys

import numpy as np
import pyvista as pv

from simnibs.simulation import eeg

from projects.mnieeg import utils


def montages_to_vtk(subject_id):
    """Collect the montages (original and projected) in a VTK multiblock file
    for easy visualization.
    """
    io = utils.SubjectIO(subject_id)

    mb = pv.MultiBlock()
    for fname in io.simnibs.match("montage_*.csv", "subject"):
        name = fname.stem[len("montage_") :]
        montage = eeg.make_montage(fname)
        mb[name] = montage_to_polydata(montage)

    mb.save(io.simnibs.get_path("subject") / "montages.vtm")


def montage_to_polydata(montage):
    points = montage.ch_pos
    if montage.landmarks:
        points = np.row_stack((points, montage.get_landmark_pos()))
    return pv.PolyData(points)


def remap_points_tris(points, tris):
    u = np.unique(tris)
    remap = np.zeros(u.max() + 1, dtype=int)
    remap[u] = np.arange(len(u))
    tris = remap[tris]
    points = points[u]
    return points, tris


# def get_skin_surface(m2m):

#     img = nib.load(m2m / "labeling.nii.gz")
#     data = img.get_fdata()
#     data = ~np.isin(data, (0, 517))  # background

#     # Create the spatial reference
#     grid = pv.UniformGrid()

#     # Set the grid dimensions: shape + 1 because we want to inject our values on
#     #   the CELL data
#     grid.dimensions = np.array(data.shape) + 1

#     grid.cell_arrays["data"] = data.flatten(order="F")  # Flatten the array!

#     surf = grid.extract_surface()

#     verts, faces, normals, _ = measure.marching_cubes(
#         data, level=0.5, step_size=2, allow_degenerate=False
#     )
#     verts = verts @ img.affine[:3, :3].T + img.affine[:3, 3]
#     surf = pv.make_tri_mesh(verts, faces)
#     surf.smooth(500, inplace=True)
#     return surf


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    montages_to_vtk(subject_id)
