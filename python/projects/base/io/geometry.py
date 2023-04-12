import itertools
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
import os.path as op
import pyvista as pv

from ..geometry import triangle_normal

def cells_to_vtk(cells):
    assert cells.ndim == 2
    return np.column_stack((np.full(cells.shape[0], cells.shape[1]), cells)).ravel()

def src_to_multiblock(src):
    """Initialize a vtk multiblock from an mne.source_space.SourceSpaces
    object.
    """

    mb = pv.MultiBlock()
    for hemi in src:
        if hemi['id'] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
            name = 'lh'
        elif hemi['id'] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
            name = 'rh'
        else:
            raise ValueError
        # Convert to mm
        points = hemi['rr'].astype(np.float32) * 1e3
        triangles = hemi['tris']
        triangles = cells_to_vtk(triangles)
        mb[name] = pv.PolyData(points, triangles)
    return mb

def add_stc_to_multiblock(mb, stc, time, name):
    """Works in-place."""
    index = stc.time_as_index(time)

    for hemi in mb.keys():
        if hemi == 'lh':
            data = stc.lh_data
            vertno = stc.lh_vertno
        elif hemi == 'rh':
            data = stc.rh_data
            vertno = stc.rh_vertno

        if len(index) == 1:
            data = data[:, index].squeeze()
        elif len(index) == 2:
            data = data[:, slice(*index)].mean(1)
        else:
            raise ValueError

        if mb[hemi].n_points == vertno.shape[0]:
            mb[hemi][name] = data
        elif mb[hemi].n_points > vertno.shape[0]:
            # Put nan where no sources
            data_aug = np.full(mb[hemi].n_points, np.nan)
            data_aug[vertno] = data
            mb[hemi][name] = data_aug
        else:
            raise RuntimeError('Something is wrong here...')


def read_surface(fname):
    """Load a surface mesh. Return the
    vertices and faces of the mesh. If .stl file, assumes only one solid, i.e.
    only one mesh per file.

    gii
    off
    stl

    PARAMETERS
    ----------
    fname : str
        Name of the file to be read (.off or .stl file).

    RETURNS
    ----------
    vertices : ndarray
        Triangle vertices.
    faces : ndarray
        Triangle faces (indices into "vertices").
    """

    if fname.endswith('off'):
        with open(fname, "r") as f:
            # Read header
            hdr = f.readline().rstrip("\n").lower()
            assert hdr == "off", ".off files should start with OFF"
            while hdr.lower() == "off" or hdr[0] == "#" or hdr == "\n":
                hdr = f.readline()
            hdr = [int(i) for i in hdr.split()]

            # Now read the data
            vertices = np.genfromtxt(itertools.islice(f,0,hdr[0]))
            faces    = np.genfromtxt(itertools.islice(f,0,hdr[1]),
                                     usecols=(1,2,3)).astype(np.uint)

    elif fname.endswith('stl'):
        # mesh_flat
        #   rows 0-2 are the vertices of the 1st triangle
        #   rows 3-5 are the vertices of the 2nd triangle
        #   etc.

        # try to read as ascii
        try:
            mesh_flat = []
            with open(fname,"r") as f:
                for line in f:
                    line = line.lstrip().split()
                    if line[0] == "vertex":
                        mesh_flat.append(line[1:])
            mesh_flat = np.array(mesh_flat, dtype=np.float)

        except UnicodeDecodeError:
            # looks like we have a binary file
            with open(fname, "rb") as f:
                # Skip the header (80 bytes), read number of triangles (1
                # byte). The rest is the data.
                np.fromfile(f, dtype=np.uint8, count=80)
                np.fromfile(f, dtype=np.uint32, count=1)[0]
                data = np.fromfile(f, dtype=np.uint16, count=-1)
            data = data.reshape((-1,25))[:,:24].copy().view(np.float32)
            mesh_flat = data[:,3:].reshape(-1,3) # discard the triangle normals

        # The stl format does not contain information about the faces, hence we
        # will need to figure this out.

        # Get the unique vertices by viewing the array as a structured data
        # type where each column corresponds to a field of the same data type
        # as the original array (thus, each row becomes 'one object')
        # Use the inverse indices into the unique vertices as faces

        #dt = np.dtype([("", mesh_flat.dtype)] * mesh_flat.shape[1])
        #_, uidx, iidx = np.unique(mesh_flat.view(dt), return_index=True,
        #                          return_inverse=True)
        _, uidx, iidx = np.unique(mesh_flat, axis=0, return_index=True,
                                  return_inverse=True)
        # sort indices to preserve original ordering of the points
        q = np.argsort(uidx)
        vertices = mesh_flat[uidx[q]]
        # We have swapped around the points so we need to update iidx.
        # q[0] is the first point, thus we need to replace all
        # occurrences of q[0] in iidx with 0 and so on.
        # sort q to get the correct triangle mapping
        faces = np.argsort(q)[iidx].reshape(-1, 3)

    elif fname.endswith('gii'):
        gii = nib.load(fname)
        vertices, faces = gii.darrays[0].data, gii.darrays[1].data
    else:
        #raise IOError("Invalid file format. Only files of type .off and .stl are supported.")
        raise TypeError("Unsupported surface format '{}'".format(op.splitext(fname)[1][1:]))

    return vertices, faces

def write_surface(vertices, faces, fname, file_format="off", binary=True):
    """Save a surface mesh described by points in space (vertices) and indices
    into this array (faces) to an .off or .stl file.

    PARAMETERS
    ----------
    vertices : ndarray
        Array of vertices in the mesh.
    faces : ndarray, int
        Array describing the faces of each triangle in the mesh.
    fname : str
        Output filename.
    file_format : str, optional
        Output file format. Choose between "off" and "stl" (default = "off").
    binary : bool
        Only used when file_format="stl". Whether to save file as binary (or
        ascii) (default = True).

    RETURNS
    ----------
    Nothing, saves the surface mesh to disk.
    """
    nFaces = len(faces)
    file_format = file_format.lower()

    # if file format is specified in filename, use this
    if fname.split(".")[-1] in ["stl","off"]:
        file_format = fname.split(".")[-1].lower()
    else:
        fname = fname+"."+file_format

    if file_format == "off":
        nVertices = len(vertices)
        with open(fname, "w") as f:
            f.write("OFF\n")
            f.write("# (optional comments) \n\n")
        f = open(fname, 'a')
        np.savetxt(f,np.array([nVertices,nFaces,0])[np.newaxis,:],fmt="%u")
        np.savetxt(f,vertices,fmt="%0.6f")
        np.savetxt(f,np.concatenate((np.repeat(faces.shape[1],nFaces)[:,np.newaxis],faces),axis=1).astype(np.uint),fmt="%u")
        f.close()

    elif file_format == "stl":
        mesh = vertices[faces]
        tnormals  = triangle_normal(vertices, faces)
        data = np.concatenate((tnormals, np.reshape(mesh, [nFaces,9])),
                              axis=1).astype(np.float32)

        if binary:
            with open(fname, "wb") as f:
                f.write(np.zeros(80, dtype=np.uint8))
                f.write(np.uint32(nFaces))
                f.write(np.concatenate((data.astype(np.float32,order="C",copy=False).view(np.uint16),np.zeros((data.shape[0],1),dtype=np.uint16)),axis=1).reshape(-1).tobytes())
        else:
            with open(fname, "w") as f:
                f.write("solid MESH\n")
                for t in range(len(data)):
                    f.write(" facet normal {0} {1} {2}\n  outer loop\n   vertex {3} {4} {5}\n   vertex {6} {7} {8}\n   vertex {9} {10} {11}\n  endloop\n endfacet\n"\
                    .format(*data[t,:]))
                f.write("endsolid MESH\n")
    else:
        raise IOError("Invalid file format. Please choose off or stl.")