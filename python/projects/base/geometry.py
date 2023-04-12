"""
Utilities for geometric computations.
"""

import itertools
import numpy as np
import scipy.sparse as ss


def get_adjacency_matrix(tris, verts): # , verts=None
    """Make sparse adjacency matrix for vertices with connections `tris`.
    """
    N = tris.max() + 1

    pairs = list(itertools.combinations(np.arange(tris.shape[1]), 2))
    row_ind = np.concatenate([tris[:, i] for p in pairs for i in p])
    col_ind = np.concatenate([tris[:, i] for p in pairs for i in p[::-1]])

    # if verts is not None:
    # else:
    data = np.linalg.norm(verts[row_ind]-verts[col_ind], axis=1)
    # data = np.ones_like(row_ind)
    A = ss.csr_array((data / 2, (row_ind, col_ind)), shape=(N, N))

    return A


def get_graph_laplacian(tris):
    A = get_adjacency_matrix(tris)
    D = get_degree_matrix(A)
    return D - A


def get_degree_matrix(A):
    """A : adjacency matrix"""
    data = np.array(A.sum(1)).squeeze()
    offset = 0
    return ss.dia_matrix((data, offset), shape=A.shape)


def get_adjacency_matrix_of_elements(e):
    """Return the adjacency matrix of the elements.

    Useful for example for finding connected components by element faces, e.g.,

        n_comps, labels = ss.csgraph.connected_components(A)

    e : array describing the elements, e.g., triangles or tetrahedra

    """
    ne, edim = e.shape
    v_per_face = edim - 1

    # Make the edges (triangles) / faces (tetrahedra)
    sf = np.sort(e, axis=1)
    # combinations: e.g., (0,1), (0,2), (1,2)
    c = [i for c in itertools.combinations(np.arange(edim), v_per_face) for i in c]

    # an edge consists of two (triangles) or three (tetrahedra)
    dt = np.dtype([("", e.dtype)] * v_per_face)

    edges = sf[:, c].ravel().view(dt)

    # argsort to bring like edges (and thus connected elemenets) next to each
    # other, thus 0,1 are connected, 2,3 etc.

    sedges = np.argsort(edges)

    # convert edge indices to element indices
    # each element has 'edim' edges/faces and this consists of 'v_per_face' vertices
    # idx : each row has indices of two elements that are connected
    idx = np.floor(sedges / edim).astype(np.int).reshape(-1, v_per_face)
    # to make A symmetric
    idx = np.vstack((idx, idx[:, ::-1]))

    data = np.ones(len(idx), dtype=np.int)

    # adjacency matrix of elements
    A = ss.coo_matrix((data, (idx[:, 0], idx[:, 1])), shape=(ne, ne))

    return A


def triangle_normal(vertices, faces):
    """Get normal vectors for each triangle in the mesh.

    PARAMETERS
    ----------
    mesh : ndarray
        Array describing the surface mesh. The dimension are:
        [# of triangles] x [vertices (of triangle)] x [coordinates (of vertices)].

    RETURNS
    ----------
    tnormals : ndarray
        Normal vectors of each triangle in "mesh".
    """
    mesh = vertices[faces]

    tnormals = np.cross(
        mesh[:, 1, :] - mesh[:, 0, :], mesh[:, 2, :] - mesh[:, 0, :]
    ).astype(np.float)
    tnormals /= np.sqrt(np.sum(tnormals ** 2, 1))[:, np.newaxis]

    return tnormals


def vertex_normal(vertices, faces):
    """

    """
    face_normals = triangle_normal(vertices, faces)

    out = np.zeros_like(vertices)
    for i in range(len(faces)):
        out[faces[i]] += face_normals[i]
    out /= np.linalg.norm(out, ord=2, axis=1)[:, None]

    return out

