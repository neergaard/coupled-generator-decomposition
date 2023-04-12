
import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
from pathlib import Path
import scipy.sparse
from scipy.spatial.ckdtree import cKDTree



def cart_to_sph(points):
    """

    physics/ISO convention

    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    points : x, y, z in columns

    RETURNS

    (r, theta, phi)
    """
    points = np.atleast_2d(points)
    r = np.linalg.norm(points, axis=1)
    # arctan2 chooses the correct quadrant
    theta = np.arccos(points[:, 2] / r) # polar angle
    phi = np.arctan2(points[:, 1], points[:, 0]) # azimuth angle
    return np.squeeze(np.stack([r, theta, phi], axis=1))

def sph_to_cart(points):
    """
    points : r, theta, phi in columns
    """
    points = np.atleast_2d(points)
    r = points[:, 0]
    theta = points[:, 1]
    phi = points[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.squeeze(np.stack([x,y,z], axis=1))




def make_adjacency_matrix(tris):
    """Make sparse adjacency matrix for vertices with connections f.
    """
    N = tris.max() + 1

    row_ind = np.concatenate((tris[:,0], tris[:,0], tris[:,1], tris[:,1],
                              tris[:,2], tris[:,2]))
    col_ind = np.concatenate((tris[:,1], tris[:,2], tris[:,0], tris[:,2],
                              tris[:,0], tris[:,1]))

    A = scipy.sparse.csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)),
                                shape=(N,N))
    A[A > 0] = 1

    return A

def equalize_on_central_surf(central_surf, sphere_surf, sphere_surf_sub):

    new_rr = sphere_surf_sub['rr']
    tree = cKDTree(sphere_surf['rr'])
    tris = sphere_surf['tris']
    n_points = tree.n

    #new_rr = sub['lh', 'sphere.reg']['rr'].copy()
    #tree = cKDTree(pd.points)
    #tris = pd.faces.reshape(-1, 4)[:, 1:]

    # Convert area from cell attribute to point attribute and smooth
    # -----------------------------------------------------------------------------


    A = make_adjacency_matrix(tris)

    # Resample to points

    #from_pt_tris = mne.surface._triangle_neighbors(tris, pd.n_points)
    from_pt_tris = mne.surface._triangle_neighbors(tris, n_points)
    n_neighbors = [len(i) for i in from_pt_tris]
    n_neighbors_arr = np.array(n_neighbors)
    idx = np.cumsum([0] + n_neighbors)[:-1]
    from_pt_tris = np.concatenate(from_pt_tris)

    # could be weighted by triangle area somehow..?

    _, central_area = triangle_normals(**central_surf, return_area=True)
    _, sphere_area = triangle_normals(**sphere_surf, return_area=True)

    sphere_area_rescaled = sphere_area / sphere_area.std() * central_area.std()

    x = central_area / sphere_area_rescaled
    #x = pd['ratio_areas'].copy()
    x = np.add.reduceat(x[from_pt_tris], idx)
    x /= n_neighbors_arr
    #x = np.log(x)
    # Rescale to [0, 1]
    #x = (x - x.min()) / (x.max() - x.min())

    # scipy.sparse.triu(A) to get upper triangle (for edges)

    n_smooth = 100
    for i in range(n_smooth):
        x = np.add.reduceat(x[A.indices], A.indptr[:-1])
        x /= n_neighbors_arr
    #    if (i+1) % 2 == 0:
    #        pd[f'smooth{i+1}'] = x
    # to make the gradients less extreme
    #x = np.log(x) # or sqrt
    pd['smooth'] = x

    #pd[f'smooth{i+1}'] = np.sqrt(pd[f'smooth{i+1}'])

    mesh_g = pd.compute_derivative(scalars="smooth")
    grad = mesh_g['gradient']
    # project onto plane orthogonal to normal
    #grad - np.sum(grad * pd.point_normals, 1)[:, None] * pd.point_normals

    norm_grad = np.linalg.norm(grad, axis=1)
    sqrt_norm_grad = np.sqrt(norm_grad)

    grad_scaled = grad / sqrt_norm_grad[:, None]
    #grad_scaled[grad_scaled > 1] = 1.5

    # Update/move the points
    # -----------------------------------------------------------------------------

    A = make_adjacency_matrix(sub['lh', 'sphere.reg']['tris'])
    edges = np.stack(A.nonzero())
    w = A.getnnz(1)
    edgeidx = np.concatenate((np.array([0]), w.cumsum()[:-1]))



    # Update gradient
    # 1) use gradient of nearest neighbor on full resolution mesh.
    # 2) project to triangle and use weighted sum of its nodes' gradients (linear
    # interp.)

    # Laplacian smooth
    b = 1e-3
    area_summary = []
    prev_change = np.inf
    for q in range(1,101):

        # new position due to laplacian smoothing
        x = np.add.reduceat(new_rr[edges[1]], edgeidx, axis=0)
        x /= w[:, None]

        # Estimate gradient at new_rr by linear interpolation using three nearest
        # neighbors
        #_, i = tree.query(new_rr)
        d, i = tree.query(new_rr, 3)
        invd = 1/d
        affinity = np.nan_to_num(invd / np.sum(invd, 1)[:, None], nan=1)

        # el = np.linalg.norm(new_rr[edges[1]]-new_rr[edges[0]], axis=1)
        # minel = np.minimum.reduceat(el, edgeidx, axis=0)
        # maxmove = 0.25 * minel

        # new position = laplacian smooth + b * grad
        prev_rr = new_rr.copy()

        #new_rr = x + b * grad_scaled[i]
        new_rr = x + b * np.sum(grad_scaled[i] * affinity[..., None], axis=1)

        # move = new_rr-prev_rr
        # actualmove = np.linalg.norm(move, axis=1)
        # new_rr = prev_rr + move / actualmove[:, None] * np.minimum(actualmove, maxmove)[:, None]

        new_rr /= np.linalg.norm(new_rr, axis=1, keepdims=True)

        # Change in node positions
        change = np.linalg.norm(new_rr-prev_rr)
        rel_change = (change - prev_change) / prev_change
        prev_change = change
        print('{:3d} : {:0.4f} %'.format(q, rel_change * 100))
        # break when relative change is less than 1 %

        if q % 10 == 0:
            pd1 = pv.PolyData(new_rr, cells_to_vtk(sub['lh', 'sphere.reg']['tris']))
            pd1.save(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\simnibs\sub-01\m2m_sub-01\surfaces\t' + f'{q}.vtk')

        pd2 = pv.PolyData(full['lh', 'central']['rr'][i[:, 0]], cells_to_vtk(sub['lh', 'sphere.reg']['tris']))
        area = pd2.compute_cell_sizes(length=False, volume=False)['Area']
        area_summary.append((area.min(), area.max(), area.mean(), area.std()))

    # Redo convex hull on final point cloud

    a = np.array(area_summary)

pd2 = pv.PolyData(sub['lh', 'central']['rr'], cells_to_vtk(sub['lh', 'sphere.reg']['tris']))
pd2.save(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\simnibs\sub-01\m2m_sub-01\surfaces\central0.vtk')

pd2 = pv.PolyData(full['lh', 'central']['rr'][i[:,0]], cells_to_vtk(sub['lh', 'sphere.reg']['tris']))
pd2.save(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\simnibs\sub-01\m2m_sub-01\surfaces\central' + f'{q}.vtk')

pd2 = pv.PolyData(full['lh', 'central']['rr'], cells_to_vtk(full['lh', 'central']['tris']))
pd2['arearatio'] = central_area / sphere_area_rescaled
pd2.save(r'C:\Users\jdue\Documents\phd_data\analysis\ds000117\simnibs\sub-01\m2m_sub-01\surfaces\centralfull.vtk')


