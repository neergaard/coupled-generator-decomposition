import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np

def sourcespace_from_charm(simnibs_organizer, file_organizer, n_vertices=None):

    if n_vertices is not None:
        n_triangles = 2 * 10000 - 4
    else:
        n_triangles = None

    # Full resolution surfaces
    lh = simnibs_organizer.get_path('m2m') / 'surfaces' / 'lh.central.gii'
    rh = simnibs_organizer.get_path('m2m') / 'surfaces' / 'rh.central.gii'

    # lh first
    src = []
    for f in (lh, rh):

        src_, hemi = sourcespace_from_gii(f, n_triangles)
        src.append(src_)

        filename = f.stem + '.' + str(n_vertices) + f.suffix
        hemi.to_filename(f.parent / filename)

    src = mne.SourceSpaces(src)

    filename = file_organizer.get_filename(suffix='src', extension='fif')
    src.save(filename, overwrite=True)

    return src

def sourcespace_from_gii(filename, n_triangles):

    hemi = nib.load(filename)
    pts, tris = hemi.agg_data()
    
    if n_triangles is not None:
        #pts = hemi.agg_data('pointset')
        #tris = hemi.agg_data('triangle')
        pts, tris = mne.surface.decimate_surface(pts, tris, n_triangles)
        pts = np.asarray(pts, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.int32)
        hemi = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(pts, 'NIFTI_INTENT_POINTSET'),
                                       nib.gifti.gifti.GiftiDataArray(tris, 'NIFTI_INTENT_TRIANGLE')))

    if filename.stem.startswith('lh'):
        surf_id = 'lh'
    elif filename.stem.startswith('rh'):
        surf_id = 'rh'
    else:
        raise ValueError

    src = make_sourcespace(pts, tris, coord_frame='mri', surf_id=surf_id)

    return src, hemi

def make_sourcespace(pos, tris=None, coord_frame='mri', surf_id=None):
    """Setup a discrete MNE source space object (as this is more flexible
    than the surface source space).

    pos : ndarray (n, 3)
        Source positions
    tris : ndarray (m, 3)
        If source positions are vertices of a surface, this defines the
        surface.
    coord_frame : str
        mri or head
    sid : str
        surface id. lh, rh, or None.
    """

    # mm -> m (input assumed to be in mm)
    pos = pos * 1e-3 # To avoid inplace modification
    npos = len(pos)

    # Source normals
    if tris is None:
        # Define an arbitrary direction
        nn = np.zeros((npos, 3))
        nn[:, 2] = 1.0
    else:
        nn = None # Calculate later

    if coord_frame == 'mri':
        coord_frame = FIFF.FIFFV_COORD_MRI
    elif coord_frame == 'head':
        coord_frame = FIFF.FIFFV_COORD_HEAD
    else:
        raise ValueError('coord_frame must be mri or head')

    assert surf_id in ('lh', 'rh', None)
    if surf_id == 'lh':
        surf_id = FIFF.FIFFV_MNE_SURF_LEFT_HEMI
    elif surf_id == 'rh':
        surf_id = FIFF.FIFFV_MNE_SURF_RIGHT_HEMI
    elif surf_id is None:
        surf_id = FIFF.FIFFV_MNE_SURF_UNKNOWN

    # Assumed to be in mm, thus mm -> m
    #pos = dict(
    #    rr = pos * 1e-3,
    #    nn = source_normals * 1e-3
    #    )
    #src = mne.setup_volume_source_space(subject=None, pos=pos, verbose=False)

    src = dict(
        id = surf_id,
        type = 'discrete',
        np = npos,
        ntri = 0,
        coord_frame = coord_frame,
        rr = pos,
        nn = nn,
        tris = None,
        nuse = npos,
        inuse = np.ones(npos, dtype=np.int),
        vertno = np.arange(npos),
        nuse_tri = 0,
        use_tris = None
        )

    # Unused stuff
    src.update(dict(
        nearest = None,
        nearest_dist = None,
        pinfo = None,
        patch_inds = None,
        dist = None,
        dist_limit = None,
        subject_his_id = None
        ))

    if tris is not None:
        # Setup as surface source space
        # MNE doesn't like surface source spaces that are not LH or RH
        assert src['id'] in (FIFF.FIFFV_MNE_SURF_LEFT_HEMI,
                             FIFF.FIFFV_MNE_SURF_RIGHT_HEMI)
        surf = dict(rr=pos, tris=tris)
        surf = mne.surface.complete_surface_info(surf)

        src['type'] = 'surf'
        src['tris'] = surf['tris']
        src['ntri'] = surf['ntri']
        src['nn'] = surf['nn'] # vertex normals

        # we use all tris, so the following is not really used
        src['use_tris'] = None # else [nuse_tri x 3] of indices into src['tris']
        src['nuse_tri'] = 0    # else len(src['use_tris'])

    mne.source_space._complete_source_space_info(src)

    return src #mne.source_space.SourceSpaces(src)


# Source Space Reduction / Projection on Basis Functions

def prepare_basis_functions(fwd, surface_lh, surface_rh, sigma=0.6, min_dist=3, exp_degree=8, random_seed=0,
                            write=True, symmetric_sources=False):
    """

    sigma :
        the 'amount' of activity to propagate
    min_dist :
        minimum distance between basis functions (counted in mesh edges)
    exp_degree :
        number of edge expansions to do for each basis function...
    random_seed :
        For reproducible results.
    write :
        Write
    symmetric_sources : bool


    """

    # convert to surface normal
    #fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    v_lh, f_lh = io_misc.read_surface(surface_lh)
    v_rh, f_rh = io_misc.read_surface(surface_rh)

    print('Selecting vertices')
    centers_lh = select_vertices(f_lh, min_dist, random_seed)
    if symmetric_sources:
        centers_rh = match_opposite_hemisphere(centers_lh, surface_lh, surface_rh)
    else:
        centers_rh = select_vertices(f_rh, min_dist, random_seed)

    B_lh = make_basis_functions(surface_lh, centers_lh, sigma, exp_degree, write)
    B_rh = make_basis_functions(surface_rh, centers_rh, sigma, exp_degree, write)
    B = [B_lh, B_rh]

    if symmetric_sources:
        #B_bi = ss.vstack([B_lh, B_rh])
        B.append(ss.vstack([B_lh, B_rh]))

    # Support over vertices
    #plt.figure(); plt.hist(B_lh.sum(1),100); plt.show()

    # Number of basis functions overlapping with each vertice
    #plt.figure(); plt.hist(B_lh.getnnz(1),100); plt.show()

    D = project_forward_to_basis(fwd, B, make_bilateral=symmetric_sources)

    return D, B

def project_forward_to_basis(fwd, basis, make_bilateral=False):
    """Project forward solution from a source space onto a set of spatial basis
    functions. Implements

        D = AB

    where A [K x N] is the gain matrix, B [N x C] is the spatial basis
    functions, and D [K x C] is the gain matrix projected onto B.

    A may be [K x 3N] so B may need to be expanded to accomodate this.

    Here K sensors, N source locations, and C basis function.

    basis : list | scipy sparse matrix
        List of basis sets (each containing a number of basis functions).
        Should match the number of source spaces in 'fwd'.
    make_bilateral : bool
        Make a summed forward projection in addition to those contained in
        'basis'.

    """

    if isinstance(fwd, str):
        fwd = mne.read_forward_solution(fwd)
    if not isinstance(basis, list):
        basis = [basis]
    nbases = len(basis)
    nchs = len(fwd['info']['chs'])

    # Determine number of sources per location
    if fwd['source_ori'] is FIFF.FIFFV_MNE_FIXED_ORI:
        spl = 1
    elif fwd['source_ori'] is FIFF.FIFFV_MNE_FREE_ORI:
        spl = 3

    # Make the iteration items of source space lengths and basis functions
    src_sizes = [s['np'] for s in fwd['src']]
    if nbases == len(fwd['src']):
        iter_items = zip(src_sizes, basis)
    elif nbases == 1 and basis[0].shape[0]*spl == fwd['sol']['data'].shape[1]:
        iter_items = zip( [sum(src_sizes)]  , basis )
    else:
        raise ValueError('Basis functions do not seem to match forward solution.')

    print('Projecting forward solution')
    gains = list()
    initcols = slice(0,0)
    for s,b in iter_items:
        gain = np.zeros((nchs, b.shape[1]*spl))
        for i in range(spl):
            cols = slice(i+initcols.start, initcols.stop+s*spl, spl)
            gain[:,i::spl] = ss.csc_matrix.dot(fwd['sol']['data'][:,cols], b)
        gains.append(gain)
        initcols = slice(cols.stop, cols.stop)

    if make_bilateral:
        gains.append( sum(gains) )

    gains = np.concatenate(gains, axis=1)
    #fwd['sol']['data'] = gains
    #fwd['nsource'] =

    return gains


def project_back_sourceestimate(invsol, basis):
    """Project an inverse solution from source space back to sensor space.

    invsol :
        Inverse solution array of N sources by K time points.
    basis : list
        List of sparse arrays defininf the basis functions.
    scale :
        Scaling vector describing the sum of the original forward solution per
        basis function (which was normalized to one). Now that we are
        projecting back, we want to get back to the same scaling.


    """

    # 4000 x 1 or 4000 x time...
    # invsol
    #
    if not isinstance(basis, list):
        basis = [basis]

    invsol = np.atleast_2d(invsol)

    print('Projecting source estimates to original sources space')

    # inverse solution in nano ampere?
    nA = 1#1e9

    # rows in inverse solution
    nsol = invsol.shape[0]
    # n basis functions
    nsrc = sum([b.shape[1] for b in basis])

    spl = nsol//nsrc
    ninvsol = list()
    initcols = slice(0,0)
    for b in basis:
        isol = np.zeros((b.shape[0]*spl, invsol.shape[1]))
        for i in range(spl):
            cols = slice(i+initcols.start, initcols.stop+b.shape[1]*spl, spl)
            isol[i::spl,:] = b.dot(invsol[cols]) * nA
        ninvsol.append(isol)
        initcols = slice(cols.stop, cols.stop)
    #ninvsol = np.concatenate(ninvsol,axis=0)

    return ninvsol

def make_basis_functions(f, centers, sigma=0.6, degree=8, write=False):
    """

    """

    if isinstance(f, str):
        base, _ = op.splitext(f)
        v, f = io_misc.read_surface(f)
    else:
        write = False
        warn('Cannot write if surface is not a filename.')

    assert degree >= 1

    A = make_adjacency_matrix(f)
    A = A*sigma

    # Add diagonal
    A += ss.eye(*A.shape)

    # Keep only the basis functions we wish to use
    B = A[centers]
    for i in range(2,degree+1):
        B += B.dot(A)/i

    # Normalize basis functions
    B = B.multiply(1/B.sum(1))
    B = B.T.tocsc()

    # Threshold
    B = B.multiply(B>np.exp(-8))

    # Calculate some statistics on the support of each source
    support = B.getnnz(1)
    print('Source Support (# bases)')
    print('Min  : {:d}'.format(support.min()))
    print('Mean : {:.2f}'.format(support.mean()))
    print('Max  : {:d}'.format(support.max()))

    support = B.sum(1)
    print('Source Support (Weight)')
    print('Min  : {:.3f}'.format(support.min()))
    print('Mean : {:.3f}'.format(support.mean()))
    print('Max  : {:.3f}'.format(support.max()))

    if write:
        print('Writing to disk...', end=' ')

        origin = np.zeros(B.shape[0])
        origin[centers] = 1
        basis = np.asarray(B.sum(1)).squeeze()
        vtk = io_misc.as_vtk(v, f, pointdata=dict(origin=origin, basis=basis))

        vtk.tofile(base+'_basis', 'binary')
        print('Done')

    return B

def match_opposite_hemisphere(targets, src, dst):
    """Find the vertices of 'dst' closest to the 'target' vertices in 'src'.

    """
    if isinstance(src, str):
        sv, sf = io_misc.read_surface(src)
    if isinstance(dst, str):
        dv, df = io_misc.read_surface(dst)

    # Get offset from x=0 of src
    sx_offset = sv[:,0].max(0)
    sv[:,0] -= sx_offset

    # Get the vertices in src and mirror the coordinate around x=0
    stargets = sv[targets]
    stargets[:,0] = -stargets[:,0]

    # Get offset from x=0 of dst
    dx_offset = dv[:,0].min(0)
    dv[:,0] -= dx_offset

    # Find the closes match in dst
    tree = cKDTree(dv)
    d, dtargets = tree.query(np.atleast_2d(stargets))

    print('Targets found in destination')
    print('Distance (min)  : {:.2f}'.format(d.min()))
    print('Distance (mean) : {:.2f}'.format(d.mean()))
    print('Distance (max)  : {:.2f}'.format(d.max()))

    return dtargets

def make_adjacency_matrix(f):
    """Make sparse adjacency matrix for vertices with connections f.
    """
    N = f.max()+1

    row_ind = np.concatenate((f[:,0],f[:,0],f[:,1],f[:,1],f[:,2],f[:,2]))
    col_ind = np.concatenate((f[:,1],f[:,2],f[:,0],f[:,2],f[:,0],f[:,1]))

    #row_ind = np.concatenate((f[:,0],f[:,0],f[:,1]))
    #col_ind = np.concatenate((f[:,1],f[:,2],f[:,2]))

    A = ss.csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=(N,N))
    A[A>0] = 1

    return A

def select_vertices(f, min_dist=3, random_seed=None):
    """Select vertices from 'f' that are a minimum of 'min_dist' from each
    other.
    """
    A = make_adjacency_matrix(f)
    idx = reshape_sparse_indices(A)

    # Ensure that (1) A includes diagonal, (2) A is binary
    if any(A.diagonal()==0):
        A += ss.eye(*A.shape)
    A = A.multiply(A>0)

    #
    if random_seed is not None:
        assert isinstance(random_seed, Integral)
        np.random.seed(random_seed)

    # vertex enumerator
    venu = np.arange(A.shape[0])

    vertices_left = np.ones_like(venu, dtype=bool)
    vertices_used = np.zeros_like(venu, dtype=bool)

    #B = recursive_dot(A, A, recursions=min_dist-1)

    while any(vertices_left):
        i = np.random.choice(venu[vertices_left])
        vertices_used[i] = True

        # Find neighbors and remove remove those
        #if min_dist > 1:
        #    B = recursive_dot(A[i], A, min_dist-1)
        #else:
        #    B = A[i]
        #vertices_left[B.indices] = False


        vertices_left[recursive_index(idx, i, min_dist)] = False

    return np.where(vertices_used)[0]

def recursive_dot(A, B, recursions=1):
    """Recursive dot product between A and B, i.e.,

        A.dot(B).dot(B) ...

    as determined by the recursions arguments. If recursions is 1, this
    corresponds to the usual dot product.
    """

    assert isinstance(recursions, Integral) and recursions > 0

    if recursions == 1:
        return A.dot(B)
    else:
        return recursive_dot(A.dot(B), B, recursions-1)

def recursive_index(indices, start, recursions=1, collapse_levels=True):
    """Recursively index into 'indices' starting from (and including) 'start'.

    """
    assert recursions >= 0

    #start = [start] if not isinstance(start, list) else start

    levels = list()
    levels.append([start])

    i = 0
    while i < recursions:
        ith_level = set()
        for j in levels[i]:
            ith_level.update(indices[j])
        ith_level.difference_update(flatten(levels)) # remove elements from rings

        levels.append(list(ith_level))
        i+=1

    return flatten(levels) if collapse_levels else levels

def reshape_sparse_indices(A):
    """Reshape indices of sparse matrix to list."""
    return [A.indices[slice(A.indptr[i],A.indptr[i+1])].tolist() for i in range(len(A.indptr)-1)]

def flatten(l):
    """Flatten list of lists.
    """
    return [item for sublist in l for item in sublist]
