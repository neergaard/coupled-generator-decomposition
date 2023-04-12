"""

Tools for bridging the gap between SimNIBS and the world of EEG.


* Prepare an EEG layout file in SimNIBS format. This can be used when running a
forward simulation.
* Run a forward simulation computing the leadfield.
* Convert the leadfield to MNE or FieldTrip format.

"""
import h5py
import mne
import numpy as np
from pathlib import Path
import warnings

try:
    from simnibs.simulation import sim_struct
    # from simnibs.simulation import cond
except ModuleNotFoundError:
    warnings.warn('Failed to import SimNIBS. Some functions may be broken.')

def prepare_eeg_layout(info, trans, simnibs_organizer):
    """Create a cap file for simnibs to use for leadfield simulation.

    Read electrode positions in head coordinates
    Transform electrode positions to MRI coordinates
    Write the layout file

    """
    sub = simnibs_organizer.get_path('subject')
    filename = sub / 'electrode_layout.csv'
    _prepare_eeg_layout(info, trans, filename)

def _prepare_eeg_layout(info, trans, filename):

    trans = mne.transforms._ensure_trans(trans, 'head', 'mri')

    electrodes = {}
    for ch in info['chs']:
        # to MRI coordinates
        # to mm
        electrodes[ch['ch_name']] = mne.transforms.apply_trans(trans, ch['loc'][:3]) * 1e3

    # There should be no spaces after ','!
    with open(filename, 'w') as f:
        f.write('Type,X,Y,Z,ElectrodeName\n')
        for ch_name, ch_pos in electrodes.items():
            ch_pos = ch_pos
            line = ','.join(['Electrode', *map(str, ch_pos), ch_name]) + '\n'
            f.write(line)

def make_forward(simnibs_organizer):
    """
    Save simulation in simnibs_root / subject / fem_subject
    """

    m2m = simnibs_organizer.get_path('m2m')
    fem = simnibs_organizer.get_path('fem')
    if not fem.exists():
        fem.mkdir()
    mesh = simnibs_organizer.match('*.msh')
    lh = simnibs_organizer.match('surfaces/lh.central.*.gii')
    rh = simnibs_organizer.match('surfaces/rh.central.*.gii')
    sources = (lh, rh)
    eeg_layout = simnibs_organizer.match('electrode_layout.csv', 'subject')

    return _make_forward(m2m, fem, mesh, sources, eeg_layout)

def _make_forward(m2m, fem, mesh, sources, eeg_layout):

    # Use standard conductivities
    # The paths should be strings otherwise errors might occur when writing the
    # .hdf5 file
    lf = sim_struct.TDCSLEADFIELD()
    lf.fnamehead = str(mesh)
    lf.subpath = str(m2m)
    lf.pathfem = str(fem)
    lf.field = 'E'
    lf.eeg_cap = str(eeg_layout)
    # We do not want field estimates in the eyes (tag = 1006)
    lf.tissues = []
    lf.interpolation = tuple(str(s) for s in sources)

    # Custom conductivities
    # s = cond.standard_cond()
    # s[1].value = 0.3   # GM (brain)
    # s[3].value = 0.006 # bone
    # s[4].value = 0.3   # skin
    # lf.cond = s

    # Which tissues to interpolate from
    # lf.interpolation_tissue = [2] # default = 2

    lf.run(save_mat=False, cpus=1)

    return fem / lf._lf_name()

def convert_forward(output_format, forward, info=None, trans=None):
    """
    If the solution was interpolated to two surfaces it is assumed that the first
    one corresponds to the left hemisphere and the second one to the right
    hemisphere. This only really applies when exporting to MNE-python, however,
    when exporting to FieldTrip information on left and right hemispheres will be
    included as well.

    Read the forward solution
    Convert to 'EEG mode'.
    Rereference to an average reference.

    If output_format == 'mne'
        convert solution to head coordinate system.
    If output_format == 'fieldtrip'
        No additional processing


    output_format : str
        mne or fieldtrip
    forward : str
        Filename
    info : str | instance of mne.Info
        Filename
    trans : str | instance of mne.Transform
        Filename
    """
    forward = Path(forward)
    if output_format == 'mne':
        assert info is not None, "info must be supplied when output_format = 'mne'!"
        assert trans is not None, "trans must be supplied when output_format = 'mne'!"

    print('Reading leadfield...', end=' ', flush=True)
    with h5py.File(forward, "r") as f:
        fwd = f['mesh_leadfield']['leadfields']['tdcs_leadfield']

        # Channel info
        ch_names = np.array([ch.decode('UTF-8') for ch in fwd.attrs['electrode_names']])
        ch_ref = fwd.attrs['reference_electrode']

        # Source space info
        pts = f['mesh_leadfield']['nodes']['node_coord'][:]
        elm = f['mesh_leadfield']['elm']['node_number_list'][:] - 1 # for python indexing

        # check if triangles
        is_tri = np.any(f['mesh_leadfield']['elm']['elm_type'][:] == 2)
        if is_tri:
            # Mixing triangles and tetrahedra is not allowed so this is safe
            elm = elm[:, :3]

        is_hemispheres = False
        interpolated = True if fwd.attrs['interpolation'] == 'True' else False
        if interpolated:
            ntags = len(np.unique(f['mesh_leadfield']['elm']['tag1']))
            nids = len(f['mesh_leadfield'].attrs['interp_id'])
            if ntags == nids == 2:
                is_hemispheres = True
                nix = f['mesh_leadfield'].attrs['node_ix']
                eix = f['mesh_leadfield'].attrs['elm_ix']

        # Get the leadfield and invert it when going from TES to 'EEG mode'
        fwd = -fwd[:]

    print('Done')

    nchan, nsrc, nori = fwd.shape # Leadfield is always calculated in x, y, z

    print('Rereferencing leadfield to an average reference')
    fwd = np.insert(fwd, np.where(ch_names == ch_ref)[0],
                   np.zeros((1, *fwd.shape[1:])), axis=0)
    fwd -= fwd.mean(0)
    nchan += 1

    print('Getting source positions')
    # if interpolated then the solution is per vertex otherwise it is per element
    if is_hemispheres:
        print('- Leadfield was interpolated to two surfaces. Assuming left and right hemispheres (in that order)')
        # Solution was interpolated to two surfaces and no tissue surfaces.
        # Hence, assume the that the surfaces interpolated to correspond to left
        # and right hemispheres, respectively (and in that order)
        # This is the order in which it will be if interpolation='middle gm'
        # was used. (Also, MNE-Python seems to set up source spaces with src[0]
        # being lh.)
        lh_pos, rh_pos = np.vsplit(pts, [nix[1]])
        lh_tri, rh_tri = np.vsplit(elm, [eix[1]])
        rh_tri -= nix[1] # fix indexing
        pos = np.vstack((lh_pos, rh_pos))
    # Create one source space
    elif interpolated:
        pos = pts
    else:
        # Use barycenters of tetrahedra as source positions
        pos = pts[elm].mean(1)

    if output_format == 'mne':
        """
        Bad channels should *not* be included in the leadfield simulations.
        Should any such channels (as marked in the Info structure) be found
        in the leadfield a warning will be issued.
        """
        import mne
        mne.set_log_level('warning')

        from projects.base.sourcespace import make_sourcespace

        print('Exporting to MNE format')

        # instance of Raw, Epochs, Evoked containing Info
        if not isinstance(info, mne.Info):
            info = mne.io.read_info(info)
        # Pick only the EEG channels
        #info.pick_channels([info['ch_names'][i] for i in  mne.pick_types(info, eeg=True)])

        bads_in_ch_names = np.any(np.isin(ch_names, info['bads']))
        if np.any(bads_in_ch_names):
            import warnings
            warnings.warn('Bad channels from Info found in leadfield: {}'.format(ch_names[bads_in_ch_names]))

        # MRI -> HEAD transformation, "-trans.fif"
        if not isinstance(trans, mne.Transform):
            trans = mne.read_trans(trans)
        trans = mne.transforms._ensure_trans(trans, 'mri', 'head')

        if is_hemispheres:
            src_lh = make_sourcespace(lh_pos, lh_tri, surf_id='lh')
            src_rh = make_sourcespace(rh_pos, rh_tri, surf_id='rh')
            src = mne.SourceSpaces([src_lh, src_rh])
        else:
            src = make_sourcespace(pos)

        kwargs = dict(
            src=src,
            mri_head_t=trans,
            info=info,
            bem=None,
            mindist=0,
            n_jobs=1,
            meg=False,
            ignore_ref=True,
            allow_bem_none=True
        )
        _, _, _, _, _, eegnames, _, info, update_kwargs, _ \
            = mne.forward._make_forward._prepare_for_forward(**kwargs)

        #cmd = 'python export_leadfield.py mne {} {} {}'.format(forward, info, trans)
        cmd = ''
        info['command_line'] = cmd
        update_kwargs['info']['command_line'] = cmd

        # Check that we have the same channels in data and leadfield
        assert np.all(eegnames == ch_names), 'Inconsistencies between channels in Info and leadfield'

        print('Transforming leadfields from MRI to head coordinate system')
        """
        A note on forward solutions and coordinate systems.

        Below we use

            Q[mri]      : dipole moments per position in MRI coordinates
            Q[head]     : dipole moments per position in HEAD coordinates
            I           : 3x3 identity matrix
            mri_head_t  : 3x3 transformation from MRI to HEAD coordinates
            head_mri_t  : 3x3 transformation from HEAD to MRI coordinates
            (i.e., without translation)

        MNE forward solutions are in head coordinates, however, they are
        actually computed in MRI coordinates by transforming electrodes to MRI
        coords and, instead of computing the solution along the x/y/z axes such
        that Q[mri] = I, they use Q[mri] = head_mri_t.T as the dipole moments
        (see forward/_compute_forward/_prep_field_computation).
        In this way, the solution corresponds to dipole moments Q[head] = I
        (i.e. along x/y/z in head coordinate system).

        SimNIBS forward solutions are in the coordinate system of the volume
        conductor model, which, in general, will be MRI coordinates since this
        is the space of the original MRI image(s). Solutions are computed in
        this coordinate system, i.e., using Q[mri] = I. Thus, we need to
        transform the SimNIBS solution from Q[mri] = I to Q[head] = I.

        Since

            head_mri_t.T @ mri_head_t.T = I

        we can convert a forward solution with Q[mri] = I to one with
        Q[head] = I like

            Q[head] = Q[mri] @ mri_head_t.T
        """

        # Convert forward solution from MRI coordinates to head coordinates
        # fwd = mne.transforms.apply_trans(trans, fwd, move=False)
        fwd = fwd @ trans['trans'][:3, :3].T
        fwd = fwd.reshape((nchan, nsrc*nori))
        # leadfield should be shape (3 * n_dipoles, n_sensors) so transpose
        fwd = mne.forward._make_forward._to_forward_dict(fwd.T, eegnames)
        fwd.update(**update_kwargs)

        filename = forward.parent / f'{forward.stem}-fwd.fif'
        mne.write_forward_solution(filename, fwd, overwrite=True)

    elif output_format == 'fieldtrip':
        from scipy.io import savemat

        print('Exporting to FieldTrip format')

        # Create a cell array of matrices by filling a numpy object
        # Each cell is n_channels by n_orientations
        fwd_cell = np.empty(nsrc, dtype=np.object)
        for i in range(nsrc):
            fwd_cell[i] = fwd[:, i]

        # The forward structure of FieldTrip
        fwd = {}
        fwd["pos"] = pos
        fwd["inside"] = np.ones(nsrc, dtype=np.bool)[:, None] # all sources are valid
        fwd["unit"] = "mm"
        fwd["leadfield"] = fwd_cell
        fwd["label"] = ch_names.astype(np.object)[:, None]
        fwd["leadfielddimord"] = r"{pos}_chan_ori"
        #fwd["cfg"] = "Generated by SimNIBS {}".format(simnibs_version)

        # The leadfield of a particular electrode may be plotted in FieldTrip
        # like so
        #
        # fwd_mat = cell2mat(fwd.leadfield);
        # eix = 10; % electrode
        # ori = 2;  % orientation (1,2,3 corresponding to x,y,z)
        # ft_plot_mesh(fwd, 'vertexcolor', fwd_mat(eix, ori:3:end)');
        #
        # fwd_info includes information about which positions and elements
        # correspond to left and right hemisphere, respectively, as well as
        # the vertex normals for projecting the leadfield onto these if
        # desired.

        variables = dict(fwd=fwd)
        if is_hemispheres:
            # Modify indices of rh_tri so that fwd.pos(fwd.tri, :) will plot
            # both hemispheres
            fwd['tri'] = np.vstack((lh_tri, rh_tri+nix[1]))+1 # matlab indexing

            # Add some info to the .mat file
            variables['fwd_info'] = dict(
                vertices_lh = [1, nix[1]],
                vertices_rh = [nix[1]+1, len(pts)],
                tris_lh = [1, eix[1]],
                tris_rh = [eix[1]+1, len(elm)]
            )

            lh_normals = compute_vertex_normals(lh_pos, lh_tri)
            rh_normals = compute_vertex_normals(rh_pos, rh_tri)
            variables['fwd_info']['vertex_normals'] = np.vstack((lh_normals,
                                                                 rh_normals))
        filename = forward.parent / f'{forward.stem}-fwd.mat'
        savemat(filename, variables)

    else:
        raise ValueError('Output format must be "mne" or "fieldtrip"')

    print('Done')

    return filename

def compute_vertex_normals(v, f):

    # Triangle normals
    n = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    n /= np.linalg.norm(n, ord=2, axis=1)[:, None]

    # Each triangle that a vertex is a part of contributes to the normal of
    # this vertex
    out = np.zeros_like(v)
    for i in range(len(f)):
        out[f[i]] += n[i]
    out /= np.linalg.norm(out, ord=2, axis=1)[:, None]

    return out