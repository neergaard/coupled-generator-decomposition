import h5py
import meshio
import numpy as np
import sys

from simnibs import simulation
# from simnibs.simulation.eeg import prepare_forward

from projects.simval.config import Config
from projects.simval.sphere_utils import idx_to_density

def _make_electrodes(sensor_pos):
    elec = []
    for i,p in enumerate(sensor_pos):
        elec.append(simulation.sim_struct.ELECTRODE())
        elec[i].name = 'EEG{:03d}'.format(i)
        elec[i].channelnr = i
        elec[i].centre = p.tolist()
        elec[i].definition = 'plane'
        elec[i].shape = 'ellipse'
        elec[i].dimensions = [10,10]
        # Reducing thickness reduces potential drop across eletrode (e.g.,
        # 0.01)
        elec[i].thickness = [4] # [0.01]
    return elec

def _get_conductivities():
    # Conductivities
    cond = Config.sphere.conductivity
    S = simulation.cond.standard_cond()
    S[1].value = cond['brain'] # GM ('brain' in this case)
    S[3].value = cond['skull']
    S[4].value = cond['scalp']
    # Remove conductivity gradient between electrode and skin, hence reducing
    # effects of other electrodes which would effectively make the skin
    # compartment better conducting hence less current would penetrate into
    # the brain compartment
    # S[99].value = cond['scalp']
    # S[499].value = cond['scalp']

    return S

def run_simulation(i):
    """
    The magnitudes from simnibs are consistently SMALLER than those from the
    analytical solution. Setting the conductivity of the electrodes equal to
    that of scalp and reducing the thickness all but eliminates this.
    """
    mesh_name = 'sphere_3_layer'
    _, d = idx_to_density(i)
    fem_dir = d / f'fem_{mesh_name}'
    # fem_dir = d / f'fem_{mesh_name}_thin_electrodes'

    sensor_pos = meshio.read(Config.path.SPHERE / 'sensor_positions.stl').points
    electrodes = _make_electrodes(sensor_pos)

    src_off = Config.path.SPHERE / 'src_coords.off'

    cond = _get_conductivities()

    lf = simulation.sim_struct.TDCSLEADFIELD()
    lf.fnamehead = str(d / f'{mesh_name}.msh')
    lf.pathfem = str(fem_dir)
    lf.field = 'E'
    lf.tissues = []
    lf.cond = cond
    lf.interpolation = str(src_off),
    lf.interpolation_tissue = [2]
    lf.eeg_cap = None
    lf.electrode = electrodes
    lf.solver_options = 'pardiso'

    lf.run(save_mat=False)

    fwd = prepare_forward(fem_dir / f'{mesh_name}_leadfield.hdf5')
    np.save(fem_dir / 'gain.npy', fwd['data'])

def prepare_forward(fwd_name, contains_ref=False):
    """Hacked version of the same function from simulation.eeg """
    with h5py.File(fwd_name, "r") as f:
        lf = f["mesh_leadfield"]["leadfields"]["tdcs_leadfield"]

        # TODO: Support for non-'middle gm' exclusive source spaces.

        # Interpolated solutions are per point, solutions sampled in tissues
        # are per cell (tetrahedra).

        # Get channel info
        ch_names = lf.attrs["electrode_names"].tolist()
        ch_ref = lf.attrs["reference_electrode"]

        # Get source space info
        # points = f['mesh_leadfield']['nodes']['node_coord'][:]
        # tris = f['mesh_leadfield']['elm']['node_number_list'][:, :3] - 1
        # point_start_idx = f['mesh_leadfield'].attrs['node_ix']
        # tris_start_idx = f['mesh_leadfield'].attrs['elm_ix']

        # Surface IDs (these are lh and rh)
        # interp_ids = f['mesh_leadfield'].attrs['interp_id'].tolist()

        # Get the leadfield and invert it when going from TES to 'EEG mode' due
        # to reciprocity
        lf = -lf[:]

    # Forward solution
    # Insert the reference channel and rereference to an average reference
    if not contains_ref:
        lf = np.insert(lf, ch_names.index(ch_ref), np.zeros((1, *lf.shape[1:])), axis=0)
    else:
        ch_names.remove(ch_ref)
    lf -= lf.mean(0)
    nchan, nsrc, nori = lf.shape  # leadfield is always calculated in x, y, z
    assert len(ch_names) == nchan

    return dict(
        data=lf,
        ch_names=ch_names,
        n_channels=nchan,
        n_sources=nsrc,
        n_orientations=nori,
    )

if __name__ == '__main__':
    assert len(sys.argv) == 2
    run_simulation(int(sys.argv[1]))


# @ray.remote
# def custom_run(lf, src_points):
#     #Run a simulation and interpolate to src_points.

#     dir_name = lf.pathfem

#     lf._set_logger()
#     lf._prepare()
#     #lf._add_electrodes_from_cap()
#     #assert all([len(e.thickness) != 3 for e in lf.electrode])
#     lf._add_el_conductivities()

#     roi = lf.tissues# + [2]

#     # Get names for leadfield and file of head with cap
#     fn_hdf5 = op.join(dir_name, lf._lf_name())
#     fn_el = op.join(dir_name, lf._el_name())

#     w_elec, electrode_surfaces = lf._place_electrodes()
#     mesh_io.write_msh(w_elec, fn_el)
#     scalp_electrodes = w_elec.crop_mesh([1005] + electrode_surfaces)
#     scalp_electrodes.write_hdf5(fn_hdf5, 'mesh_electrodes/')

#     # Write roi, scalp and electrode surfaces hdf5
#     roi_msh = w_elec.crop_mesh(roi)

#     M = roi_msh.interp_matrix(
#         src_points,
#         out_fill='nearest',
#         element_wise=True)

#     def post(out_field, M):
#         return M.dot(out_field)

#     post_pro = partial(post, M=M)

#     fn_roi = op.join(dir_name, lf._mesh_roi_name())

#     #mesh_lf.write(fn_roi)
#     #mesh_lf.write_hdf5(fn_hdf5, 'mesh_leadfield/')

#     # Run Leadfield
#     dset = 'mesh_leadfield/leadfields/tdcs_leadfield'

#     c = sim_struct.SimuList.cond2elmdata(lf, w_elec)
#     fem.tdcs_leadfield(
#         w_elec, c, electrode_surfaces, fn_hdf5, dset,
#         current=1., roi=roi,
#         post_pro=post_pro, field=lf.field,
#         solver_options=lf.solver_options,
#         n_workers=1)

#     with h5py.File(fn_hdf5, 'a') as f:
#         f[dset].attrs['electrode_names'] = [el.name.encode() for el in lf.electrode]
#         f[dset].attrs['reference_electrode'] = lf.electrode[0].name
#         f[dset].attrs['electrode_pos'] = [el.centre for el in lf.electrode]
#         f[dset].attrs['electrode_cap'] = lf.eeg_cap.encode() if lf.eeg_cap is not None else 'none'
#         f[dset].attrs['electrode_tags'] = electrode_surfaces
#         f[dset].attrs['tissues'] = lf.tissues
#         f[dset].attrs['field'] = lf.field
#         f[dset].attrs['current'] = '1A'
#         f[dset].attrs['units'] = 'V/m' # E field
#         f[dset].attrs['d_type'] = 'node_data' # map_to_surf
#         f[dset].attrs['mapped_to_surf'] = 'True'
#     lf._finish_logger()

# models = (1, 3, '3_eq')

# # Make / clean up run dirs
# m2m = {}.fromkeys(dirs)
# for d in dirs:
#     m2m[d] = {}.fromkeys(models)
#     for k in m2m[d]:
#         m2m[d][k] = op.join(d, 'm2m_sph{}'.format(k))
#         if op.exists(m2m[d][k]):
#             for f in glob.iglob(op.join(m2m[d][k], '*')):
#                 os.remove(f)
#         else:
#             os.mkdir(m2m[d][k])

# lf = {d:{} for d in dirs}
# for d in dirs:
#     # Sphere 1
#     x = sim_struct.TDCSLEADFIELD()
#     x.fnamehead = op.join(d, 'sph1.msh')
#     x.subpath = m2m[d][1]
#     x.pathfem = m2m[d][1]
#     x.field = 'E'
#     x.tissues = []
#     x.cond = S
#     x.interpolation = (src_off[1], )
#     x.interpolation_tissue = [5]
#     x.eeg_cap = None
#     x.electrode = elec
#     lf[d][1] = x

#     # Sphere 3
#     x = sim_struct.TDCSLEADFIELD()
#     x.fnamehead = op.join(d, 'sph3.msh')
#     x.subpath = m2m[d][3]
#     x.pathfem = m2m[d][3]
#     x.field = 'E'
#     x.tissues = []
#     x.cond = S # conductivities of compartments
#     x.interpolation = (src_off[3], )
#     x.interpolation_tissue = [2]
#     x.eeg_cap = None
#     x.electrode = elec
#     lf[d][3] = x

#     # Sphere 3_eq
#     x = sim_struct.TDCSLEADFIELD()
#     x.fnamehead = op.join(d, 'sph3.msh')
#     x.subpath = m2m[d]['3_eq']
#     x.pathfem = m2m[d]['3_eq']
#     x.field = 'E'
#     x.tissues = []
#     x.cond = Seq # !
#     x.interpolation = (src_off[1], )
#     x.interpolation_tissue = [2,4,5]
#     x.eeg_cap = None
#     x.electrode = elec
#     lf[d]['3_eq'] = x

# ray.init(log_to_driver=False)

# @ray.remote
# def run_simulation(lf):
#     lf.run(save_mat=False)

# # Start the jobs
# result_ids = []
# for d in dirs:
#     for m,i in zip((1, 3, '3_eq'), (1, 3, 1)):
#         result_ids.append(custom_run.remote(lf[d][m], src_points[i]))
#         #result_ids.append(run_simulation.remote(lf[d][m]))

# # Report progress
# while len(result_ids) > 0:
#     print('Jobs remaining {:2d}'.format(len(result_ids)))
#     finished, not_finished = ray.wait(result_ids)
#     result_ids = not_finished

# print('All jobs finished')
# #ray.shutdown()

"""
for d in dirs:
    # sphere 1 layer
    # --------------------------------------
    lf = sim_struct.TDCSLEADFIELD()
    lf.fnamehead = op.join(d, 'sph1.msh')
    lf.subpath = m2m[1]
    lf.pathfem = m2m[1]
    lf.field = 'E'
    lf.tissues = [5] # volume/surface to read field values
    lf.cond = S
    lf.interpolation = (src_off[1], )
    lf.eeg_cap = None
    lf.electrode = elec

    #lf.run(save_mat=False)
    custom_run(lf, src_points[1], cpus)

    # sphere 3 layers
    # --------------------------------------
    lf = sim_struct.TDCSLEADFIELD()
    lf.fnamehead = op.join(d, 'sph3.msh')
    lf.subpath = m2m[3]
    lf.pathfem = m2m[3]
    lf.field = 'E'
    lf.tissues = [2]
    lf.cond = S # conductivities of compartments
    lf.interpolation = (src_off[3], )
    lf.eeg_cap = None
    lf.electrode = elec

    #lf.run(save_mat=False)
    custom_run(lf, src_points[3], cpus)

    # sphere 3 layers equal conductivities
    # --------------------------------------
    lf = sim_struct.TDCSLEADFIELD()
    lf.fnamehead = op.join(d, 'sph3.msh')
    lf.subpath = m2m['3_eq']
    lf.pathfem = m2m['3_eq']
    lf.field = 'E'
    lf.tissues = [2,4,5]
    lf.cond = Seq # !
    lf.interpolation = (src_off[1], )
    lf.eeg_cap = None
    lf.electrode = elec

    #lf.run(save_mat=False)
    custom_run(lf, src_points[1], cpus)
"""
