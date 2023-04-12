import meshio
import numpy as np
from functools import partial

#from simnibs.simulation.analytical_solutions.sphere import potential_homogeneous_dipole, potential_dipole_3layers

from projects.simval.config import Config
from projects.simval.sphere_utils import potential_dipole_3layers

cond = Config.sphere.conductivity
radii = Config.sphere.radii

sol_dir = Config.path.SPHERE / 'analytical_solutions'
if not sol_dir.exists():
    sol_dir.mkdir()

sensor_pos = meshio.read(Config.path.SPHERE / 'sensor_positions.stl').points

src_pos = np.load(Config.path.SPHERE / 'src_coords.npy')
src_pos = src_pos.reshape(-1, 3)

# x, y, z
dip_ori = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])


print('Computing Analytical Solutions')
sol = np.zeros((len(sensor_pos), len(src_pos), 3))
# if m == 1:
    # fun = partial(potential_homogeneous_dipole, radius[m], S_scalp,
    #                 src_points[m], detector_positions=elec)
fun = partial(potential_dipole_3layers, radii, cond['brain_scalp'],
                cond['skull'], src_pos, surface_points=sensor_pos,
                nbr_polynomials=100)
for i in range(len(dip_ori)):
    V = fun(dipole_moment=dip_ori[i])
    sol[:, :, i] = V
sol -= sol.mean(0)
np.save(sol_dir / 'solution_3_layer.npy', sol)
