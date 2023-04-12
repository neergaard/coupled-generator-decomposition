import meshio
import numpy as np

from simnibs.segmentation.brain_surface import fibonacci_sphere

from projects.simval.config import Config
from projects.simval.sphere_utils import sph_to_cart

"""
Approximate distance from scalp to brain is 15 mm:

Lu (2019). Scalp-to-cortex distance of left primary motor cortex and its
computational head model: Implications for personalized neuromodulation
"""

print('Generating spheres')

# node_density = n_nodes / area
radii = np.asarray(Config.sphere.radii)
node_densities = np.asarray(Config.sphere.node_densities)
areas = 4*np.pi*radii**2
n_nodes = np.round(np.outer(node_densities, areas)).astype(int)

for nd, nn in zip(node_densities, n_nodes):
    print('Density {}'.format(nd))
    brain, skull, scalp = [fibonacci_sphere(i, j) for i,j in zip(nn, radii)]
    for s, n in zip((brain, skull, scalp), Config.sphere.names):
        m = meshio.Mesh(s[0], dict(triangle=s[1]))
        d = Config.path.SPHERE / 'density_{}'.format(str(nd).replace('.','')).rstrip('0')
        if not d.exists():
            d.mkdir(parents=True)
        m.write(d / 'sph_{}.stl'.format(n))

print('Generating electrodes')

electrodes = fibonacci_sphere(Config.sphere.n_sensors, Config.sphere.radii[-1])
m = meshio.Mesh(electrodes[0], dict(triangle=electrodes[1]))
m.write(Config.path.SPHERE / 'sensor_positions.stl')

print('Generating sources...')

rng = np.random.default_rng()
theta = rng.uniform(low=0, high=2*np.pi, size=Config.sphere.n_rays)
phi = rng.uniform(low=0, high=np.pi, size=Config.sphere.n_rays)

radii = np.arange(Config.sphere.ray_start, Config.sphere.ray_stop+1)
coords = np.row_stack([sph_to_cart(r, theta, phi) for r in radii])

src_dir = Config.path.SPHERE
for name, array in zip(('radii', 'theta', 'phi', 'coords'), (radii, theta, phi, coords)):
    np.save(src_dir / f'src_{name}.npy', array)

# mock surface file for simnibs simulations
m = meshio.Mesh(coords, cells=dict(triangle=np.zeros((1, 3), dtype=int)))
m.write(Config.path.SPHERE / 'src_coords.off')

print(f'Created {len(radii)} sources in {Config.sphere.n_rays} random directions')
