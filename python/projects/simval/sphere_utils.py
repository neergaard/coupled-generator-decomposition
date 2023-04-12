from numbers import Integral
import numpy as np

from projects.simval.config import Config

nax = np.newaxis

def idx_to_density(i=None):
    if i is not None:
        density = Config.sphere.node_densities[i]
        d = Config.path.SPHERE / f"density_{str(density).replace('.', '')}"
        assert d.is_dir()
    else:
        density = Config.sphere.node_densities
        d = [Config.path.SPHERE / f"density_{str(i).replace('.', '')}" for i in density]
    return density, d

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

def sph_to_cart(r, theta, phi):
    """
    points : r, theta, phi in columns
    """
    # points = np.atleast_2d(points)
    # r = points[:, 0]
    # theta = points[:, 1]
    # phi = points[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.squeeze(np.stack([x,y,z], axis=1))



def sample_close_points_on_sphere(points, s, N=100):
    """

    For each point in points, sample points around the north pole ([0, 0, 1])
    and rotate the sampled points to the correct position of the sphere.

    points :
        Points in cartesian coordinates.
    s : float
       Spread of the sampled points. If points are in mm, then ensure that
       99.7 percent of the sampled points are within s mm.
    N :
        Number of points to sample.
    """
    points = np.atleast_2d(points)
    Np = points.shape[0]
    r, theta, phi = np.atleast_2d(cart_to_sph(points)).T

    assert np.all(r-r[0] < 1e-5), 'All points must be on the same sphere (have same radius)'

    # find the angle that corresponds to d mm in radians on a sphere with radius r
    circumference = 2 * np.pi * r.mean()
    angle = s / circumference * 2 * np.pi

    # ensure that 99.7 percent of the sampled points are within s mm (as measured
    # on the sphere surface) by choosing sigma to be 1/3 of the angle
    sigma = angle / 3

    # sample in the vicinity of the point by sampling from a normal distribution of the angles
    stheta = np.random.normal(scale=sigma, size=(Np, N))
    sphi = np.random.uniform(0, 2*np.pi, (Np, N))
    sps = np.stack((np.broadcast_to(r, (N, Np)).T, stheta, sphi), axis=2)
    sp = sph_to_cart(sps.reshape(-1, 3)).reshape((Np, N, 3))

    # Rotate the sampled points from the north pole to point location
    # - theta rotates around y (xz plane)
    # - phi rotates around z (xz plane)
    RzRy = np.zeros((Np, 3, 3))
    RzRy[:, 0, 0] = np.cos(theta)*np.cos(phi)
    RzRy[:, 0, 1] = -np.sin(phi)
    RzRy[:, 0, 2] = np.sin(theta)*np.cos(phi)
    RzRy[:, 1, 0] = np.cos(theta)*np.sin(phi)
    RzRy[:, 1, 1] = np.cos(phi)
    RzRy[:, 1, 2] = np.sin(theta)*np.sin(phi)
    RzRy[:, 2, 0] = -np.sin(theta)
    RzRy[:, 2, 1] = 0
    RzRy[:, 2, 2] = np.cos(theta)

    sp = np.transpose(RzRy @ np.transpose(sp, [0, 2, 1]), [0, 2, 1])

    return np.squeeze(sp)


def potential_homogeneous_dipole(sphere_radius, conductivity, dipole_pos,
                                 dipole_moment, detector_positions):
    sphere_radius *= 1e-3
    dipole_pos = np.array(dipole_pos) * 1e-3
    dipole_pos = np.atleast_2d(dipole_pos)
    dipole_moment = np.array(dipole_moment, dtype=float)

    detector_positions = np.array(detector_positions) * 1e-3
    detector_positions = np.atleast_2d(detector_positions)

    assert dipole_pos.shape[1] == 3
    assert detector_positions.shape[1] == 3

    R = sphere_radius
    r0 = np.linalg.norm(dipole_pos, axis=1)
    assert np.all(r0 < sphere_radius)
    r = np.linalg.norm(detector_positions, axis=1)
    det_dip_diff = detector_positions[nax] - dipole_pos[:, nax]
    rp = np.linalg.norm(det_dip_diff, axis=2)

    if not np.allclose(r, R):
        raise ValueError('Some points are not on the surface!')

    cos_phi = dipole_pos.dot(detector_positions.T) / np.outer(r0, r)
    cos_phi[np.isnan(cos_phi)]  = 0

    r0_cosphi = r0[:, nax] * cos_phi
    second_term = 1. / (rp[..., nax] * R ** 2) * \
        (detector_positions[nax] + (detector_positions[nax] * r0_cosphi[..., nax] - R * dipole_pos[:, nax]) / \
        (R + rp - r0_cosphi)[..., nax])

    V = (2 * det_dip_diff / (rp ** 3)[:, :,nax] + second_term) @ dipole_moment
    V /= 4 * np.pi * conductivity

    return np.squeeze(V).T

def lpmn(m, n, x):
    """Like scipy.special.lpmn but vectorized in x. Does not calculate
    derivative values. Does not accept negative values of m.

    Identities from

        https://en.wikipedia.org/wiki/Associated_Legendre_polynomials

    PARAMETERS
    ----------
    m : int
        Order.
    n : int
        Degree.
    x : float | ndarray
        Value(s) for which to calculate function values.

    RETURNS
    -------
    pmn : (m+1, n+1[, x.shape]) array
        Function values for orders [0, 1, ..., m] and degrees [0, 1, ..., n].
    """
    assert isinstance(m, Integral) and isinstance(n, Integral)
    assert 0 <= m <= n
    x = np.array(x)

    pmn = np.zeros((m+1, n+1, *x.shape))

    # diagonal (identity 1)
    pmn[0,0] = 1
    for i in range(m):
        pmn[i+1, i+1] = -(2*i+1) * np.sqrt(1-x**2) * pmn[i, i]

    # n+1 diagonal (identity 3)
    for i in range(m if m == n else m+1):
        pmn[i, i+1] = x * (2*i+1) * pmn[i, i]

    # upper triangle (recurrence formula 1)
    for i in range(m+1):
        for j in range(i+1, n):
            pmn[i, j+1] = ((2*j+1) * x * pmn[i, j] - (j+i) * pmn[i, j-1]) / (j-i+1)

    return pmn

def potential_dipole_3layers(radii, cond_brain_scalp, cond_skull, dipole_pos,
                             dipole_moment, surface_points, nbr_polynomials=100):

    dipole_pos = np.array(dipole_pos)
    dipole_pos = np.atleast_2d(dipole_pos)
    dipole_moment = np.array(dipole_moment)
    surface_points = np.atleast_2d(surface_points)

    assert len(radii) == 3
    assert radii[0] < radii[1] and radii[1] < radii[2]
    assert dipole_moment.ndim == 1 and dipole_moment.shape[0] == 3
    assert dipole_pos.shape[1] == 3
    assert surface_points.shape[1] == 3

    dipole_moment_norm = np.linalg.norm(dipole_moment)
    dipole_pos_norm = np.linalg.norm(dipole_pos, axis=1)

    assert np.all(dipole_pos_norm < radii[0]), "All dipoles must be inside the inner sphere"

    xi = cond_skull / cond_brain_scalp
    R = radii[2] * 1e-3
    f1 = radii[0] * 1e-3 / R
    f2 = radii[1] * 1e-3 / R
    b = dipole_pos_norm * 1e-3 / R

    if not np.allclose(np.linalg.norm(surface_points, axis=1), R * 1e3):
        raise ValueError('Some points are not on the surface!')

    r_dir = dipole_pos / dipole_pos_norm[:, nax]
    at_origin = np.isclose(b, 0)
    if at_origin.any():
        r_dir[at_origin] = dipole_moment / dipole_moment_norm

    m_r = r_dir @ dipole_moment
    cos_alpha = surface_points @ r_dir.T / R * 1e-3
    t_dir = dipole_moment[nax] - m_r[:, nax] * r_dir

    is_radial = np.isclose(dipole_moment_norm, np.abs(r_dir @ dipole_moment))
    if is_radial.any():
        # try to set an axis in x, if the dipole is not in x
        in_x = np.isclose(np.abs(r_dir @ np.array([1, 0, 0])), 1)
        t_dir[is_radial & ~in_x] = np.array([1, 0, 0], dtype=float)

        # otherwise, set it in y
        t_dir[is_radial & in_x] = np.array([0, 1, 0], dtype=float)
        t_dir -= np.sum(r_dir * t_dir, 1)[:, nax]

    t_dir /= np.linalg.norm(t_dir, axis=1, keepdims=True)
    t2_dir = np.cross(r_dir, t_dir)
    m_t = t_dir @ dipole_moment
    beta = np.arctan2(surface_points @ t2_dir.T, surface_points @ t_dir.T)
    cos_beta = np.cos(beta)

    def d(nbrs, f1, f2):
        d_n = ((nbrs + 1) * xi + nbrs) * ((nbrs * xi) / (nbrs + 1) + 1) + \
              (1 - xi) * ((nbrs + 1) * xi + nbrs) * (f1 ** (2 * nbrs + 1) - f2 ** (2 * nbrs + 1)) - \
              nbrs * (1 - xi) ** 2 * (f1 / f2) ** (2 * nbrs + 1)
        return d_n

    P = lpmn(1, nbr_polynomials, cos_alpha)
    P = P[:, 1:] # discard 0th order polynomium

    nbrs = np.arange(1, nbr_polynomials+1)

    # dimensions of the lines in the sum:
    # poly, dipole
    # poly
    # poly, elec, dipole
    potentials = np.nan_to_num(
        np.sum(
            ((2 * nbrs[:, nax] + 1) / nbrs[:, nax] * b[nax] ** (nbrs[:, nax] - 1))[:, nax] * \
            ((xi * (2 * nbrs + 1) ** 2) / (d(nbrs, f1, f2) * (nbrs + 1)))[:, nax, nax] * \
            (nbrs[:, nax, nax] * m_r[nax, nax] * P[0] - m_t[nax, nax] * P[1] * cos_beta[nax]),
            axis=0)
        )
        # Why should it be a minus there?

    potentials /= 4 * np.pi * cond_brain_scalp * R ** 2

    return np.squeeze(potentials)
