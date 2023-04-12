import subprocess
import sys

from projects.simval.sphere_utils import idx_to_density

def mesh_sphere(i):
    _, d = idx_to_density(i)

    # activate_simnibs = 'conda activate simnibs'
    # Seems I have problems reading meshes with format 4.X
    run_gmsh = '/mrhome/jesperdn/git_repos/simnibs/simnibs/external/bin/linux/gmsh -3 -bin -format msh2 -o {} {}'
    # call = '; '.join((activate_simnibs, run_gmsh))
    call = run_gmsh

    subprocess.run(["bash", "-c", call.format(d / 'sphere_3_layer.msh', d / 'sphere_3_layer.geo')])

if __name__ == '__main__':
    assert len(sys.argv) == 2
    mesh_sphere(int(sys.argv[1]))
