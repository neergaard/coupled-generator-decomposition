from projects.simval.config import Config

options = \
"""// to mesh the volumes call gmsh with
// gmsh -3 -bin -o file.msh file.geo

Mesh.Algorithm3D=4;
Mesh.Optimize=1;
Mesh.OptimizeNetgen=1;

"""

merge1 = \
"""Merge "{}";

"""

merge3 = \
"""Merge "{0}";
Merge "{1}";
Merge "{2}";

"""

mesh1 = \
"""Surface Loop(1) = {1};
Volume(1) = {1};
Physical Surface(1005) = {1};
Physical Volume(5) = {1};
"""

mesh3 = \
"""Surface Loop(1) = {1}; // brain.stl
Surface Loop(2) = {2}; // skull.stl
Surface Loop(3) = {3}; // scalp.stl

Volume(1) = {1};       // brain
Volume(2) = {1, 2};    // skull (outside brain, inside skull)
Volume(3) = {2, 3};    // scalp (outside skull, inside scalp)

// LHS: target surface region number, RHS: surface number (i.e. from merge ...)
Physical Surface(1002) = {1};
Physical Surface(1004) = {2};
Physical Surface(1005) = {3};

// LHS: target volume region number, RHS: volume number
Physical Volume(2) = {1};
Physical Volume(4) = {2};
Physical Volume(5) = {3};
"""

for d in Config.path.SPHERE.glob('density_*'):
    assert d.is_dir()
    surfaces = tuple(d.glob('*.stl'))
    brain = [i for i in surfaces if 'brain' in i.stem]
    skull = [i for i in surfaces if 'skull' in i.stem]
    scalp = [i for i in surfaces if 'scalp' in i.stem]

    f = open(d / "sphere_3_layer.geo", "w")
    f.write(options + merge3.format(*brain+skull+scalp) + mesh3)
    f.close()
