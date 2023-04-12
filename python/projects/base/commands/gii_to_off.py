from pathlib import Path
import sys

import nibabel as nib
import pyvista as pv

if __name__ == "__main__":
    fname = Path(sys.argv[1])
    gii = nib.load(fname)
    s = pv.make_tri_mesh(gii.agg_data("pointset"), gii.agg_data("triangle"))
    pv.save_meshio(fname.with_suffix(".off"), s)
    
