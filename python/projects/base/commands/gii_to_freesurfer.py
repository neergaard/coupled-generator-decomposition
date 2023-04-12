from pathlib import Path
import sys

import nibabel as nib

if __name__ == "__main__":
    fname = Path(sys.argv[1])
    gii = nib.load(fname)
    nib.freesurfer.write_geometry(
        fname.with_suffix(""), gii.agg_data("pointset"), gii.agg_data("triangle")
    )
