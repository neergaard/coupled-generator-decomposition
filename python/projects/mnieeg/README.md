# Usage

Files in `subject` subfolder process one subject and are called using `slurm` by using the `slurm_master` script in `project-tools/projects-bash/projects/mnieeg` like `slurm_master charm`, i.e., *without* `.py`.

Files in `group` subfolder are run directly as they process all subjects.

## Order to Run

##### Prepare the Data
- `phatmags_to_bids`
Extract data from the PhaTMagS project directory and add information about bad channels, bad segments, and fiducial coordinates. This creates `/data`.


##### Create Head Model
- `charm`
Create head model using CHARM. This creates `/simnibs`.
- `charm_mni152`
Create head model of the MNI152 template using CHARM. This creates `/simnibs/sub-mni152`. Run as `python charm_mni152.py` and not via slurm.

##### Detect (and Correct) Outliers Due to Equipment Malfunction
- `correct_digitizations`
Detect outliers in the digitizations by transforming all channel positions to MNI space (using the nonlinear warp from CHARM), computing the median, and comparing with this. Outliers are replaced with this median in MNI space and warped back to subject space (unless more than one outlier is present---it is a little complicated).


##### Preprocess EEG Data
- `preprocess`
Preprocess EYES OPEN data. This includes filtering, HEOG/VEOG artifact removal using ICA, interpolation of bad channels, setting an average reference.
- `covariance`
Compute covariance.


##### Compute the Forward Solutions
From here on we work only with the chosen session/included subjects.

1. `prepare_for_forward`
Create the montage files, info object, head-mri transformation (identity; positions are explicitly transformed to MRI space), and downsample the source space for each forward model. This step also adapts the MNI152 head model and surfaces to the subject using Brainstorm's method. Will write the montages to a `vtm` file for visual inspection.
2. `compute_forward`
Run forward simulations generating the leadfield matrix.
3. `compute_forward_sample_cond`
Run forward simulation on the reference but with alternative conductivities sampled from beta distributions. We use this for generating the sources in the simulations. (This will also prepare it for inverse!)
4. `prepare_for_inverse`
Generate forward object, source space, and source morph files in MNE format. Morph forward solutions to *fsaverage* (to enable comparisons between models generated using the subject MRI and the MNI template).

##### Compute Inverse Operators and Resolution Metrics
1. `assemble_inverse`
Compute inverse operators for all forward models and different SNR levels.
2. `compute_resolution_metrics`
Compute resolution *matrix* and the desired resolution *metrics* for all combinations of forward models, inverse operators, and SNR levels. Only the metrics are saved.

##### Prepare for Group Analysis
1. `collect_channel_eval`
Collect all channel positions in a DataFrame.
2. `collect_forward_eval`
Compute and collect forward evaluation metrics (RDM and lnMAG) in a DataFrame.
3. `collect_inverse_eval`
Collect all resolution metrics in a single DataFrame.


`viz_headmodel_charm_vs_template`
`viz_digitizations_topo`
Must be run before 3d!
`viz_digitizations_3d`

`viz_channel`
Contains MNE plots, i.e., channel topomaps.
`viz_forward`
Contains PyVista plots and densities.


## CHARM Segmentations QA
- `Subject 1` Bad affine registration. Fixed by restricting rotation around x axis .
- `Subject 9` Bad neck. Fixed by adjusting `neck_search_bounds`.
- `Subject 14` Weird thing around nose. OK
- `Subject 20` Bad occipital area (skin moved inwards) probably due to bad affine registration (too much rotation upwards around x). Fixed by adjusting `affine_scales`.
- `Subject 23` Bad neck. Fixed by adjusting `neck_search_bounds`.
- `Subject 28` Bad frontal area (skin moved inwards). Template just seems to fit head shape poorly. Fixed by adjusting `affine_scales`.

## Correction of `easycap_BC_TMS64_X21.csv`

Channels 50 and 52 were corrected by replacing the x coordinate with the mean of [49, 51] and [51, 53], respectively. Channel 22 was corrected by replacing the x and y coordinate with the means of [21, 23]. Finally, all channels were (re)projected onto the MNI skin surface.


## /

`config.py`
Specify global configuration.

`make_mnihead_3d.py`
Prepare a 3D model of the MNI head (from a CHARM segmentation) which can be printed on a 3D printer.

## commands/

`charm.py`

`covariance.py`

`create_symlinks_to_template`
Create symlinks to the template subject (sub-template) for each subject so that the solution without an MRI can be easily computed for each subject.

`preprocess.py`


`phatmags_to_bids.py`

`prepare_for_forward`
Prepare each for each forward model, i.e., template, digitize, warp, and standard. This involves generating the MRI-head transform, info object and csv file (for SimNIBS) with the relevant electrode positions.

`prepare_forward`
Generate source space and morph and compute forward solution for each scenario.

## phatmags/

`bads.py`
Annotations of bad channels and segments for each subject. Also whether or not to exclude a particular subject.

`mrifiducials.py`
Coordinates of fiducials in each subject.

## polaris_tracking/

`polaris_digitize.py`
Digitize points using an NDI Polaris device.