



`python prepare_skull_reco_data.py`
Create the mri session for each subject.

#### prepare the head models
`slurm_master_bash fs_reconall`

`update_manual_segmentation`
Correct manual segmentations using FreeSurfer surfaces.

`charm`
`charm_reference`
`charm_mni152.py` # edit slurm array to 0..4!
to be able to run this using atlases 1 to 4 we need to replace `spineAlphas = alphas[:, 48]` with `spineAlphas = alphas[:, 9]` and have an empty dict work with `writeBiasCorrectedImagesAndSegmentation`!


`compute_mnieeg_cov` NOT USED

#### prepare the montages and data
`prepare_montage_and_data`
    `make_montage` - (projected) montage csv for reference and charm
    `make_data` - info, cov, mri_head_t


`deform_mni_template` - also creates the montage file for template (just a copy of reference)


#### compute forward models and prepare
`prepare_forward_simnibs`
Does make_forward_solution. This needs to be run first as it also computes morph and makes src space.
`prepare_forward_mne`
Does make_bem_surfaces and make_forward_solution.
`prepare_forward_fieldtrip`
Does make_headmodel and make_forward_solution.
`compute_forward_sample_cond`
Run forward simulation on the reference but with alternative conductivities sampled from beta distributions. We use this for generating the sources in the simulations. (This will also prepare it for inverse!)

#### compute inverse operators

`compute_resolution_metrics`
Choose a random subject from MNIEEG and use its covariance matrices when generating and inverting the data.


`viz_head_models`
`viz_forward`
