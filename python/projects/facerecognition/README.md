


#### prepare the head models
1. `freesurfer_reconall`
1. `charm`
1. `python charm_mni152.py`
1.2. `prepare_flash`


1. `python bids_update.py`
Creates symbolic links of sidecar files to derivative folder.


#### Preprocess EEG/MEG Data
2. `preprocess`


#### Forward
`prepare_for_forward`
Prepare Info, mri to head transformation, and montage.

`deform_mni_template`
also creates the montage file for template (just a copy of reference)

#### compute forward models and prepare
1 `prepare_forward_simnibs`
Does make_forward_solution. This needs to be run first as it also computes morph and makes src space.
`prepare_forward_mne`
Does make_bem_surfaces and make_forward_solution.
`prepare_forward_fieldtrip`
Does make_headmodel and make_forward_solution.


#### compute inverse operators

