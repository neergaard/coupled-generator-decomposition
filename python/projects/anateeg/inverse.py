import itertools

import mne
import numpy as np
import pandas as pd
from scipy.spatial import distance

from simnibs.simulation import eeg_mne_tools

from projects.mnieeg.forward import morph_forward
from projects.mnieeg.inverse import (
    _get_gain_matrix,
    compute_dipole_fit_linear,
    compute_lcmv,
    compute_music,
    init_results_dataframe,
    make_resolution_and_inverse_matrix,
    peak_err,
    sample_noise_vectors,
    sample_vector_in_cone,
    sd_ext,
    cog_err,
    simulate_data,
    smooth_fwd,
)

from projects.mnieeg import utils as mnieeg_utils
from projects.mnieeg.config import Config as mnieeg_Config

from projects.anateeg import utils
from projects.anateeg.config import Config
from projects.anateeg.mne_tools import match_fwd_to_src

LIMIT_GB = 3


def get_cov_from_mnieeg(subject_id):
    # subject_id is used as seed for reproducibility
    rng = np.random.default_rng(seed=int(subject_id))
    sub = rng.choice(list(mnieeg_utils.GroupIO().subjects), size=1)[0]
    print(f"Using noise covariance from subject {sub} from the MNIEEG project")
    mnieeg_io = mnieeg_utils.SubjectIO(sub)
    mnieeg_io.data.update(stage="preprocessing", suffix="cov")
    cov_fwd = mne.read_cov(mnieeg_io.data.get_filename(space="forward"))
    cov_inv = mne.read_cov(mnieeg_io.data.get_filename(space="inverse"))
    return cov_fwd, cov_inv


def compute_resolution_metrics(subject_id):
    """
    """
    random_state = int(subject_id)

    signal_template, times = np.load(
        mnieeg_Config.path.RESOURCES / "sub-01_first_signal_comp_xdawn.npy"
    )
    mask = times > 0  # = 99 samples
    signal_template = signal_template[mask]  # - 0.2 to 0.5 s; 0 at index 40
    times = times[mask]
    n_samples = len(times)

    nave = Config.inverse.NAVE
    cone_angle = Config.inverse.CONE_ANGLE
    fixed_orient_prior = Config.inverse.orientation_prior == "fixed"

    kw_morph = dict(stage="forward", suffix="morph", extension="h5")
    kw_reset = dict(forward=None, snr=None, inverse=None)

    io = utils.SubjectIO(subject_id)
    # Since this is faked and not derived from data, we need to add an average
    # projector for MNE to allow us to invert data
    info = mne.io.read_info(io.data.get_filename(stage="forward", suffix="info"))
    projs = mne.io.proj.make_eeg_average_ref_proj(info)
    info["projs"].extend([projs])

    print("Computing the (scaled) noise whitener")
    # choose a random subject from mnieeg and use their covariance estimates
    # Noise is sampled from `cov_fwd`
    # The noise whitener is computed from `cov_inv`
    cov_fwd, cov_inv = get_cov_from_mnieeg(subject_id)
    noise_whitener, _ = mne.cov.compute_whitener(cov_inv, info, pca="white")
    noise_whitener *= np.sqrt(nave)

    print("Generating (scaled) noise vectors")
    noise_vecs = sample_noise_vectors(cov_fwd, n_samples * nave, random_state)
    noise_vecs = np.ascontiguousarray(noise_vecs.T).reshape(-1, nave, n_samples)
    noise_vecs = noise_vecs.mean(1)
    noise_vecs_std = noise_vecs.std()

    print("Generating true source topographies")
    ref = Config.forward.REFERENCE + "_sample_cond"
    fwd_ref_fname = io.data.get_filename(stage="forward", forward=ref, suffix="fwd")
    fwd_ref = mne.convert_forward_solution(
        mne.read_forward_solution(fwd_ref_fname), force_fixed=True
    )
    if Config.inverse.SMOOTH_SOURCE_ACTIVITY:
        print("  Smoothing source activity")
        smooth_fwd(fwd_ref)

    # Match SNR of noise and signal (leadfield) by matching the global
    # field power (std since these are zero mean vectors)
    gain_ref = _get_gain_matrix(fwd_ref)
    gain_ref /= gain_ref.std(1, keepdims=True)
    gain_ref *= noise_vecs_std

    print("  Computing distance matrix")
    unit = dict(m=1, cm=1e2, mm=1e3)
    src_pos = unit["cm"] * np.row_stack([s["rr"] for s in fwd_ref["src"]])
    dist_mat = distance.squareform(distance.pdist(src_pos))

    morph_ref = mne.read_source_morph(
        io.data.get_filename(forward=Config.forward.REFERENCE, **kw_morph)
    )

    comb_fun_met = tuple(
        itertools.product(Config.resolution.FUNCTIONS, Config.resolution.METRICS)
    )
    df = init_results_dataframe(Config)

    print("Generating data")
    data_snr_max, nw_data_snr_max, nw_data_cov, data_whitener = simulate_data(
        Config.inverse.SNR, gain_ref, noise_vecs, noise_whitener, signal_template
    )

    for fm, fm_src in zip(Config.forward.MODELS, Config.forward.SOURCE_SPACE):
        fm_name = Config.plot.fwd_model_name[fm]
        print(f"Computing resolution metrics for model {fm_name}")
        io.data.update(forward=fm)
        fwd = mne.read_forward_solution(
            io.data.get_filename(stage="forward", suffix="fwd", extension="fif")
        )

        # (1) charm and template both have a different source space so morph
        # (2) fieldtrip and mne, and use the same source space as the
        #     reference model, however, some sources may be unused if they are
        #     outside of gray matter, hence, we restrict the simulations to
        #     these valid sources and interpolate results onto fsaverage
        morph = mne.read_source_morph(io.data.get_filename(**kw_morph))
        if fm_src != "reference":
            assert fm in ("charm", "template")
            print(f"Morphing {fm_name} to the space of reference model")

            morph_fm_to_ref = eeg_mne_tools.make_source_morph(
                fwd["src"],
                fwd_ref["src"],
                io.simnibs[fm].get_path("m2m"),
                io.simnibs["reference"].get_path("m2m"),
                Config.forward.SUBSAMPLING,
            )
            morph_forward(fwd, morph_fm_to_ref, fwd_ref["src"])
            morph = morph_ref

        # Only simulate valid sources (those which exist in `fwd`)
        if fwd["nsource"] != fwd_ref["nsource"]:
            assert fm in ("mne", "fieldtrip")
            assert fwd["nsource"] < fwd_ref["nsource"]
            print(f"Using {fwd['nsource']} sources")
            verts_in_use = np.concatenate(
                [s["vertno"] + n for s, n in zip(fwd["src"], (0, fwd["src"][0]["np"]))]
            )
        else:
            verts_in_use = slice(None)
        use_data_snr_max = {k: v[verts_in_use] for k, v in data_snr_max.items()}
        use_nw_data_snr_max = {k: v[verts_in_use] for k, v in nw_data_snr_max.items()}
        use_nw_data_cov = {k: v[verts_in_use] for k, v in nw_data_cov.items()}
        use_data_whitener = {k: v[verts_in_use] for k, v in data_whitener.items()}
        use_dist_mat = dist_mat[verts_in_use][:, verts_in_use]

        print("Projecting the forward solution onto 'noisy' normals")
        for s in fwd["src"]:
            s["nn"] = sample_vector_in_cone(s["nn"], cone_angle)

        # Base inv is shared for all types of MNE solvers
        inv = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            cov_inv,
            fixed=fixed_orient_prior or "auto",
            # depth=Config.inverse.DEPTH_WEIGHTING,
        )

        print("Extracting gain matrices and whitening")
        if fixed_orient_prior:
            # gain.shape == (n_src, n_chan)
            gain = _get_gain_matrix(mne.convert_forward_solution(fwd, force_fixed=True))
        else:
            # (n_src, n_chan, 3)
            gain = _get_gain_matrix(fwd)
        # gain /= gain_free.std(1, keepdims=True)
        # gain *= noise_vecs_std
        nw_gain = np.squeeze(noise_whitener @ np.atleast_3d(gain))

        for snr in Config.inverse.SNR:
            for im in Config.inverse.METHODS:
                im_type = Config.inverse.METHOD_TYPE[im]

                print(f"Computing inverse solutions for {im}")
                if im_type == "mne":
                    lambda2 = snr ** -2
                    inv_prep = mne.minimum_norm.prepare_inverse_operator(
                        inv.copy(), nave, lambda2, im
                    )
                    res_mat = make_resolution_and_inverse_matrix(
                        use_data_snr_max[snr], inv_prep, im, lambda2
                    )
                else:
                    if im_type == "dipole":
                        res_mat = compute_dipole_fit_linear(
                            nw_gain, use_nw_data_snr_max[snr]
                        )
                    elif im_type == "music":
                        res_mat = compute_music(nw_gain, use_nw_data_cov[snr])
                    elif im_type == "beamformer":
                        res_mat = compute_lcmv(gain, use_data_whitener[snr], nw_gain)
                    else:
                        raise ValueError

                for fun, met in comb_fun_met:
                    column = (fm_name, im, snr, fun, met)
                    if column in df:
                        if met == "peak_err":
                            res_met = peak_err(res_mat, use_dist_mat, fun)
                        elif met == "sd_ext":
                            res_met = sd_ext(res_mat, use_dist_mat, fun)
                        elif met == "cog_err":
                            res_met = cog_err(res_mat, src_pos[verts_in_use], fun)
                        else:
                            raise ValueError
                        # We already ensured morph_ref is a valid morpher
                        df[column] = morph.morph_mat @ res_met

    prefix = "fixed" if fixed_orient_prior else None
    io.data.update(
        stage="source",
        prefix=prefix,
        space="fsaverage",
        suffix="res",
        extension="pickle",
        **kw_reset,
    )
    io.data.path.ensure_exists()
    df.to_pickle(io.data.get_filename())
