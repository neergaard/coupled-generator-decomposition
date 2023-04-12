import itertools

import mne
from mne.io.constants import FIFF
import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy.sparse
import scipy.stats
from scipy.spatial.transform import Rotation

from simnibs.simulation import eeg_mne_tools

from projects.base import geometry
from projects.base.utils import BlockTimer

from projects.mnieeg import utils

from projects.mnieeg.config import Config
from projects.mnieeg.forward import morph_forward

from projects.facerecognition.utils import fsaverage_as_index

# mne.set_log_level("warning")

LIMIT_GB = 3


def sample_covariance_matrix(cov, nave, random_state=None):
    # Sample a scatter matrix from a Wishart distribution of the data
    # covariance matrix (https://en.wikipedia.org/wiki/Wishart_distribution)

    # Get the maximum likelihood estimate of the covariance matrix
    # If X (n_parameters, n_samples) is the data matrix then
    #   scatter matrix = X @ X.T
    #   covariance matrix = scatter matrix / n_samples
    reg = 1e-16 * np.eye(cov["dim"])
    # realistic number of samples:
    # n_epochs * n_seconds * sampling_freq
    n_samples = np.round(nave * 0.2 * 200).astype(int)
    wishart = scipy.stats.wishart(df=n_samples, scale=cov.data + reg)
    scatter_matrix_samples = wishart.rvs(size=1, random_state=random_state)
    return scatter_matrix_samples / n_samples  # max. likelihood estimate of cov


def smooth_fwd(fwd):
    """In-place"""
    # Smooth source activations are modelled by the abs(graph laplacian)
    # where each column is normalized such that it sums to one
    comb_tris = np.row_stack(
        [s["tris"] + offset for s, offset in zip(fwd["src"], [0, fwd["src"][0]["np"]])]
    )
    L = geometry.get_graph_laplacian(comb_tris)
    L.data = np.abs(L.data)
    norm = np.array(L.sum(1)).squeeze()  # L is symmetric
    L.data = L.data / norm[L.indices]  # normalize columns
    # L.data = L.data / np.repeat(norm, np.diff(L.indptr)) # normalize rows

    fwd["sol"]["data"] = fwd["sol"]["data"] @ L


def sample_noise_vectors(cov, size, random_state):
    # Draw `nave` noise vectors from a multivariate gaussian and average
    # emulating `nave` epochs
    multi_norm = scipy.stats.multivariate_normal(cov=cov.data, allow_singular=True)
    return multi_norm.rvs(size=size, random_state=random_state)
    # cov_fwd_sample = np.cov(noise_vecs.T)

    # noise = noise[multi_norm.logpdf(noise).argmax()]
    # noise /= np.linalg.norm(noise)

    # data_cov = {
    #     snr: make_data_cov(fwd_ref["sol"]["data"], cov.data, snr)
    #     for snr in Config.inverse.SNR
    # }


def make_batches(length, max_batch_size):
    start_ix = np.arange(0, length, max_batch_size).tolist()
    return list(zip(start_ix, start_ix[1:] + [length]))


def simulate_data(snr_levels, gain_ref, noise_vecs, noise_whitener, signal_template):
    """Simulate data with a particular SNR *in whitened space*, i.e., matched on the
    square root of the GFP.

    gain_ref = (n_ch, n_src)
    noise_vecs = (n_samples, n_epochs, n_ch)

    The noise_whitener should be constructed such that noise_vecs averaged over
    epochs and then whitened gives at GPF of 1 at each time point.

    gain_ref : (n_ch, n_src)
    noise_vecs : (n_ch, n_epochs, n_samples)
    noise_whitener : (n_ch, n_ch)

    """
    batch_size = 1000

    n_src, n_ch = gain_ref.shape
    _, n_samples = noise_vecs.shape
    # n_data = n_epochs * n_samples
    rank = np.linalg.matrix_rank(noise_whitener)

    # Match SNR in whitened space
    s = gain_ref.copy()
    s_gfp = np.sum((s @ noise_whitener.T) ** 2, 1, keepdims=True) / rank
    # np.sqrt(s_gfp) == snr in MNE-Python
    s /= np.sqrt(s_gfp)

    # Create SNR signal vector
    # fs = 200
    # tmin, tmax = 0, 0.5
    # s_start, s_stop = 0.1, 0.3  # signal start, stop
    # t = np.linspace(0, 0.5, n_samples)
    # print(f"Sampling frequency is {n_samples / (tmax-tmin)} Hz")
    # # t = np.arange(0, 0.5, 1 / fs)
    # snr_signal = np.zeros_like(t)
    # twin = slice(np.argmin(np.abs(t - s_start)), np.argmin(np.abs(t - s_stop)))
    # snr_signal[twin] = np.sin(np.linspace(0, np.pi, twin.stop - twin.start))
    # twin = slice(np.argmin(np.abs(t - 0)), np.argmin(np.abs(t - s_start)) + 1)
    # snr_signal[twin] = -0.5 * np.sin(np.linspace(0, np.pi, twin.stop - twin.start))
    # twin = slice(np.argmin(np.abs(t - s_stop)) - 1, np.argmin(np.abs(t - 0.4)))
    # snr_signal[twin] = -0.5 * np.sin(np.linspace(0, np.pi, twin.stop - twin.start))
    # twin = slice(np.argmin(np.abs(t - 0.4)) - 1, np.argmin(np.abs(t - 0.5)))
    # snr_signal[twin] = 0.5 * np.sin(np.linspace(0, np.pi, twin.stop - twin.start))

    # rescale so that max == 1
    snr_signal = signal_template / np.abs(signal_template.max())
    snr_max = snr_signal.argmax()  # snr_signal[snr_max] == 1 approximately

    # Data at max SNR
    print("Computing data vectors at max SNR")
    data_snr_max = {
        snr: snr * snr_signal[snr_max] * s + noise_vecs[None, :, snr_max]
        for snr in snr_levels
    }
    print("Whitening data vectors at max SNR")
    nw_data_snr_max = {snr: data_snr_max[snr] @ noise_whitener.T for snr in snr_levels}

    # Instead of explicitly forming the data vectors for all time points and
    # epochs, construct the data covariance as a sum of signal, noise, and
    # cross covariance components. This allows us to easily compute the daa
    # covariance for multiple SNR levels.

    # variance
    #   var(data) = SNR**2 * var(signal) + var(noise) + 2 * SNR * cross_var(signal,noise)
    # covariance
    #   cov(data) = SNR**2 * cov(signal) + cov(noise) + SNR * [cov(signal,noise) + cov(noise,signal)]
    signal = snr_signal[None, None] * s[..., None]
    normalizer = 1 / (n_samples - 1)

    # broadcast_shape = (batch_size, n_ch, n_samples)

    print("Computing signal covariance")
    # scale signal_cov by n_epochs as signal is only one epoch
    signal_demean = signal - signal.mean(-1, keepdims=True)
    signal_cov = signal_demean @ signal_demean.swapaxes(1, 2) * normalizer
    print("Computing noise covariance")
    noise_vecs_demean = noise_vecs - noise_vecs.mean(1, keepdims=True)
    # noise_vecs_demean /= np.sqrt(n_epochs)
    # noise_vecs_demean_aug = np.broadcast_to(
    #     noise_vecs_demean, shape=broadcast_shape
    # ).swapaxes(
    #     1, 2
    # )  # augment to size of chunk
    noise_cov = (
        noise_vecs_demean.reshape(n_ch, n_samples)
        @ noise_vecs_demean.reshape(n_ch, n_samples).T
        * normalizer
    )

    # cx_trace = np.trace(
    #     noise_whitener @ signal_cov @ noise_whitener.T, axis1=1, axis2=2
    # ).mean(0)
    # cn_trace = np.trace(noise_whitener @ noise_cov @ noise_whitener.T)
    # cx_trace_rescale = cn_trace / cx_trace

    # signal_cov *= cx_trace_rescale
    # signal_demean *= np.sqrt(cx_trace_rescale)

    with BlockTimer("Computing cross covariance"):
        cross_cov = np.zeros((n_src, n_ch, n_ch))
        batches = make_batches(n_src, batch_size)
        for batch in batches:
            sel = slice(*batch)
            # signal_demean_aug = np.broadcast_to(
            #     signal_demean[sel, :, None], broadcast_shape
            # ).reshape(cov_shape)
            noise_vecs_demean_aug = np.broadcast_to(
                noise_vecs_demean, shape=(batch[1] - batch[0], n_ch, n_samples)
            ).swapaxes(
                1, 2
            )  # augment to size of chunk
            cross_cov[sel] = signal_demean[sel] @ noise_vecs_demean_aug * normalizer

    print("Computing data covariance")
    # Compute a data covariance by actually forming the signal forming
    # data = snr * signal_demean + noise_vecs_demean_aug
    # data_cov_explicit = np.cov(data[-3].reshape(n_ch, n_data))
    # np.allclose(data_cov[-3]*1e11, data_cov_explicit[-3]*1e11)
    data_cov = {
        snr: (
            snr ** 2 * signal_cov
            + noise_cov[None]
            + snr * (cross_cov + cross_cov.swapaxes(1, 2))
        )
        for snr in snr_levels
    }

    print("Whitening data covariance")
    nw_data_cov = {
        snr: noise_whitener @ data_cov[snr] @ noise_whitener.T for snr in snr_levels
    }

    print("Computing data whitener")
    data_whitener = {snr: compute_whitener(data_cov[snr]) for snr in snr_levels}

    return data_snr_max, nw_data_snr_max, nw_data_cov, data_whitener


def init_results_dataframe(config):
    # CTFs are the same for all MNE methods since the noise normalization
    # simply scale each row by the same factor. Thus, we include the CTFs only
    # for MNE
    final_model_names = [config.plot.fwd_model_name[m] for m in config.forward.MODELS]
    columns = pd.MultiIndex.from_tuples(
        [
            (model, method, snr, function, metric)
            for model, method, snr, function, metric in itertools.product(
                final_model_names,
                config.inverse.METHODS,
                config.inverse.SNR,
                config.resolution.FUNCTIONS,
                config.resolution.METRICS,
            )
            if function != "ctf" or method in config.inverse.METHOD_CTF
        ],
        names=(
            "Forward",
            "Inverse",
            "SNR",
            "Resolution Function",
            "Resolution Metric",
        ),
    )
    index = fsaverage_as_index(config.inverse.FSAVERAGE)
    df = pd.DataFrame(columns=columns, index=index, dtype=float)
    return df


def compute_resolution_metrics(subject_id):
    """Compute resolution metrics for all combinations of forward models (fm),
    inverse models (im), and SNR levels (snr) using the forward model
    "digitize" as reference/ground truth, i.e.,

        R(fm, im, snr) = M(fm, im, snr) @ G(digitize)

    Resolution metrics are morphed to fsaverage.


    L = leadfield
    G = inverse operator
    n = noise vector = diag of noise cov. with a little added noise perhaps

    R = GL =  GL and with measurement noise R = G(L+Ln)

    Noise is independent of which source is estimated...

    - G is constructed using the CORRECT leadfield and noise cov
    - Noise is added to the "measurements" (i.e., the leadfield which
      defines the expected measurements from a source activity of
      value  1)

    R = GL = G(LI)
    R = GL(I + n)
    R = G(L+N) = GL + GN = R(signal) + R(noise)
    R = G(LS + N)

    R = GL(source_cov + N)
    where source cov could be the graph laplacian normalized so that
    the norm of each column is 1 (or so that diag is 1). This would
    correspond to slightly spatially extended source activity.

    ax.annotate('$m$', xy=(snr_max,0),  xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
    ax.vlines(snr_max, *ax.get_ylim(), color="k", linewidth=0.5, linestyle="--")


    snr_max = times[signal_template.argmax()]

    fig, ax = plt.subplots(figsize=figure_sizing.get_figsize("single"))
    ax.plot(times, signal_template/signal_template.max())
    ax.annotate(r"$m = \arg\max_t ~ c_t$", xy=(snr_max, ax.get_ylim()[0]), xycoords="data",
        xytext=(0.225, 0.1), textcoords="data",
        arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.3"))
    ax.grid(alpha=0.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("a.u.")
    ax.set_title("Canonical Time Course, $c_t$")
    fig.savefig("/mrhome/jesperdn/canonical_time_course.svg")


    """
    random_state = int(subject_id)

    signal_template, times = np.load(
        Config.path.RESOURCES / "sub-01_first_signal_comp_xdawn.npy"
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
    info = mne.io.read_info(
        io.data.get_filename(stage="forward", forward="digitized", suffix="info")
    )

    # Noise is sampled from `cov_fwd`
    # The noise whitener is computed from `cov_inv`
    cov_fwd = mne.read_cov(
        io.data.get_filename(stage="preprocessing", space="forward", suffix="cov")
    )
    cov_inv = mne.read_cov(
        io.data.get_filename(stage="preprocessing", space="inverse", suffix="cov")
    )

    print("Computing the (scaled) noise whitener")
    # The covariance (and hence noise whitener) is scaled to the noise level in
    # a single epoch. Since the noise falls with sqrt(nave), rescale such that
    # it matches the noise level of an evoked response created by averaging
    # nave epochs
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

    # Signal-to-noise ratio
    # https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    # SNR = E[ signal**2 ] / E[ noise**2 ]
    # If the expected value of the noise (E[noise]) is zero then
    #   E[ noise**2 ] = E[ (noise-E[ noise ]) **2 ] = variance
    # On the other hand, the expected value of the signal may not be zero and
    # therefore
    #   E[signal**2] = np.sum(signal**2) / len(signal)
    # whereas
    #   E[noise**2] = np.sum(noise**2)/len(noise)
    #               = np.sum((noise-noise.mean())**2)/len(noise)
    #               = np.var(noise)

    # In MNE, SNR is estimated as GFP.
    # GFP (global field power), i.e., "spatial standard deviation", either
    #   - RMS over channels (at each time point)
    #   - For EEG data with an average reference this is equal to its standard
    #   deviation

    # evoked data over all conditions projected onto the first xdawn signal
    # component for sub-01 in facerecognition data set...

    print("Generating data")
    data_snr_max, nw_data_snr_max, nw_data_cov, data_whitener = simulate_data(
        Config.inverse.SNR, gain_ref, noise_vecs, noise_whitener, signal_template
    )

    for fm in Config.forward.MODELS:
        fm_name = Config.plot.fwd_model_name[fm]
        print(f"Computing resolution metrics for model {fm_name}")
        io.data.update(forward=fm)
        info = mne.io.read_info(
            io.data.get_filename(stage="forward", suffix="info", extension="fif")
        )
        fwd = mne.read_forward_solution(
            io.data.get_filename(stage="forward", suffix="fwd", extension="fif")
        )
        # Check if the morpher is the same as the reference morpher
        # morph is the same for all models except template based one(s)
        morph = mne.read_source_morph(io.data.get_filename(**kw_morph))
        if (morph_ref.morph_mat.nnz != morph.morph_mat.nnz) or (
            (morph_ref.morph_mat != morph.morph_mat).nnz > 0
        ):
            print(f"Morphing {fm_name} to the space of reference model")
            assert fm == "template_nonlin"

            morph_fm_to_ref = eeg_mne_tools.make_source_morph(
                fwd["src"],
                fwd_ref["src"],
                io.simnibs_template.get_path("m2m"),
                io.simnibs.get_path("m2m"),
                Config.forward.SUBSAMPLING,
            )
            morph_forward(fwd, morph_fm_to_ref, fwd_ref["src"])

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
                        data_snr_max[snr], inv_prep, im, lambda2
                    )
                else:
                    # Here, `res_mat` is *not* a resolution matrix but just a
                    # collection of all simulations in columns (i.e., point
                    # spread functions)!
                    if im_type == "dipole":
                        res_mat = compute_dipole_fit_linear(
                            nw_gain, nw_data_snr_max[snr]
                        )
                    elif im_type == "music":
                        res_mat = compute_music(nw_gain, nw_data_cov[snr])
                    elif im_type == "beamformer":
                        res_mat = compute_lcmv(gain, data_whitener[snr], nw_gain)
                    else:
                        raise ValueError

                for fun, met in comb_fun_met:
                    column = (fm_name, im, snr, fun, met)
                    if column in df:
                        if met == "peak_err":
                            res_met = peak_err(res_mat, dist_mat, fun)
                        elif met == "sd_ext":
                            res_met = sd_ext(res_mat, dist_mat, fun)
                        elif met == "cog_err":
                            res_met = cog_err(res_mat, src_pos, fun)
                        else:
                            raise ValueError
                        # We already ensured morph_ref is a valid morpher
                        df[column] = morph_ref.morph_mat @ res_met

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
    # df_mat.to_pickle(
    #     io.data.get_filename(
    #         space="fsaverage", extension="pickle", suffix="mat", **kw_reset
    #     )
    # )


def compute_resolution_matrix_properties():
    """
    Dirichlet spread
    Backus-Gilbert spread
    Norms
    Unit covariance size

    """
    # res_mat = mne.minimum_norm.make_inverse_resolution_matrix(
    #     fwd_ref, inv_prep, im, lambda2
    # )

    # df_mat.loc[fm, im, snr] = [
    #     unit_covariance_size(inv_mat),
    #     dirichlet_spread(res_mat),
    #     backusgilbert_spread(res_mat, dist_mat),
    # ]

    # df_mat.loc[(fm, im, snr), "Unit Covariance Size"] = unit_covariance_size(
    #     inv_mat
    # )
    # df_mat.loc[(fm, im, snr), "Dirichlet Spread"] = dirichlet_spread(res_mat)
    # df_mat.loc[(fm, im, snr), "Backus-Gilbert Spread"] = backusgilbert_spread(
    #     res_mat, dist_mat
    # )

    # df_mat.loc[(fm, im, snr), "Matrix Norm"] = np.linalg.norm(res_mat)
    # df_mat.loc[(fm, im, snr), "Diagonal Norm"] = np.linalg.norm(
    #     np.diag(res_mat)
    # )

    # psf_norm = np.linalg.norm(res_mat, axis=0)
    # df_mat.loc[(fm, im, snr), "PSF_norm_mean"] = psf_norm.mean()
    # df_mat.loc[(fm, im, snr), "PSF_norm_std"] = psf_norm.std()

    # diag[(fm,im,snr)] = np.abs(np.diag(res_mat))
    # magn[(fm, im, snr)] = np.abs(res_mat).max(0)
    # norm[(fm,im,snr)] = np.linalg.norm(res_mat, axis=0)
    # if fm != Config.forward.REFERENCE:
    #     topo[(fm, im, snr)] = morph.morph_mat @ rdm(res_mat, res_mat_ref[im,snr])
    #     magn[(fm, im, snr)] = morph.morph_mat @ lnmag(res_mat, res_mat_ref[im,snr])


def sample_vector_in_cone(cone_vecs, angle):
    """

    Sample a new vector within a cone whose size is given by `angle` around
    each vector in `cone_vec`.

    That is, the angle between the vector defining the cone direction and the
    sample will be in the interval [0, angle].

    cone_vecs
    angle : float
        Angle in degrees between 0 and 180.

    """
    cone_vecs = np.atleast_2d(cone_vecs)
    assert 0 < angle <= 180

    seed = 0
    rng = np.random.default_rng(seed=seed)

    cone_vecs /= np.linalg.norm(cone_vecs, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(cone_vecs, axis=1), 1)

    n = cone_vecs.shape[0]

    theta = np.deg2rad(angle)

    # Uniformly on the entire unit sphere
    # z = (np.random.random(100)-0.5)*2
    z = np.cos(rng.random(n) * theta)
    phi = 2 * np.pi * rng.random(n)
    samples = np.column_stack(
        (np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z)
    )
    z_vec = np.array([0, 0, 1], dtype=float)

    # cone_vec = -np.random.random(3)  # np.array([1, 0, 0])
    # cone_vec /= np.linalg.norm(cone_vec)

    rotation_axis = np.cross(z_vec, cone_vecs)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=1, keepdims=True)
    rotation_angle = np.arccos(cone_vecs @ z_vec)

    quat = np.empty((n, 4))
    quat[:, :3] = np.sin(0.5 * rotation_angle[:, None]) * rotation_axis
    quat[:, 3] = np.cos(0.5 * rotation_angle)

    cone_vec_samples = np.empty_like(cone_vecs)
    for i, q in enumerate(quat):
        r = Rotation.from_quat(q)
        cone_vec_samples[i] = r.apply(samples[i])

    # The degrees with which the cone vectors have been rotated
    # np.rad2deg(np.arccos(np.sum(cone_vec * cone_vec_samples, 1)))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.quiver(*np.zeros_like(vec).T, *vec.T, linewidth=0.2)
    # ax.quiver(*np.zeros_like(vec).T, *vec_r.T, color="b", alpha=1, linewidth=0.2)
    # ax.quiver(0, 0, 0, *cone_vec, color="r", linewidth=2)
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])

    # i = 26
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.quiver(0, 0, 0, *cone_vec[i], linewidth=1)
    # ax.quiver(0, 0, 0, *cone_vec_samples[i], color="b", linewidth=1)
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])

    return cone_vec_samples


# points = np.concatenate([s['rr'] for s in fwd_ref['src']])
# tris = np.concatenate([s['tris']+i for s,i in zip(fwd_ref['src'], [0,10000])])
# m = pv.make_tri_mesh(points, tris)


def compute_resolution_metric(res_mat, dist_mat, fun, metric, threshold=0.5):

    if metric == "peak_err":
        met = peak_err(res_mat, dist_mat, fun)
    elif metric == "sd_ext":
        met = sd_ext(res_mat, dist_mat, fun, threshold)
    elif metric == "cog_err":
        met = cog_err(res_mat, dist_mat, fun)
    else:
        raise ValueError
    return met


def peak_err(res_mat, dist_mat, fun):
    assert res_mat.shape == dist_mat.shape
    assert fun in ("psf", "ctf")

    if fun == "psf":
        dist_mat = dist_mat.T
        res_mat = res_mat.T
    idx = np.arange(res_mat.shape[0])
    return dist_mat[idx, np.abs(res_mat).argmax(1)]


def sd_ext(res_mat, dist_mat, fun, threshold=0.5):
    assert res_mat.shape == dist_mat.shape
    assert fun in ("psf", "ctf")

    # Ensure that the quantity of interest is in rows. This way the `indices`
    # are continuous in the array for reduceat
    res_mat2 = res_mat ** 2
    if fun == "psf":
        dist_mat = dist_mat.T
        res_mat2 = res_mat2.T

    if not threshold:
        return np.sqrt(np.sum(res_mat2 * dist_mat ** 2, 1) / res_mat2.sum(1))

    assert 0 < threshold < 1
    mask = res_mat2 >= threshold * res_mat2.max(1)[:, None]
    res_mat2_mask = res_mat2[mask]
    dist_mat_mask = dist_mat[mask]
    indices = [0] + mask.sum(1)[:-1].cumsum().tolist()
    return np.sqrt(
        np.add.reduceat(res_mat2_mask * dist_mat_mask ** 2, indices)
        / np.add.reduceat(res_mat2_mask, indices)
    )


def cog_err(res_mat, src_pos, fun):
    """Vectorized center-of-gravity error computation (high memory usage)."""
    # res_mat = mne.minimum_norm.spatial_resolution._rectify_resolution_matrix(res_mat)
    # locations = mne.minimum_norm.spatial_resolution._get_src_locations(
    #     src
    # )  # locs used in forw. and inv. operator
    # locations *= 100  # convert to cm
    abs_res_mat = np.abs(res_mat.T if fun == "ctf" else res_mat)
    cogs = (abs_res_mat / abs_res_mat.sum(0)).T @ src_pos
    return np.linalg.norm(cogs - src_pos, axis=1)

    # # (from mne.minimum_norm.spatial_resolution.py)
    # # get vertices from source space
    # vertno_lh = src[0]["vertno"]
    # vertno_rh = src[1]["vertno"]
    # vertno = [vertno_lh, vertno_rh]

    # # Convert array to source estimate
    # return mne.SourceEstimate(cog_err, vertno, tmin=0.0, tstep=1.0)


def compute_whitener(cov, pca="white"):

    # For each cov/row
    #   cov = v @ w * v.T
    #   cov**-1 = v.T @ w**-1 @ v = v.T @ w**-0.5 @ w**-0.5 @ v
    #           = (cov**-0.5).T @ cov**-0.5
    # so
    #   cov**-0.5 = w**-0.5 @ v
    # i.e., broadcast each eigval to *rows* of v (eigvecs in columns)!
    n_src, n_chan, _ = cov.shape  # n_src, n_chan, n_chan
    w, v = np.linalg.eigh(cov)
    # rescale to allow rank estimation
    # original values of cov are ~1e-14
    # original values of w are ~1e-16
    nonzero = w * 1e5 ** 2 > np.finfo(float).resolution
    w[~nonzero] = 0  # remove negative values (noise)
    whitener = np.zeros((n_src, n_chan))
    whitener[nonzero] = 1 / np.sqrt(w[nonzero])
    whitener = whitener[..., None] * v.swapaxes(1, 2)
    if pca == "white":
        pass
    elif pca == False:
        whitener = v @ whitener
    else:
        raise ValueError
    return whitener


def make_data_cov_sampling(gain, noise_vecs, snr):
    """
    Make data covariance matrices by sampling.
    """
    n_samples, n_chan = noise_vecs.shape

    # Make a vector whose variance over *time* == 1
    var = np.random.randn(n_samples)
    var /= var.std()
    # signal_vecs are on the scale of noise_vecs
    signal_vecs = (
        var[:, None, None]
        * snr
        * gain
        / gain.std(0, keepdims=True)
        * np.std(noise_vecs, axis=1, keepdims=True)[..., None]
    )
    # Now signal_vecs.var() == noise_vecs.var() * snr**2 (approximately)
    data_vecs = signal_vecs + noise_vecs[..., None]
    data_vecs = data_vecs.transpose(2, 1, 0)  # (n_sources, n_channels, n_samples)
    data_cov = data_vecs @ data_vecs.transpose(0, 2, 1) / (n_samples - 1)
    # ensure positive semidefinite
    data_cov += np.finfo(float).eps * np.eye(n_chan)[None]
    return data_cov


def make_data_cov_direct(gain, noise_cov, snr):
    """Create data covariance matrices for each source in source space using
    its leadfield to make the signal covariance matrix (outer product) and
    adding the noise covariance.

    The noise covariance is scaled such that the variance of the noise in the
    direction of the signal matches that of the signal (i.e., SNR = 1) and then
    scaled to obtain the desired SNR (here we again use SNR of amplitudes).

    """
    match_variance = "total"

    # outer product of leadfield vectors
    n_chan = gain.shape[0]
    signal_cov = gain.T[..., None] * gain.T[:, None]  # / (8099 - 1)
    # signal_cov /= n_chan

    # Correct to ensure positive semidefinite by adding to the diagonal...
    tol = 1e-7  # ratio of the largest to smallest eigenvalue
    idx = np.arange(n_chan)
    signal_cov[:, idx, idx] += tol * np.trace(signal_cov, axis1=1, axis2=2)[:, None]

    # variance/power
    noise_var = np.trace(noise_cov)
    signal_var = np.trace(signal_cov, axis1=1, axis2=2)
    scale = noise_var / signal_var
    # scale = signal_var / noise_var

    signal_cov *= scale[:, None, None]
    # noise_cov = noise_cov[None] * scale[:, None, None]

    # limit = 1e-10
    # r = (-limit, limit)
    # plt.hist(data_cov[0].ravel(), 100, range=r);
    # plt.hist(data_cov_sampled.ravel(), 100, range=r);

    if match_variance == "total":
        data_cov = signal_cov * snr ** 2 + noise_cov

    elif match_variance == "signal":

        wn, vn = np.linalg.eigh(noise_cov)  # wn when scaled = wn * scaling

        # Due to signal_cov being rank 1
        # ws, vs = np.linalg.eigh(signal_cov)
        # ws1 = ws[:, -1]
        # vs1 = vs[..., -1]
        vs1 = gain / np.linalg.norm(gain, axis=0, keepdims=True)
        ws1 = signal_var

        # scale the noise covariance matrix such that the variance of the noise in
        # the direction of the signal are the same (snr = 1)
        scale_factor = np.sum(wn * vs1 @ vn) / ws1
        scale_factor_snr = scale_factor / snr ** 2
        data_cov = signal_cov * scale_factor_snr + noise_cov
        # data_cov /= scale_factor_snr
    else:
        raise ValueError

    return data_cov

    # wd, vd = np.linalg.eigh(data_cov)

    plt.figure()
    plt.imshow(data_cov_sampled)
    plt.colorbar()

    i = 0
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im, _ = mne.viz.plot_topomap(sig[i], info, axes=ax[0], show=False)
    im, _ = mne.viz.plot_topomap(noise[i], info, axes=ax[1], show=False)
    im, _ = mne.viz.plot_topomap(data[i], info, axes=ax[2], show=False)
    fig.colorbar(im, ax=ax, shrink=0.7)

    w, v = np.linalg.eigh(data_cov_sampled)
    ww, vv = np.linalg.eigh(np.cov(sig.T))
    plt.figure()
    plt.plot(v[:, -1])
    plt.plot(vv[:, -1])

    i = 0
    plt.figure()
    plt.plot(sig[i])
    plt.plot(noise[i])
    plt.plot(data[i])

    # fmt: off
    plt.figure();plt.imshow(noise_cov);plt.colorbar();
    plt.figure();plt.imshow(signal_cov);plt.colorbar();
    plt.figure();plt.imshow(data_cov);plt.colorbar();
    # fmt: on

    plt.plot(np.abs(vs[:, -1]))
    plt.plot(np.abs(vd[:, -1]))

    plt.plot(wn)
    plt.plot(ws)
    plt.plot(wd)

    the_mean = gain_ref[:, 0] / 41606945.19429742 / 3
    the_mean = 3 * gain_ref[:, 0] / gain_ref[:, 0].std() * noise_vec.std()
    multi_norm = scipy.stats.multivariate_normal(
        mean=the_mean, cov=noise_cov, allow_singular=True
    )
    vecs = multi_norm.rvs(size=100)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im, _ = mne.viz.plot_topomap(vecs[0], info, axes=ax[0], show=False)
    im, _ = mne.viz.plot_topomap(vecs.mean(0), info, axes=ax[1], show=False)
    fig.colorbar(im, ax=ax, shrink=0.85)


def compute_dipole_fit_linear(nw_gain, nw_data):
    """
    noise whitened gain and data...

    Normally, one would do a linear scan on a grid, find the position of the
    dipole which explains the data the best, and then do a nonlinear search in
    this vicinity for a better fitting dipole.
    However, here we *only* do the linear scan but on a high density surface.

    The forward solution should have free orientation in order to estimate the
    moment (orientation and strength).

    Notation

        b = data
        n = noise
        C = noise covariance matrix
        H = leadfield matrix
        x = parameters (i.e., the source moment, three component vector)

    The problem is to find the moment such that we explain the data as well as
    possible with a dipole at each position

        d = Hx + n

    The ordinary least squares solution is simply the *left* pseudo inverse
    (H is (n_sens, 3) so this exists and pinv(H) @ H == I).

    If we know the covariance of the residuals, we can use this weighting
    matrix to whiten the data such that the cov(residuals) = I, i.e., compute
    the generalized least squares solution. We estimate this covariance as the
    noise covariance (e.g., from baseline data).

    parameter estimate
    (A = C**(-1/2) @ H, i.e., whitened leadfield)
    (b = C**(-1/2) @ d, i.e., whitened data)

        x = A**(-1) @ b = (H.T @ iC @ H)**(-1) H.T @ iC @ b

    predicted data

        A @ x = wH @ x = b(predicted)


    RETURNS
    -------
    GOF :
        Goodness-of-fit with simulations in columns.

    """
    if nw_gain.ndim == 2:
        n_src, _ = nw_gain.shape
        n_ori = 1
        u = nw_gain / np.linalg.norm(nw_gain, axis=1, keepdims=True)
    elif nw_gain.ndim == 3:
        n_src, _, n_ori = nw_gain.shape
        u, _, _ = np.linalg.svd(nw_gain, full_matrices=False)
        u = u.swapaxes(1, 2)
    else:
        raise ValueError

    # Total variance of data
    var_data = np.sum(nw_data ** 2, axis=1)
    nw_data = nw_data.T

    # wG_inv = v.transpose(0,2,1) * s[:,None]**-1 @ u.transpose(0,2,1)
    # mom = np.sum(wG_inv * wd[:,None], 1)
    # wd_pred[0] = wG[0] @ mom[0]

    # variance ratio
    #  gof = np.sum(np.sum(wG * wG_inv[:, None], -1)**2, -1) / np.sum(wd**2, -1)

    # Data resolution matrix/matrices
    # R = wG @ wG**(-1) = U @ U.T
    # R = u @ u.transpose(0,2,1)

    # Creating the full array (n_src, n_ori, n_src) which we don't want
    # choose a batch size such that the large array is ~LIMIT_GB
    gof = np.zeros((n_src, n_src))
    batch_size = np.floor(n_src * LIMIT_GB / (n_src ** 2 * n_ori * 8) * 1e9).astype(int)
    batches = make_batches(n_src, batch_size)

    timer = BlockTimer(f"Computing dipole fits ({len(batches)} batches)")
    timer.start()
    for batch in batches:
        sel = slice(*batch)

        # GOF when explaining the data of the ith source using the jth dipole
        #   gof[j,i] = data[i].T @ R[j].T @ R[j] @ data[i] / data[i].T @ data[i]
        #            = data[i].T @ R[j] @ data[i] / data[i].T @ data[i]
        # and we do this for all combinations of i and j. The denominator is
        # the same for all rows as it is the variance of the simulated data.
        #
        # Note that because R = U @ U.T then
        #   R.T @ R = (U @ U.T).T @ U @ U.T = U @ U.T @ U @ U.T = U @ U.T = R

        dUUd = np.tensordot(u[sel], nw_data, axes=1) ** 2
        if n_ori == 3:
            dUUd = dUUd.sum(1)
        gof[sel] = dUUd / var_data
    timer.stop()

    return np.ascontiguousarray(gof.T)


def compute_lcmv(gain, dw, nw_gain):
    """

    LCMV does not try to explain the data but instead tries to suppress as much
    data signal as possible while also having a unit gain constraint on the
    source for which a particular filter is designed. In this sense, everything
    (noise and signal) is regarded as something to be suppressed as much as
    possible and hence the data covariance matrix is used instead of the noise
    covariance.
    It might seem unintuitive at first that the sources are being suppressed,
    however, the unit gain constraint ensures that we see the source of
    interest and if, say, there is a source at this position, then the
    surrounding sources will probably also be active and we want to suppress
    these as much as possible. On the other hand, we do not care much about
    suppressing sources which are not active since filtering a vector of (near)
    zeros will probably just result in filter output of near zeros. In this
    case, we can design our filters to suppress the sources which are actually
    present in the data.
    In principle, we can perfectly suppress n_sens/3-1 sources as there are
    n_sens DOF in a filter and each constraint uses 3 DOF (assuming the
    leadfields/transfer function of sources are linearly independent!).


    Moment of source
                                                shape
        W = inv(H.T @ Ci @ H) @ H.T @ Ci        (n_ori, n_sens)
        mom = W @ d                             (n_ori,)

    Source covariance due to data (signal + noise)

        W @ C @ W.T = inv(H.T @ Ci @ H) @ H.T @ Ci @ C @ Ci.T @ H @ inv(H.T @ Ci @ H).T
                    = inv(H.T @ Ci @ H) @ (H.T @ Ci @ H) @ inv(H.T @ Ci @ H)
                    = inv(H.T @ Ci @ H)

    Source covariance due to noise

        W @ C(noise) @ W.T = inv(H.T @ Ci(noise) @ H)

    which uses the matrix inversion lemma (see Van Veen, 1997, eq. 26).

    C = symmetric
    Ci = inv(C) = symmetric
    H.T @ Ci @ H = symmetric because Ci is symmetric


    G   : gain
    dw  : data whitener
    nwG : noise whitened gain

    """

    assert gain.shape == nw_gain.shape

    if gain.ndim == 2:
        n_src, n_sens = gain.shape
        n_ori = 1
        var_n = 1 / np.sum(nw_gain ** 2, 1)
        gainT = gain.T
    elif gain.ndim == 3:
        n_src, n_sens, n_ori = gain.shape
        s_noise = np.linalg.svd(nw_gain, compute_uv=False)
        var_n = np.sum(s_noise ** -2, 1)
        gainT = gain.transpose(1, 2, 0)
    else:
        raise ValueError

    assert dw.shape == (n_src, n_sens, n_sens)

    # Creating the full array (n_src, n_sens, n_src) which we don't want
    # choose a batch size such that the largest array takes up at most
    # `limit_GB`
    nai_power = np.zeros((n_src, n_src))
    batch_size = np.floor(
        n_src * LIMIT_GB / (n_src ** 2 * n_sens * n_ori * 8) * 1e9
    ).astype(int)
    batches = make_batches(n_src, batch_size)

    timer = BlockTimer(f"Computing LCMV solutions ({len(batches)} batches)")
    timer.start()
    for batch in batches:
        sel = slice(*batch)
        # Array shapes:
        # dw_gain = (batch_size, n_sens[, n_ori], n_src) <---- largest array
        # wD = (batch_size, n_sens, n_src)          <---- also large(st) array
        # wG2 = (batch_size[, n_ori], n_src)
        # wGwD = (batch_size[, n_ori], n_src)

        # An estimate of the dipole moment at each location is obtained by
        # applying the filter weights to the data, i.e.,
        #   filter @ D = inv(wG2) @ wGwD = (batch_size[, n_ori], n_src)
        # The dipole power/strength at each location is
        dw_gain = np.tensordot(dw[sel], gainT, axes=1)

        # wD = np.tensordot(W[s], D, axes=1)
        # wGwD = np.sum(wG * wD, 1) # i.e., G.T @ C**(-1) @ D

        if n_ori == 1:
            var_q = 1 / np.sum(dw_gain ** 2, 1)
        elif n_ori == 3:
            s_data = np.linalg.svd(dw_gain.transpose(0, 3, 1, 2), compute_uv=False)
            var_q = np.sum(s_data ** -2, 2)

            # dwGT = dw_gain.transpose(0, 3, 1, 2)
            # var_q = np.trace(
            #     np.linalg.inv(dwGT.swapaxes(2, 3) @ dwGT), axis1=2, axis2=3
            # )
        else:
            raise ValueError
        nai_power[sel] = var_q / var_n
    timer.stop()

    return np.ascontiguousarray(nai_power.T)


def compute_music(nw_gain, nw_data_cov):
    """

    The ith column is the simulation of the ith source, i.e., using the
    projector from the ith source on all leadfields (the max is the source
    to which this data is most sensitive).
    -> like point spread function?

    The ith row corresponds to applying the projectors of all sources on the
    ith leadfield (the max is the [source] data to which this source is most
    sensitive).
    -> like cross talk function?

    but the "activity" of each position is determined indepently.

    """
    # n_src = forward['n_sources']
    # fwd_mat, C = _get_white_fwd_and_cov(forward, data_cov, whitener)
    wG = nw_gain
    wC = nw_data_cov

    # Number of assumed sources/sources to find. In this case, since we have
    # one source which is much stronger than the rest, it is basically the same
    # as projecting onto the first component only
    n_dip = 5

    # Construct the signal space projector for each source
    _, U = np.linalg.eigh(wC)
    phi_sig = U[..., -n_dip:]  # eigenvector corresponding to largest eigenvalue
    # P = u @ u.swapaxes(2,1) # signal space projector

    if wG.ndim == 2:
        n_src, _ = wG.shape
        n_ori = 1
        wG_norm2_inv = 1 / np.sum(wG ** 2, 1)
        wG = wG.T
    elif wG.ndim == 3:
        # transpose...
        n_src, _, n_ori = wG.shape
        assert n_ori == 3
        Ug, Sg, VgT = np.linalg.svd(wG, full_matrices=False)
        # UgT = Ug.transpose(0, 2, 1)
        UgT = Ug.transpose(1, 2, 0)
    else:
        raise ValueError

    # EFFICIENT COMPUTATION OF MUSIC SOLUTION

    # G = wG[0]
    # phi_sig = phi_sig[0]
    # mne.beamformer._rap_music._compute_subcorr(G, phi_sig)

    # P = u @ u.T
    # P @ G = (u @ u.T) @ G = u @ (u.T @ G)

    # g = a column of G
    # P = the projector of one particular source
    # (we use that the inner product of u = I, i.e., u.T @ u = I)
    # norm(Pg)**2 = (Pg).T @ Pg = g.T @ P.T @ P @ g = g @ (uu.T).T @ uu.T @ g
    #             = g.T @ uu.T @ uu.T @ g = g.T @ u @ u.T @ g
    #             = (u.T @ g).T @ (u.T @ g)
    # Thus, we only need to compute the inner product between each u and g
    # (once), square it, and sum.
    # As long as P is not close to full rank then this way is faster, e.g., if
    # P = UU.T where U.shape = (70, 10).

    # n_ori = 1
    subcorr = np.zeros((n_src, n_src))
    # batch_size = np.floor(
    #     n_src * LIMIT_GB / (n_src ** 2 * n_ori * n_dip * 8) * 1e9
    # ).astype(int)
    batch_size = 500
    batches = make_batches(n_src, batch_size)

    timer = BlockTimer(f"Computing MUSIC solutions ({len(batches)} batches)")
    timer.start()
    for batch in batches:
        sel = slice(*batch)

        if n_ori == 1:
            subcorr[sel] = (
                np.sum(np.tensordot(phi_sig[sel].swapaxes(-2, -1), wG, axes=1) ** 2, 1)
                * wG_norm2_inv
            )
        elif n_ori == 3:
            # similar to mne.beamformer._rap_music._subcorr

            # (n_src, n_ori, n_dip, n_src)
            tmp = np.tensordot(phi_sig[sel].swapaxes(1, 2), UgT, axes=1).swapaxes(1, 2)

            # If we need directions...
            # (n_src, n_src, n_ori, n_ori)
            # Uc, Sc, _ = np.linalg.svd(tmp.transpose(0, 3, 1, 2), full_matrices=False)

            # If we only need subspace correlations...
            Sc = np.linalg.svd(tmp.transpose(0, 3, 1, 2), compute_uv=False)

            subcorr[sel] = Sc[..., 0]

            # Orientation of maximal subspace correlation

            # (n_src, n_ori, n_ori)
            # p = VgT.transpose(0, 2, 1) / Sg[:, None, :]
            # X = np.squeeze(p[None] @ Uc[..., :1])
            # (n_src, n_src, n_ori)
            # e.g., [0,1,:] is the orientation obtained when reconstructing source 0
            # using gain matrix columns associated with source 1.
            # X /= np.linalg.norm(X, axis=2, keepdims=True)

            # Sign is ambiguous so align with surface normal...
            # ---> get the surface normal...
            # X *= np.sign(np.sum(X * surf_normal[sel], 1)) or 1
    timer.stop()

    return np.ascontiguousarray(subcorr.T)


def _get_gain_matrix(fwd):
    bads = fwd["info"]["bads"]
    ch_names = [c for c in fwd["info"]["ch_names"] if (c not in bads)]
    fwd = mne.pick_channels_forward(fwd, ch_names, ordered=True)
    gain = fwd["sol"]["data"].copy()
    if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
        gain = gain.reshape(fwd["nchan"], fwd["nsource"], 3).transpose(1, 0, 2)
        gain = np.ascontiguousarray(gain)
    else:
        gain = gain.T
    return gain


def make_resolution_and_inverse_matrix(fwd_mat, inv, method, lambda2):
    """Forward and inverse should match.

    Modified from _get_matrix_from_inverse_operator


    fwd_mat :
        Noisy reference gain matrix (i.e., the simulated sources).

        (n_src, n_sens)
    inv :
        The inverse operator
    """
    # fwd = mne.minimum_norm.resolution_matrix._convert_forward_match_inv(forward, inv)
    # inv_mat = mne.minimum_norm.resolution_matrix._get_matrix_from_inverse_operator(
    #     inv, fwd, method=method, lambda2=lambda2
    # )

    inv_info = mne.minimum_norm.resolution_matrix._prepare_info(inv)  # adds sfreq
    bads_inv = inv_info["bads"]
    ch_names = inv_info["ch_names"]
    n_chan = len(ch_names)
    ch_idx_bads = [ch_names.index(ch) for ch in bads_inv]

    # create identity matrix as input for inverse operator
    # set elements to zero for non-selected channels
    id_mat = np.eye(n_chan)

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = mne.EvokedArray(id_mat, info=inv_info, tmin=0.0)

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combine components

    # check if inverse operator uses fixed source orientations
    is_fixed_inv = mne.minimum_norm.resolution_matrix._check_fixed_ori(inv)

    # choose pick_ori according to inverse operator
    pick_ori = None if is_fixed_inv else "vector"

    # columns for bad channels will be zero
    invmat_op = mne.minimum_norm.apply_inverse(
        ev_id, inv, lambda2=lambda2, method=method, pick_ori=pick_ori
    )
    # turn source estimate into numpy array
    invmat = invmat_op.data

    # remove columns for bad channels
    # take into account it may be 3D array
    invmat = np.delete(invmat, ch_idx_bads, axis=invmat.ndim - 1)

    # if 3D array, i.e. multiple values per location (fixed and loose),
    # reshape into 2D array
    if invmat.ndim == 3:
        v0o1 = invmat[0, 1].copy()
        v3o2 = invmat[3, 2].copy()
        shape = invmat.shape
        invmat = invmat.reshape(shape[0] * shape[1], shape[2])
        # make sure that reshaping worked
        assert np.array_equal(v0o1, invmat[1])
        assert np.array_equal(v3o2, invmat[11])

    res_mat = invmat @ fwd_mat.T
    if pick_ori == "vector":
        print("Pooling orientations")
        res_mat = mne.minimum_norm.inverse.combine_xyz(res_mat)
    return res_mat


# Plot effect of adding smoothness and noise...
# i = 16899
# snr = 10

# idx = source_activity.indices[
#     source_activity.indptr[i] : source_activity.indptr[i + 1]
# ]
# fwd_mat1 = np.sum(fwd_mat[:, idx] * source_activity[:, i].data, 1)
# arrays = [
#     fwd_mat[:, i],
#     fwd_mat1,
#     noise_dict[snr][:, i],
#     fwd_mat1 + noise_dict[snr][:, i],
# ]
# titles = ["clean", "clean smooth", "noise", "noisy smooth"]

# kw = dict(vmin=fwd_mat[:, i].min(), vmax=fwd_mat[:, i].max(), show=False)
# fig, axes = plt.subplots(1, 4, constrained_layout=True, figsize=(15, 4))
# for arr, ax, title in zip(arrays, axes, titles):
#     im, _ = mne.viz.plot_topomap(arr, info, axes=ax, **kw)
#     ax.set_title(title)
# cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.025)
# fig.show()


def unit_covariance_size(inv_mat, inv):
    """Source variance."""
    # i.e., np.trace(inv_mat @ inv_mat.T)
    return np.sum(inv_mat ** 2)
    return np.sum(np.sum(inv_mat ** 2, axis=1) * inv["source_cov"]["data"])


def dirichlet_spread(res_mat):
    # resmat - np.eye(*resmat.shape)
    return np.linalg.norm(res_mat - np.diag(res_mat)) ** 2


def backusgilbert_spread(res_mat, dist_mat):
    return np.linalg.norm(res_mat * dist_mat) ** 2


def visualize_resolution_matrix(resmat):
    vmin, vmax = (
        np.array([-1, 1]) * np.abs(np.percentile(resmat.ravel(), [5, 95])).max()
    )
    print((vmin, vmax))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(resmat, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    # plt.colorbar();


# fwd = mne.read_forward_solution(io.data.get_filename(stage='forward',forward='template_nonlin',suffix='fwd', extension='fif'))
# fwd = mne.convert_forward_solution(fwd, force_fixed=True)
# print(np.linalg.norm(fwd['sol']['data']))
# y = np.linalg.norm(fwd['sol']['data'],axis=0)

# im = 'MNE'
# fig,axes = plt.subplots(3, 2, sharex='col', sharey='all',
#         constrained_layout=True, figsize=(10,10))
# for i,(row,snr) in enumerate(zip(axes, (3,10,25))):
#     for ax,met,title in zip(row, (topo, magn), ('rdm', 'lnmag')):
#         for model in Config.forward.MODELS[1:]:
#             ax.hist(met[model, im, snr], 'auto', histtype='step')
#         if i == 2:
#             ax.set_xlabel(title)
#     row[0].set_ylabel(f'SNR = {snr}')

# fm = 'template_nonlin'
# snr = 25
# plt.scatter(df.loc['RDM', fm, '15', 'normal'].to_numpy(), topo[fm, 'MNE', snr]);
# plt.plot([0,1.5],[0,1.5],alpha=0.5,color='gray')
# plt.xlim([0, 1.5]);
# plt.ylim([0, 1.5]);

# fm = 'custom_nonlin'
# fig,axes=plt.subplots(2,3,sharey='row',constrained_layout=True,figsize=(12,8))
# axes[0,0].scatter(df.loc['RDM', fm, '15', 'normal'].to_numpy(), topo[fm, 'MNE', 3])
# axes[0,1].scatter(df.loc['RDM', fm, '15', 'normal'].to_numpy(), topo[fm, 'MNE', 10])
# axes[0,2].scatter(df.loc['RDM', fm, '15', 'normal'].to_numpy(), topo[fm, 'MNE', 25])
# axes[1,0].scatter(df.loc['lnMAG', fm, '15', 'normal'].to_numpy(), magn[fm, 'MNE', 3])
# axes[1,1].scatter(df.loc['lnMAG', fm, '15', 'normal'].to_numpy(), magn[fm, 'MNE', 10])
# axes[1,2].scatter(df.loc['lnMAG', fm, '15', 'normal'].to_numpy(), magn[fm, 'MNE', 25])

# the_max = max(axes[0,0].get_xlim()[-1], axes[0,0].get_ylim()[-1])
# for ax in axes[0]:
#     ax.set_xlim([0, the_max])
#     ax.set_ylim([0, the_max])
#     ax.plot([0, the_max], [0, the_max], color='gray', alpha=0.5)


# for k,v in magn.items():
#     mb['lh'][f'mag {k}'] = v[:10242]
#     mb['rh'][f'mag {k}'] = v[10242:]
# for k,v in topo.items():
#     mb['lh'][f'rdm {k}'] = v[:10242]
#     mb['rh'][f'rdm {k}'] = v[10242:]

# mb.save('/home/jesperdn/nobackup/src.vtm')


# This is the correct order!
# combs = tuple((*i, *j) for i, j in itertools.product(model_comb, res_comb))
# n_times = len(combs)
# with open(io.data.path.get() / "models_at_time_index.csv", "w") as f:
#     csvwriter = csv.writer(f)
#     for i in combs:
#         csvwriter.writerow(i)
# for fm, snr, im in model_comb:
#     io.data.update(forward=fm, **kw_reset)
#     morph = mne.read_source_morph(io.data.get_filename(**kw_morph))
#     io.data.update(snr=snr, inverse=im)
#     inv = mne.minimum_norm.read_inverse_operator(
#         io.data.get_filename(**kw_inv)
#     )
#     res_matrix = mne.minimum_norm.make_inverse_resolution_matrix(
#         fwd_ref, inv, im, snr ** -2
#     )
#     data = np.zeros((n_src, n_res_comb))
#     for i, (function, metric) in enumerate(res_comb):
#         res_metric = mne.minimum_norm.resolution_metrics(
#             res_matrix, inv["src"], function, metric
#         )
#         data[:, i] = res_metric.data
#     vertno = [s["vertno"] for s in inv["src"]]
#     stc = mne.SourceEstimate(data, vertno, tmin=0.0, tstep=1.0)
#     stc = morph.apply(stc)
#     stc.save(io.data.get_filename(extension=None))

# for function in Config.resolution.FUNCTIONS:
#     io.data.update(res_function=function)
#     for metric in Config.resolution.METRICS:
#         res_metric = mne.minimum_norm.resolution_metrics(
#             res_matrix, inv["src"], function, metric
#         )
#         Apply morph (to fsaverage)
#         res_metric = morph.apply(res_metric)
#         res_metric.save(
#            io.data.get_filename(res_metric=metric, extension=None)
#         )

# vertno = [s["vertno"] for s in inv["src"]]
# stc = mne.SourceEstimate(data, vertno, tmin=0.0, tstep=1.0)
# stc = morph.apply(stc)
# stc.save(io.data.get_filename())


# proj = mne.io.proj.make_eeg_average_ref_proj(inv["info"])
# fwd_ref['info']['projs'] = [proj]
# inv["projs"] = [proj]

