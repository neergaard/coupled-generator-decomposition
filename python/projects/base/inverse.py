

n = 10 # number of components/sources to find

X = np.random.random((60, 100)) # elec x time
G = np.random.random((60, 1000, 3)) # elec x sources



epochs = mne.read_epochs(r'C:\Users\jdue\Documents\phd_data\wakeman2015_bids_analysis\sub-01\ses-meg\preprocessing\sub-01_ses-meg_proc-pa_epo.fif')
epochs = epochs['face/famous']
epochs.pick_types(eeg=True)
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0, method='shrunk')
#noise_cov = noise_cov['data']
#data_cov = mne.compute_covariance(epochs, tmin=0, tmax=None, method='shrunk')
#data_cov = data_cov['data']
evoked = epochs.average()

forward = mne.read_forward_solution(r'C:\Users\jdue\Documents\phd_data\wakeman2015_bids_analysis\sub-01\ses-meg\forward\sub-01_ses-meg_proc-simnibs_fwd.fif')
src = forward['src']

#forward = mne.convert_forward_solution(forward, force_fixed=True)


is_free_ori, info, _, _, G, whitener, _, _ = mne.beamformer._compute_beamformer._prepare_beamformer_input(
    epochs.info, forward, noise_cov=noise_cov, rank=None)

X = whitener @ evoked.data

data_cov = X @ X.T


#G = G['sol']['data']
#G /= np.linalg.norm(G, axis=0)


sources = np.zeros((60000, 140)).T
source_orientation = np.random.random(3)
source_orientation /= np.linalg.norm(source_orientation)
sources[:,:3] = 10 * source_orientation
sources = sources.T

X = G @ sources

#X += 10 * np.random.random((70,140))

data_cov = np.cov(X)

sources, maps = TRAP_MUSIC(C, G, 10)

def TRAP_MUSIC(G, X, n, music_type='vector'):
    """
    U :
        Data covariance matrix
    G :
        Gain (leadfield) matrix
    n : int
        Number of sources to find.
    return_maps : bool
        Whether or not to return the full localization maps of each recursion.
    music_type :


    REFERENCES
    ----------
    Mosher (1999).
    Makela (2018).

    """
    if music_type == 'scalar':
        n_sens, n_sources = G.shape
    elif music_type == 'vector':
        G = G.reshape(len(G), -1, 3).transpose(1, 0, 2)
        n_sources, n_sens, n_ori = G.shape
        source_orientations = np.zeros((n_targets, 3))
    else:
        raise ValueError
    source_positions = np.zeros(n_targets, dtype=np.int)
    source_topographies = np.zeros((n_targets, n_sens))
    subcorrs = np.zeros((n_targets, n_sources))

    # Q = I in first iteration
    QG = G
    QU = X

    #QU = data_cov
    #_, U = np.linalg.eigh(data_cov)
    # Eigenvalues/-vectors are sorted in ascending order so reverse
    #U = U[:, ::-1]
    #U = U[:, :n_targets-k] # Us, The truncation

    for k in range(n_targets):
        # The signal space projector, U
        U, _, _ = np.linalg.svd(QU, full_matrices=False)
        U = U[:, :n_targets-k] # Us, The truncation

        if music_type == 'scalar':
            # Mosher (1999) eq. 2; Makela (2018) eq. 3
            QGs = U.T @ QG
            subcorr = np.sum(QGs**2, axis=0) / np.sum(QG**2, axis=0)
            subcorr = np.sqrt(subcorr)
            source_ix = subcorr.argmax()
            # orientations are already known
            subcorrs[k] = subcorr
            source_topographies[k] = QG[source_ix]

        if music_type == 'vector':
            # Mosher (1999)
            # SVD of G on a per source basis
            Ug, Sg, Vgh = np.linalg.svd(QG, full_matrices=False)
            #Vg = Vg.transpose(0, 2, 1)

            # The straight forward approach is to first form the (squared)
            # subspace correlation matrix and then, using the eigenvalue
            # decomposition, find the maximum correlation with the subspace
            # (maximum eigenvalue) and its corresponding orientation
            # (corresponding eigenvector).
            # Mosher (1999) eq. 6

            #subcorr2 = QG.transpose(0, 2, 1) @ U @ U.T @ QG / (QG.transpose(0, 2, 1) @ QG)

            #subcorr2 = Ug.transpose(0, 2, 1) @ U @ U.T @ Ug
            # The first eigenvalue is the localizer value and the associated
            # eigenvector the direction which maximizes the correlation with the
            # subspace
            #D, E = np.linalg.eigh(subcorr2)
            ##source_ix = np.sqrt(D[:, -1]) # in ascending order!
            #source_ori = E[:, -1] # orientation of dipole

            # Alternativelty, instead of forming the squared subspace
            # correlation matrix explicitly, we can use the SVD
            _, Sc, Vch = np.linalg.svd(U.T @ Ug, full_matrices=False)
            source_ix = Sc[:, 0].argmax()
            # Transform Ug back to QG space (Ug = QG @ Vgh.T / Sg)
            source_ori = Vgh[source_ix].T / Sg[source_ix] @ Vch[source_ix, 0]
            source_ori /= np.linalg.norm(source_ori)
            subcorrs[k] = Sc[:, 0]
            source_topographies[k] = QG[source_ix] @ source_ori

        source_positions[k] = source_ix
        if music_type == 'vector':
            source_orientations[k] = source_ori

        if k < n_targets-1:
            # Project out the topography of the source we just found
            # Out-projector for current iteration
            # (identity in first iteration)
            # Makela (2018) eq. 5
            # Mosher (1999) eq. 9


            A = source_topographies[:k+1].T

            #Ua, _, _ = np.linalg.svd(A)
            #Q = np.identity(n_sens) - Ua @ Ua.T
            # Is this stable?
            Q = np.identity(n_sens) - (A @ np.linalg.inv(A.T @ A) @ A.T)

            # Apply
            QG = Q @ G
            QU = Q @ U


    # Undo whitening
    source_topographies


    np.linalg.norm(U.T @ QG[source_ix] @ a) / np.linalg.norm(QG[source_ix] @ a)

    b = np.array([-0.0013705 , -0.48112807,  0.87664925])
    b = E[source_ix][2]
    b /= np.linalg.norm(b)

    b = np.array([-0.030731514969695308 -0.560466105160755 0.8276069833886135])
    b = source_ori
    np.linalg.norm(U.T @ QG[source_ix] @ b) / np.linalg.norm(QG[source_ix] @ b)



    # Estimate dipole time course
    amplitude = np.linalg.lstsq(source_topographies, data)[0]

    if music_type == 'scalar':
        return source_positions, subcorrs
    elif music_type == 'vector':
        return source_positions, source_orientations, subcorrs


source 1 found: p = 1514
ori = 0.34755524036185215 -0.7804858923528837 0.5196605880140788
source 2 found: p = 13043
ori = -0.030731514969695308 -0.560466105160755 0.8276069833886135
source 3 found: p = 11901
ori = 0.3485411879002482 -0.9164398567424716 0.19661390925090164
source 4 found: p = 12005
ori = 0.2862089350053678 -0.7717482136655002 0.5678812730027314
source 5 found: p = 2254
ori = -0.34275796326668956 -0.931979974544578 0.11802671589584383
    Explained  45.4% variance

        # G.T @ Q.T @ P.T @ P @ Q @ G
        # A.T @ P.T @ P @ A with A = QG
        # canonical (subspace) correlation
        A = Q @ G
        A.T @ P @ A
        mu = np.linalg.norm(P @ Q @ G, axis=0)**2 / np.linalg.norm(Q @ G, axis=0)**2 # Localizer map
        localizer[k] = mu
        sources.append(mu.argmax())
    sources = np.asarray(sources)
    return sources, localizer
