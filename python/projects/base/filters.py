import numpy as np
from sklearn.preprocessing import StandardScaler

def recursive_least_squares(signal, reference, M=3, ff=0.99, sigma=0.01, pad="signal",
        return_weights=False, inplace=False):
    """Recursive least squares.

    Clean "signal" based on "reference".

    Stationarity. RLS assumes a stationary (i.e., a stabil/consistent)
    relationship between the reference inputs (i.e., "reference") and the
    expression of these in the signal.


    PARAMETERS
    ----------
    signal : array_like
        Noisy N-dimensional signal vector.
    reference : array_like
        Reference signal array based on which "signal" is cleaned. Dimensions are L x N with L being the number of signals used
        to clean the input signal.
    M : int
        Length of finite impulse response filter. This also denotes the first sample which will be filtered (default = 3).
    ff : float
        Forgetting factor (typically lambda) bounded between 0 and 1 (default = 0.99).
    sigma : float
        Factor used for initialization of the (inverse) sample covariance matrix (Ri = I/sigma).
    pad : "signal" | "zeros" | None
        How to pad signal to obtained an output vector of same length as input
        signal. If None, do not pad signal. If "zeros", pad with zeros. If
        "signal", pad with input signal. If "inplace" is true, this has no
        effect (default = "signal").
    return_weights : bool
        Return the history of the weights (default = False).
    inplace : bool
        Whether or not to modify the input array inplace (default = False).

    RETURNS
    ----------
    err : numpy.array
        The  signal array with the reference signal regressed out.
    Hs : numpy.array (optional)
        Array of weights.

    NOTES
    ----------
    He P., Wilson G., Russell C. 2004. Removal of ocular artifacts from electro-encephalogram by adaptive filtering.
    Med Biol Eng Comput. 2004 May 42(3):407-12.
    """
    assert (ff<=1) and (ff>=0), "Forgetting factor must be 0 =< ff =< 1"
    assert pad in [None, "zeros", "signal"]

    try:
        Nsig, Ns = signal.shape
    except ValueError:
        signal = signal[None,:]
        Nsig, Ns = signal.shape
    try:
        Nref, Nr = reference.shape
    except ValueError:
        reference = reference[None,:]
        Nref, Nr = reference.shape

    assert Ns == Nr, "Length of signal and reference signal(s) must be equal"
    N = Ns
    assert N > M, "Length of signal and reference signal(s) must be larger than M"

    # Standardize
    signal = signal.T
    signal_scaler = StandardScaler()
    signal_scaler.fit(signal)
    signal = signal_scaler.transform(signal).T

    reference = reference.T
    reference_scaler = StandardScaler()
    reference_scaler.fit(reference)
    reference = reference_scaler.transform(reference).T

    # Initialize weights (flattened to vector)
    Wp = np.zeros( (Nsig, Nref*M) )

    # Initialize (inverse) reference covariance
    # R   : the weighted covariance matrix of the reference signals
    # Ri  : R(i)^(-1)
    # Rip : R(i-1)^(-1)
    Rip = np.eye(Nref*M,Nref*M)/sigma
    Rip = np.repeat(Rip[None,...], Nsig, axis=0)

    if not inplace:
        err = np.zeros( (Nsig, N-M) )

    if return_weights:
        Ws = np.zeros((Nsig, N-M, Nref*M))

    # Start from the Mth sample
    for i in np.arange(M,N):
        # Eq. 23 : stack the reference signals column-wise
        r = reference[:,i-M:i].ravel()

        # Eq. 25 : calculate gain factor
        K = Rip @ r / (ff + r @ Rip @ r)[:,None]

        # Eq. 27 : a priori error (i.e., using the previous weights)
        alpha = signal[:,i] - Wp @ r

        # Eq. 26 : the correction factor is directly proportional to the gain vector, K, and the (a priori) error, alpha
        W = Wp + K*alpha[:,None]

        # Eq. 24 : update the (inverse) covariance matrix
        Ri = (Rip - np.outer(K, r).reshape(Nsig, Nref*M, Nref*M) @ Rip ) / ff # np.outer ravels all inputs so reshape

        # A posteriori error (i.e., using the updated weights). This is the cleaned signal
        if inplace:
            signal[:,i] -= W @ r
        else:
            err[:,i-M] = signal[:,i] - W @ r

        if return_weights:
            # Collect weights
            Ws[:,i-M,:] = W

        # Prepare for next iteration
        Rip = Ri
        Wp = W

    # Convert signal back to original scale
    signal = signal_scaler.inverse_transform(signal.T).T

    if inplace:
        if return_weights:
            return signal, Ws
        else:
            return signal
    else:
        # Convert error signal back to original scale
        err = signal_scaler.inverse_transform(err.T).T
        if pad == "signal":
            err = np.concatenate((signal[:,:M], err), axis=1)
        elif pad == "zeros":
            err = np.concatenate((np.zeros(Nsig,M), err))
        if return_weights:
            return err, Ws
        else:
            return err