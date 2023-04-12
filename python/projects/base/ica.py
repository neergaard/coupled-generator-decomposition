"""

"""
import numpy as np

def unmix(ica, data):
    """Project measurements to sources

        S = WX = A^(-1)X

    PARAMETERS
    ----------
    ica : mne ica object
    data : ndarray

    """
    # Number of principal components used for the ICA fit
    npc = ica.n_components_

    # Prewhiten
    if ica.noise_cov is None:
        data = data/ica.pre_whitener_
    else:
        data = ica.pre_whitener_ @ data

    # Center per channel (prior to PCA)
    if ica.pca_mean_ is not None:
        data -= ica.pca_mean_[:,None]

    # Project to PCA components and unmix
    sources = ica.unmixing_matrix_ @ ica.pca_components_[:npc] @ data

    return sources

def mix(ica, sources):
    """Project sources to measurements

        X = AS

    """
    npc = ica.n_components_

    # (Re)mix sources and project back from PCA space
    data = ica.pca_components_[:npc].T @ ica.mixing_matrix_ @ sources

    # Undo centering
    if ica.pca_mean_ is not None:
        data += ica.pca_mean_[:,None]

    # Undo prewhitening to restore scaling of data
    if ica.noise_cov is None:
        data *= ica.pre_whitener_
    else:
        data = np.linalg.pinv(ica.pre_whitener_) @ data

    return data
