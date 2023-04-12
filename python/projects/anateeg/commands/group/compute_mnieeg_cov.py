import mne
import numpy as np

from projects.anateeg.config import Config
from projects.mnieeg import utils


def reduce_covs(covs):

    cov = np.stack([c["data"] for c in covs])
    ch_names = covs[0].ch_names  # all should be equal
    projs = covs[0]["projs"]  # all should be equal
    nfree = np.round(np.mean([c["nfree"] for c in covs])).astype(int)

    # ch_names_ = [
    #     ch_name if i % 2 == 0 else "" for i, ch_name in enumerate(ch_names, start=1)
    # ]
    # d = cov.diagonal(axis1=1, axis2=2)
    # q1, q3 = np.quantile(d, [0.25, 0.75], 0)
    # iqr = q3 - q1
    # limit = q3 + 1.5 * iqr
    # fig, ax = plt.subplots(figsize=(12, 6))
    # box = ax.boxplot(d, labels=ch_names_)

    avg_cov = cov.mean(0)

    _, s, v = np.linalg.svd(cov.reshape(cov.shape[0], -1), full_matrices=False)
    v = v.T
    first_comp = v[:, 0].reshape(cov.shape[1:]) * s[0]

    n = np.prod(cov.shape[1:])
    s1 = np.sum(np.sign(first_comp) == 1) / n > 0.5
    s2 = np.sum(np.sign(avg_cov) == 1) / n > 0.5

    first_comp = first_comp if s1 and s2 or not s1 and not s2 else -first_comp

    avg_cov = mne.Covariance(first_comp, ch_names, [], projs, nfree)

    return avg_cov


def compute_mnieeg_cov():
    io = utils.GroupIO()
    io.data.update(stage="preprocessing")

    covs = dict(forward=[], inverse=[])
    for subject, session in io.subjects.items():
        io.data.update(subject=subject, session=session)
        for k, v in covs.items():
            this_cov = mne.read_cov(io.data.get_filename(space=k, suffix="cov"))
            v.append(this_cov)

    for k, v in covs.items():
        this_cov = reduce_covs(v)
        mne.write_cov(
            Config.path.RESOURCES / f"mnieeg_group_space-{k}_cov.fif", this_cov
        )


if __name__ == "__main__":
    compute_mnieeg_cov()
