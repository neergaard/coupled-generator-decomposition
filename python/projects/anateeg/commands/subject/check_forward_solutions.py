import sys

import mne
import numpy as np
import pyvista as pv

from simnibs.simulation import eeg

from projects.anateeg import utils
from projects.anateeg.config import Config


def check_forward_solutions(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="forward", space="fsaverage", suffix="fwd")

    fsavg = eeg.FsAverage(Config.inverse.FSAVERAGE)
    fsavg_c = fsavg.get_central_surface()

    points = np.concatenate([i["points"] for i in fsavg_c.values()])
    add_index = np.cumsum([0] + [len(i["points"]) for i in fsavg_c.values()])[:-1]
    tris = np.concatenate([i["tris"] + j for i, j in zip(fsavg_c.values(), add_index)])

    pd = pv.make_tri_mesh(points, tris)

    data = {}
    for model in Config.forward.MODELS:
        fwd = mne.read_forward_solution(io.data.get_filename(forward=model))

        data[model] = fwd["sol"]["data"].reshape(fwd["sol"]["nrow"], -1, 3)

        # data[model] = mne.convert_forward_solution(
        #    fwd["sol"]["data"].reshape(fwd["sol"]["nrow"], -1), force_fixed=True
        # )

    # fwd = mne.read_forward_solution("/mrhome/jesperdn/nobackup/sub-01/fwd.fif")
    # data["ft_with_simnibs"] = fwd["sol"]["data"].reshape(fwd["sol"]["nrow"], -1, 3)

    # data = {k: data[k] for k in sorted(data)}

    vmin, vmax = np.percentile(np.stack(list(data.values())), [1, 99])
    scalar_bar_kw = dict(clim=[vmin, vmax], below_color="blue", above_color="red")

    n_models = len(data)
    n_chan = fwd["nchan"]
    ch_pos = 1e3 * np.row_stack([ch["loc"][:3] for ch in fwd["info"]["chs"]])
    # channels = pv.PolyData(ch_pos)

    for i in range(n_chan):
        p = pv.Plotter(shape=(3, n_models), window_size=(n_models * 300, 3 * 300))
        for col, (m, d) in enumerate(data.items()):
            for row, axis in enumerate(["x", "y", "z"]):
                p.subplot(row, col)
                if row == 0:
                    p.add_text(m)
                if col == 0:
                    p.add_text(axis, "left_edge")
                # ch_indicator = np.zeros(n_chan, dtype=int)
                # ch_indicator[i] = 1
                # p.add_mesh(channels, scalars=ch_indicator)

                p.add_mesh(pd.copy(), scalars=d[i, :, row], **scalar_bar_kw)

        p.link_views()
        p.show()

        # input("Press key to continue...")

        # p.close()


if __name__ == "__main__":
    args = utils.parse_args(sys.argv)
    check_forward_solutions(getattr(args, "subject-id"))

