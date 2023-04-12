import pandas as pd

from simnibs.simulation import eeg_viz

from projects.mnieeg import evaluation_viz_surf
from projects.anateeg.config import Config

suffix =  evaluation_viz_surf.suffix


def forward_plot():
    output_dir = Config.path.RESULTS / "figure_forward"
    if not output_dir.exists():
        output_dir.mkdir()

    df_fname = Config.path.RESULTS / "forward.pickle"
    df = pd.read_pickle(df_fname)
    # FieldTrip matched conductivities
    dff = pd.read_pickle(Config.path.RESULTS / "forward_match_fieldtrip.pickle")
    dft1 = pd.read_pickle(Config.path.RESULTS / "forward_charm_t1_only.pickle")

    fig = evaluation_viz_surf.fwd_plot_density(pd.concat([df, dff, dft1], axis=1))
    fig.savefig(output_dir / f"forward_density{suffix}")

    fig = evaluation_viz_surf.fwd_plot_crf(pd.concat([df, dff, dft1], axis=1))
    fig.savefig(output_dir / f"forward_crf{suffix}")

    dfs = evaluation_viz_surf.get_summary_statistics(
        df, ["Metric", "Forward", "Orientation"]
    )

    brain = eeg_viz.FsAveragePlotter(Config.inverse.FSAVERAGE, "central")
    for metric in dfs.columns.unique("Metric"):
        fig = evaluation_viz_surf.fwd_plot_stat_on_surface(
            brain, dfs, metric, cbar_rows=1
        )
        fig.savefig(output_dir / f"forward_source_{metric}{suffix}.png")

    # Distance matrices
    df_fname = Config.path.RESULTS / "forward_distance_matrix.pickle"
    df = pd.read_pickle(df_fname)

    fig = evaluation_viz_surf.fwd_plot_distance_matrix(df)
    fig.savefig(output_dir / f"forward_distance_matrices{suffix}")


if __name__ == "__main__":
    forward_plot()
