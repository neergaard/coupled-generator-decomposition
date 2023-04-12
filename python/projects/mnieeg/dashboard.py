


#%%

import bokeh.plotting
from bokeh.palettes import d3
import panel as pn
import param

import pyvista as pv

pn.extension("vtk")
pv.set_plot_theme("document")

import pandas as pd

import numpy as np

dfs = pd.read_pickle(r"C:\Users\jdue\Desktop\dashboard\mnieeg\inverse_summary.pickle")
dfd = pd.read_pickle(r"C:\Users\jdue\Desktop\dashboard\mnieeg\inverse_density.pickle")

surface = pv.read(r"C:\Users\jdue\Desktop\dashboard\fsaverage5.vtm")


keys = ["Orientation", "Statistic", "Inverse", "Resolution Function", "Resolution Metric"]
options = {k:dfs.columns.unique(k).to_list() for k in keys}

#%%

class Dashboard(param.Parameterized):

    # grid = 4x5

    def __init__(self, surface, dfs, dfd, **params):
        super().__init__(**params)

        self.surface = surface
        self.hemispheres = self.surface.keys()
        self.df_summary = dfs
        self.df_density = dfd
        self.forwards = self.df_density.columns.unique("Forward")
        self.snrs = self.df_density.columns.unique("SNR")
        self.n_forwards = len(self.forwards)
        self.n_snrs = len(self.snrs)


        self.width = 1200
        self.height = 800
        self.surface_plot_shape = (self.n_snrs, self.n_forwards)
        self.surface_plot_size = tuple(400 * i for i in self.surface_plot_shape[::-1])

        self.gspec = pn.GridSpec(
            mode="override",
            sizing_mode="stretch_both",
            max_width=self.width,
            max_height=self.height,
        )
        # self.gspec.servable()
        self.gspec[:,0] = pn.Column(self.param)

    # Inputs ...
    orientation = param.ObjectSelector(
        default=options['Orientation'][0], objects=options['Orientation']
    )
    statistic = param.ObjectSelector(
        default=options['Statistic'][0], objects=options['Statistic']
    )
    inverse = param.ObjectSelector(
        label="Inverse Method", default=options['Inverse'][0], objects=options['Inverse']
    )
    function = param.ObjectSelector(
        default=options["Resolution Function"][0], objects=options["Resolution Function"]
    )
    metric = param.ObjectSelector(default=options['Resolution Metric'][0], objects=options['Resolution Metric'])

    # gspec = pn.GridSpec(sizing_mode="stretch_both", max_height=800)

    @param.depends("orientation", "statistic", "inverse", "function", "metric")
    def make_surface_plots(
        self,
        cbar_pad=0.1,
        cbar_size="2.5%",
        cbar_rows=None,
    ):
        """

        px : pixels per subplot

        """
        cmap = "RdBu_r"
        clim = [0, 6]

        zoom_factor = np.sqrt(2)

        plotter_kwargs = dict(
            cmap = cmap,
            clim = clim,
            scalar_bar_args = dict(
                vertical=False, n_labels=3, label_font_size=10
            )
        )

        p = pv.Plotter(shape=self.surface_plot_shape, window_size=self.surface_plot_size, notebook=False, border=False)
        for i, snr in enumerate(self.snrs):
            for j, fwd in enumerate(self.forwards):
                p.subplot(i, j)
                scalars = self.df_summary[self.statistic, self.orientation, fwd, self.inverse, snr, self.function, self.metric]
                for h in self.hemispheres:
                    p.add_mesh(self.surface[h].copy(), scalars=scalars[h])
                p.camera.zoom(zoom_factor)

                if j == 0:
                    p.add_text(str(snr), "left_edge", font_size=12)
                if i == 0:
                    p.add_text(fwd, "upper_edge", font_size=12)

        p.link_views()
        # return p

        self.gspec[:self.n_snrs, 1:self.n_forwards] = pn.panel(p.ren_win)
        return self.gspec


    @param.depends("orientation", "statistic", "inverse", "function", "metric")
    def make_density_plots(self):
        # fig1, axes = plt.subplots(1, 4, sharey=True, constrained_layout=True)
        # for ax in axes:
        #     ax.plot(np.random.rand(10))

        # fig2, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True)
        # for ax in axes:
        #     ax.plot(np.random.rand(10))
        # self.gspec[3, :4] = pn.pane.Matplotlib(fig1)
        # self.gspec[:3, 4] = pn.pane.Matplotlib(fig2)

        # sel = pd.IndexSlice[self.orientation, :, self.inverse, :, self.function, self.metric]
        # df_sel = self.df_density.loc[:, sel]

        # Forward densities
        ymax = 0
        figs = []
        for i,fwd in enumerate(self.forwards):
            fig = bokeh.plotting.figure(toolbar_location=None, width=200, height=200)

            if i == self.n_forwards-1:
                for j,snr in enumerate(self.snrs):
                    data = self.df_density[self.orientation, fwd, self.inverse, snr, self.function, self.metric]
                    ymax = max(ymax, data.max())
                    fig.line(self.df_density.index, data, color=d3["Category10"][self.n_snrs][j], legend_label=str(snr))
                fig.legend.label_text_font_size = "10pt"
            else:
                for j,snr in enumerate(self.snrs):
                    data = self.df_density[self.orientation, fwd, self.inverse, snr, self.function, self.metric]
                    ymax = max(ymax, data.max())
                    fig.line(self.df_density.index, data, color=d3["Category10"][self.n_snrs][j])
            if i > 0:
                fig.yaxis.major_label_text_font_size = '0pt'
            figs.append(fig)
        # figs[0].yaxis.axis_label = "Probability Density"
        for fig in figs:
            fig.y_range.end = ymax

        self.gspec[self.n_snrs, 1:self.n_forwards+1] = pn.Row(*figs)

        # SNR densities
        figs = []
        for i,snr in enumerate(self.snrs):
            fig = bokeh.plotting.figure(toolbar_location=None, width=200, height=200)
            # d = self.df_density.loc[:, pd.IndexSlice[self.orientation, :, self.inverse, snr, self.function, self.metric]].to_numpy().T.tolist()
            # fig.multi_line([self.df_density.index for i in range(self.n_forwards)], d, line_color=d3["Category10"][self.n_forwards])
            if i == self.n_snrs-1:
                for j,fwd in enumerate(self.forwards):
                    fig.line(self.df_density.index, self.df_density[self.orientation, fwd, self.inverse, snr, self.function, self.metric], color=d3["Category10"][self.n_forwards][j], legend_label=fwd)
                fig.legend.label_text_font_size = "10pt"
            else:
                for j,fwd in enumerate(self.forwards):
                    fig.line(self.df_density.index, self.df_density[self.orientation, fwd, self.inverse, snr, self.function, self.metric], color=d3["Category10"][self.n_forwards][j])
            if i < self.n_snrs-1:
                fig.xaxis.major_label_text_font_size = '0pt'
            figs.append(fig)
        # figs[-1].xaxis.axis_label = "cm"

        self.gspec[:self.n_snrs, self.n_forwards+1] = pn.Column(*figs)
        return self.gspec

    def update_plots(self):
        self.make_surface_plots()
        self.make_density_plots()
        return self.gspec

#%%

dashboard = Dashboard(surface, dfs, dfd)
pn.Row(dashboard.update_plots)
dashboard.gspec
