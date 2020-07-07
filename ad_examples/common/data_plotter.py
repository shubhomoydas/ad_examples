import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .utils import cbind

plt.rcParams.update({'figure.max_open_warning': 100})


class DataPlotter(object):
    """Encapsulates the layout of PDF plots

    The PDF will have a grid layout of rows x cols subplots.
    When a new plot is added, the next subplot in the grid is
    returned and if it exceeds the max available on the page,
    then a new page is started.
    """

    def __init__(self, pdfpath, rows, cols, bbox_inches='tight', save_tight=False):
        self.pdfpath = pdfpath
        self.pdf = None
        self.rows = rows
        self.cols = cols
        self.bbox_inches = bbox_inches
        self.save_tight = save_tight
        self.fig = None
        self.pidx = 0
        self.plotcount = 0

    def get_next_plot(self):
        if self.pdf is None:
            plt.rc('pdf', fonttype=42)
            self.pdf = PdfPages(self.pdfpath)
        newfig = False
        if self.pidx == 0 or self.pidx == self.rows * self.cols:
            # Start a new pdf page
            if self.fig is not None:
                if self.save_tight: plt.tight_layout()
                self.pdf.savefig(self.fig, bbox_inches=self.bbox_inches)
                self.fig = None
            newfig = True
            self.pidx = 0
        self.pidx += 1
        if newfig: self.fig = plt.figure()
        self.plotcount += 1
        return self.fig.add_subplot(self.rows, self.cols, self.pidx)

    def plot_points(self, pts, pl, labels=None,
                    lbl_color_map=None, defaultcol="grey",
                    edgecolor=None, facecolors=None,
                    s=12, marker='x', linewidths=0.8, plotindexes=None):
        """

        Colors:
            b: blue
            g: green
            r: red
            c: cyan
            m: magenta
            y: yellow
            k: black
            w: white

        """
        n = pts.shape[0]
        lbls = labels if labels is not None else [-1]*n
        if plotindexes is None:
            indexes = range(n)
        else:
            indexes = plotindexes
        if lbl_color_map is None:
            colors = [defaultcol] * n
        else:
            colors = [lbl_color_map[lbls[j]]
                      if lbls[j] >= 0 else defaultcol for j in range(n)]
        M = [(pts[j, 0], pts[j, 1], colors[j]) for j in indexes]
        for x, y, col in M:
            col = col if edgecolor is None else None
            fc = 'none' if facecolors is None or facecolors == 'none' else col
            # pl.plot(x, y, 'x', color=col, mfc="none", mec=col)
            pl.scatter(x, y, s=s, marker=marker, c=col, alpha=None,
                       edgecolors=col if edgecolor is None else edgecolor, linewidths=linewidths,
                       facecolors=fc)

    def save_fig(self):
        if self.fig is not None:
            if self.save_tight: plt.tight_layout()
            self.pdf.savefig(self.fig, bbox_inches=self.bbox_inches)
            plt.close(self.fig)
            self.fig = None

    def close(self):
        if self.fig is not None and self.pdf is not None:
            self.save_fig()
        if self.pdf is not None:
            self.pdf.close()


def plot_rect_region(pl, region, color, axis_lims, facecolor='none', alpha=1):
    xlims = axis_lims[0]
    ylims = axis_lims[1]
    xy = (max(region[0][0], xlims[0]), max(region[1][0], ylims[0]))
    width = min(region[0][1], xlims[1]) - xy[0]
    height = min(region[1][1], ylims[1]) - xy[1]
    pl.add_patch(plt.Rectangle(xy, width, height, facecolor=facecolor,
                               edgecolor=color, alpha=alpha))


def plot_sidebar(hts, dash_xy, dash_wh, pl):
    pl.add_patch(plt.Rectangle(dash_xy, dash_wh[0], dash_wh[1], facecolor='white',
                               edgecolor='black', alpha=1))
    dash_pts = cbind(np.ones(len(hts), dtype=float) * (dash_wh[0]/2) + dash_xy[0],
                     hts * dash_wh[1] + dash_xy[1])
    # print dash_pts
    pl.plot(dash_pts[:, 0], dash_pts[:, 1], 'ro', markersize=3, markerfacecolor='red')


