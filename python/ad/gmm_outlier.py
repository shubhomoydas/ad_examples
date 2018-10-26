import numpy.random as rnd
from matplotlib.patches import Ellipse
from sklearn.mixture.gaussian_mixture import GaussianMixture as GMM

from common.gen_samples import *

"""
pythonw -m ad.gmm_outlier

This example is based on:
    http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
"""


def make_ellipses(gmm, pl, colors):
    covariances = None
    for k in range(gmm.n_components):
        color = colors[k]
        # logger.debug("color: %s" % color)
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[k][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[k][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[k]
        # find the ellipse size and orientation w.r.t largest eigen value
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])  # normalize direction of largest eigen value
        angle = np.arctan2(u[1], u[0])  # find direction of the vector with largest eigen value
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = Ellipse(xy=gmm.means_[k, :2], width=v[0], height=v[1], angle=180 + angle,
                      edgecolor=color, facecolor='none', linewidth=2)
        ell.set_clip_box(pl.bbox)
        # ell.set_alpha(0.5)
        # ell.set_facecolor('none')
        pl.add_artist(ell)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/gmm_outlier.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    # sample_type = "4_"
    # sample_type = "donut_"
    sample_type = "face_"

    rnd.seed(42)

    x, y = get_demo_samples(sample_type)

    xx = yy = x_grid = Z = scores = None
    if args.plot:
        plot_sample(x, y, pdfpath="temp/gmm_%ssamples.pdf" % sample_type)

        # to plot probability contours
        xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50),
                             np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50))
        x_grid = np.c_[xx.ravel(), yy.ravel()]

    cov_type = 'full'  # 'spherical', 'diag', 'tied', 'full'
    n_classes = 3
    max_iters = 20 if sample_type == "" else 100
    gmm = GMM(n_components=n_classes, covariance_type=cov_type, max_iter=max_iters, random_state=0)
    gmm.fit(x)

    logger.debug("Means:\n%s" % str(gmm.means_))
    logger.debug("Covariances:\n%s" % str(gmm.covariances_))

    scores = gmm.score_samples(x)
    top_anoms = np.argsort(scores)[np.arange(10)]

    if args.plot:
        # colors = ["red", "blue"]
        colors = ['navy', 'turquoise', 'darkorange']

        pdfpath = "temp/gmm_%scontours.pdf" % sample_type
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=2)

        pl = dp.get_next_plot()
        make_ellipses(gmm, pl, colors)
        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')

        Z = -gmm.score_samples(x_grid)
        Z = Z.reshape(xx.shape)
        pl = dp.get_next_plot()
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
        dp.close()
