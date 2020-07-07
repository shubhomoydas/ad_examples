import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.ensemble import IsolationForest

from ..common.utils import nrow, get_command_args, configure_logger
from ..common.gen_samples import get_demo_samples, plot_sample, normalize_and_center_by_feature_range
from ..common.data_plotter import DataPlotter

"""
python -m ad_examples.ad.spectral_outlier
"""


def euclidean_dist(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist


class LabelDiffusion(object):
    """
    IMPORTANT: The results from Python's Scikit-Learn MDS API are significantly
    different (and sub-optimal) from R. Strongly recommend R's isoMDS for the last
    step of converting pair-wise distances to 2D coordinates.
    """
    def __init__(self, n_neighbors=10, k2=0.5, alpha=0.99,
                 n_components=2, eigen_solver='auto',
                 tol=0., max_iter=None, n_jobs=1, metric=True):
        self.n_neighbors = n_neighbors
        self.k2 = k2
        self.alpha = alpha
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.metric = metric

        self.alphas_ = None
        self.lambdas_ = None

    def fit_transform(self, x_in):
        n = nrow(x_in)
        x = normalize_and_center_by_feature_range(x_in)
        dists = np.zeros(shape=(n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                dists[i, j] = euclidean_dist(x[i, :], x[j, :])
                dists[j, i] = dists[i, j]

        logger.debug(dists[0, 0:10])

        neighbors = np.zeros(shape=(n, self.n_neighbors), dtype=int)
        for i in range(n):
            neighbors[i, :] = np.argsort(dists[i, :])[0:self.n_neighbors]

        logger.debug(neighbors[0, 0:10])

        W = np.zeros(shape=(n, n))
        for i in range(n):
            for j in neighbors[i, :]:
                # diagonal elements of W will be zeros
                if i != j:
                    W[i, j] = np.exp(-(dists[i, j] ** 2) / self.k2)
                    W[j, i] = W[i, j]

        D = W.sum(axis=1)
        # logger.debug(str(list(D[0:10])))

        iDroot = np.diag(np.sqrt(D) ** (-1))

        S = iDroot.dot(W.dot(iDroot))
        # logger.debug("S: %s" % str(list(S[0, 0:10])))

        B = np.eye(n) - self.alpha * S
        # logger.debug("B: %s" % str(list(B[0, 0:10])))

        A = np.linalg.inv(B)
        tdA = np.diag(np.sqrt(np.diag(A)) ** (-1))
        A = tdA.dot(A.dot(tdA))
        # logger.debug("A: %s" % str(list(A[0, 0:10])))

        d = 1 - A
        # logger.debug("d: %s" % str(list(d[0, 0:10])))
        # logger.debug("min(d): %f, max(d): %f" % (np.min(d), np.max(d)))

        mds = manifold.MDS(self.n_components,
                           metric=self.metric, dissimilarity='precomputed')
        # using abs below because some zeros are represented as -0; other values are positive.
        embedding = mds.fit_transform(np.abs(d))

        return embedding


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/spectral_outlier.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    # sample_type = "4_"
    # sample_type = "donut_"
    sample_type = "face_"

    rnd.seed(42)

    x, y = get_demo_samples(sample_type)
    n = x.shape[0]

    xx = yy = x_grid = Z = scores = None
    if args.plot:
        plot_sample(x, y, pdfpath="temp/spectral_%ssamples.pdf" % sample_type)

    n_neighbors = 10
    n_components = 2

    method = "standard"  # ['standard', 'ltsa', 'hessian', 'modified']

    # embed_type = "se"
    # embed_type = "tsne"
    # embed_type = "isomap"
    # embed_type = "mds"
    # embed_type = "lle_%s" % method
    embed_type = "diffusion"

    if embed_type == "se":
        embed = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    elif embed_type == "tsne":
        embed = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    elif embed_type.startswith("lle_"):
        embed = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                                eigen_solver='auto', method=method)
    elif embed_type == "isomap":
        embed = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    elif embed_type == "mds":
        embed = manifold.MDS(n_components=n_components)
    elif embed_type == "diffusion":
        embed = LabelDiffusion(n_neighbors=n_neighbors, n_components=n_components, metric=True)
    else:
        raise ValueError("invalid embed type %s" % embed_type)

    x_tr = embed.fit_transform(x)
    logger.debug(x_tr)

    if args.plot:
        plot_sample(x_tr, y, pdfpath="temp/spectral_%s%s.pdf" % (sample_type, embed_type))

    ad_type = 'ifor'

    outliers_fraction = 0.1
    ad = IsolationForest(max_samples=256, contamination=outliers_fraction, random_state=None)
    ad.fit(x_tr)
    scores = -ad.decision_function(x_tr)

    top_anoms = np.argsort(-scores)[np.arange(10)]

    if args.plot:

        # to plot probability contours
        xx, yy = np.meshgrid(np.linspace(np.min(x_tr[:, 0]), np.max(x_tr[:, 0]), 50),
                             np.linspace(np.min(x_tr[:, 1]), np.max(x_tr[:, 1]), 50))
        x_grid = np.c_[xx.ravel(), yy.ravel()]

        Z = -ad.decision_function(x_grid)
        Z = Z.reshape(xx.shape)

        pdfpath = "temp/spectral_%scontours_%s_%s.pdf" % (sample_type, ad_type, embed_type)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
        dp.plot_points(x_tr, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x_tr[top_anoms, 0], x_tr[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
        dp.close()
