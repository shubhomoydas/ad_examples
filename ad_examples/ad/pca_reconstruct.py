import logging
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ..common.utils import get_command_args, configure_logger
from ..common.gen_samples import get_demo_samples, interpolate_2D_line_by_point_and_vec, plot_samples_and_lines
from ..common.data_plotter import DataPlotter

"""
python -m ad_examples.ad.pca_reconstruct
"""


def plot_samples_pca(s, labels, type):
    pdfpath = "./temp/samples.pdf"
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('x')
    plt.ylabel('y')
    dp.plot_points(s, pl, labels=labels, lbl_color_map={0: "blue", 1: "red"},
                   marker='o', s=35, facecolors='none')
    dp.close()


logger = logging.getLogger(__name__)

args = get_command_args(debug=True, debug_args=["--debug", "--log_file=temp/pca_reconstruct.log"])
# print "log file: %s" % args.log_file
configure_logger(args)

# sample_type = "1_"
# sample_type = "donut_"
sample_type = "face_"

rnd.seed(42)

x, y = get_demo_samples(sample_type)

pca = PCA(n_components=1, svd_solver='full')
pca.fit(x)

V = pca.components_
logger.debug(V)

# mu = x.mean(axis=0)
Z = interpolate_2D_line_by_point_and_vec(np.array([np.min(x[:, 0]), np.max(x[:, 0])]),
                                         pca.mean_, V[0 , :])

VT = V.T

# to find correct loadings, remember that PCA components are computed on centered data.
x_ = x - pca.mean_
loadings = x_.dot(VT)
# logger.debug("loadings:\n%s" % str(loadings))

# logger.debug("first loading:\n%s" % str(loadings[0, :]))
# logger.debug("first:\n%s" % str(X[0, :]))
# tmp = VT.dot(loadings[0, :])
# logger.debug("recons:\n%s" % str(tmp))

tmp = VT.dot(loadings.T)
tmp = tmp.T

diff2 = (x_ - tmp) ** 2
err = np.sqrt(diff2.sum(axis=1))

top_anoms = np.argsort(-err)[0:10]

line_colors = ['black', 'red', 'green', 'blue', 'brown']

plot_samples_and_lines(x, [Z], line_colors=line_colors, line_legends=None, top_anoms=top_anoms,
                       pdfpath="./temp/pca_reconstruct_%soutlier.pdf" % sample_type)

