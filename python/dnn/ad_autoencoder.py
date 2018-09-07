import numpy.random as rnd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from common.gen_samples import *
from common.nn_utils import *
from common.timeseries_datasets import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
pythonw -m dnn.ad_autoencoder
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/dnn")  # for logging and plots

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/dnn/ad_autoencoder.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    ad_type = "pca"  # autoenc, pca

    sample_type = "4_"
    # sample_type = "donut_"
    # sample_type = "donut_diff_"
    # sample_type = "face_"
    # sample_type = "face_diff_"

    rnd.seed(42)

    x, y = get_demo_samples(sample_type)

    n = x.shape[0]

    outliers_fraction = 0.1

    xx = yy = x_grid = Z = scores = None
    if args.plot:
        plot_sample(x, y, pdfpath="temp/dnn/ad_%ssamples.pdf" % (sample_type))

        # to plot probability contours
        xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50),
                             np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50))
        x_grid = np.c_[xx.ravel(), yy.ravel()]

    if ad_type == "autoenc":
        ad = AutoencoderAnomalyDetector(n_inputs=x.shape[1], n_neurons=[300, 10, 300],
                                        normalize_scale=True,
                                        activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None])
        ad.fit(x)
        scores = -ad.decision_function(x)
        Z = -ad.decision_function(x_grid)
    elif ad_type == "pca":
        n_dims = 1 if x.shape[1] == 2 else 2
        ad = AutoencoderAnomalyDetector(n_inputs=x.shape[1], n_neurons=[n_dims],
                                        normalize_scale=True,
                                        activations=[None, None])
        ad.fit(x)
        scores = -ad.decision_function(x)
        Z = -ad.decision_function(x_grid)
    else:
        raise ValueError("invalid ad type: %s" % ad_type)

    logger.debug("scores:\n%s" % str(list(scores)))
    top_anoms = np.argsort(-scores)[np.arange(10)]

    if args.plot:
        # plot_samples_and_lines(x, lines=None, line_colors=None, line_legends=None,
        #                        top_anoms=top_anoms, pdfpath="temp/%s_%soutlier.pdf" % (ad_type, sample_type))
        Z = Z.reshape(xx.shape)
        pdfpath = "temp/dnn/ad_%scontours_%s.pdf" % (sample_type, ad_type)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))
        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
        dp.close()

