import numpy as np
import numpy.random as rnd
import tensorflow as tf
from common.utils import *
from common.metrics import *
from common.gen_samples import *
from common.nn_utils import *
from common.data_plotter import *
from common.timeseries_datasets import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Apply Autoencoder to detect anomalies and report AUCs.

Also has functionality to reduce data to 2D with PCA/Autoencoder for visualization.
Visualization should provide a hint on how easy/difficult is anomaly detection for the dataset.
Consider applying other dimensionality reduction techniques such as MDS, t-SNE, etc.

To execute:
pythonw -m dnn.autoencoder --n_epochs=100 --log_file=temp/dnn/autoencoder.log --debug --dataset=kddcup_sub
"""


def autoencoder_visualize(x, args):
    # scale to (-1, 1) since we will apply tanh activation
    normalizer = DiffScale()
    x = normalizer.fit_transform(x)

    pca = PCA_TF(n_inputs=x.shape[1], n_dims=2,
                 n_epochs=args.n_epochs, batch_size=20, l2_penalty=0.)
    pca.fit(x)
    pca_codings = pca.transform(x)

    autoenc = Autoencoder(n_inputs=x.shape[1], n_neurons=[300, 2, 300],
                          activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
                          n_epochs=args.n_epochs, batch_size=20, l2_penalty=0.001)
    autoenc.fit(x)
    autoenc_codings = autoenc.transform(x, layer_id=1)

    pdfpath = "temp/dnn/autoencoder_pca_%s.pdf" % args.dataset
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2)

    pl = dp.get_next_plot()
    plt.title("PCA")
    dp.plot_points(pca_codings[y == 0, :], pl, labels=y[y == 0], lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.plot_points(pca_codings[y == 1, :], pl, labels=y[y == 1], lbl_color_map={0: "grey", 1: "red"}, s=25)

    pl = dp.get_next_plot()
    plt.title("Autoencoder")
    dp.plot_points(autoenc_codings[y == 0, :], pl, labels=y[y == 0], lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.plot_points(autoenc_codings[y == 1, :], pl, labels=y[y == 1], lbl_color_map={0: "grey", 1: "red"}, s=25)

    dp.close()


def autoencoder_ad(x, y, args):
    n_hidden_dims = max(1, x.shape[1]//2)
    ad = AutoencoderAnomalyDetector(n_inputs=x.shape[1],
                                    n_neurons=[200, n_hidden_dims, 200],
                                    normalize_scale=True,  # scale to (-1, 1)
                                    activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None])
    ad.fit(x)
    scores = ad.decision_function(x)
    auc = fn_auc(np.hstack([np.transpose([y]), np.transpose([scores])]))
    logger.debug("%s auc: %f" % (args.dataset, auc))
    return auc


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/dnn")  # for logging and plots

    args = get_command_args(debug=False, debug_args=["--dataset=kddcup_sub",
                                                     "--n_epochs=100",
                                                     "--debug", "--plot",
                                                     "--log_file=./temp/dnn/autoencoder.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    rnd.seed(42)

    x, y = read_anomaly_dataset(args.dataset)

    # autoencoder_visualize(x, args)
    auc = autoencoder_ad(x, y, args)
    print ("AUC for dataset %s: %f" % (args.dataset, auc))
