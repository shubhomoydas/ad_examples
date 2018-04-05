import numpy as np
import numpy.random as rnd
import tensorflow as tf
from common.utils import *
from common.gen_samples import *
from common.nn_utils import *
from common.data_plotter import *
from common.timeseries_datasets import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Apply PCA/Autoencoder to reduce data to 2D and visualize.
Visualization should provide a hint on how easy/difficult anomaly detection is.
Consider applying other dimensionality reduction techniques such as MDS, t-SNE, etc.

To execute:
pythonw -m dnn.autoencoder --n_epochs=100 --log_file=temp/dnn/autoencoder.log --debug --dataset=kddcup_sub
"""


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=False, debug_args=["--dataset=kddcup_sub",
                                                    "--n_epochs=100",
                                                    "--debug", "--plot",
                                                    "--log_file=temp/dnn/autoencoder.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/dnn")  # for logging and plots

    rnd.seed(42)

    x, y = read_anomaly_dataset(args.dataset)

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
    dp.plot_points(pca_codings[y==0, :], pl, labels=y[y==0], lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.plot_points(pca_codings[y==1, :], pl, labels=y[y==1], lbl_color_map={0: "grey", 1: "red"}, s=25)

    pl = dp.get_next_plot()
    plt.title("Autoencoder")
    dp.plot_points(autoenc_codings[y==0, :], pl, labels=y[y==0], lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.plot_points(autoenc_codings[y==1, :], pl, labels=y[y==1], lbl_color_map={0: "grey", 1: "red"}, s=25)

    dp.close()
