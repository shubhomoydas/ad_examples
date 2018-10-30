from .gan_test_support import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Simple GAN demo:
python -m dnn.test_gan --log_file=temp/gan/simple_gan.log --debug --plot --n_epochs=200 --dataset=2

Conditional GAN demo:
python -m dnn.test_gan --log_file=temp/gan/test_gan.log --results_dir=./temp/gan --debug --plot --n_epochs=1000 --dataset=2 --conditional
python -m dnn.test_gan --log_file=temp/gan/test_gan.log --results_dir=./temp/gan --debug --plot --n_epochs=1000 --dataset=3 --conditional

python -m dnn.test_gan --log_file=temp/gan/test_gan.log --results_dir=./temp/gan --debug --plot --n_epochs=1000 --dataset=toy2 --conditional
"""


def get_gan_layer_defs(d):
    if d <= 2:
        d_z = 10
        d_nodes = 8 * d
    else:
        d_z = min(10*d, 50)
        d_nodes = min(50, 8*d)
    return {'gen_input_dim': d_z,
            'd_nodes': d_nodes,
            'discr_layer_nodes': [d_nodes, d_nodes, d_nodes, 1],
            'discr_layer_activations': [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
            'gen_layer_nodes': [d_nodes, d_nodes//2, d],
            'gen_layer_activations': [tf.nn.leaky_relu, tf.nn.softplus, None],
            'ano_gan_max_iters': 300,
            'ano_gan_tol': 1e-3}


def test_ano_gan(gan=None, x=None, y=None, opts=None, tol=1e-3, max_iters=100):
    if not gan.enable_ano_gan:
        logger.debug("AnoGAN not enabled")
        return

    if x.shape[1] != 2:
        logger.debug("AnoGAN test instances for visualization only supports 2D Toy. "
                     "For other datasets, setup test instances appropriately.")
        return

    # Test instances for illustration for Toy dataset.
    # Set these up appropriately for other datasets.
    x_test = np.array([
                       [-2.5, 6.],
                       [2., 3.],
                       [-1., 2.],
                       [0., 0.],
                       [5, -2],
                       [-3, -1],
                       [3, 3],
                       [2, 6.5],
                       ], dtype=np.float32)
    # x_test = x_test[[0]]
    all_gen_x = []
    all_losses = np.zeros(x_test.shape[0], dtype=np.float32)
    all_traces = []
    logger.debug("Averaging AnoGAN test over %d times" % opts.n_ano_gan_test)
    for i in range(opts.n_ano_gan_test):
        gen_x, losses, losses_R, losses_D, traces = gan.get_anomaly_score(x_test, ano_gan_lambda=opts.ano_gan_lambda,
                                                                          tol=tol, max_iters=max_iters,
                                                                          use_loss=not opts.ano_gan_use_dist,
                                                                          mode_avg=not opts.ano_gan_individual)
        all_gen_x.append(gen_x)
        all_losses += losses
        all_traces.append(traces)
        logger.debug("[%d] Test x:\n%s\n\nAnoGAN x:\n%s" % (i, str(x_test), str(gen_x)))
        s = ["inst#, loss, loss_R, loss_D"]
        for i in range(len(losses)):
            l, lr, ld = (losses[i], losses_R[i], losses_D[i])
            s.append("%3d, %1.4f, %1.4f, %1.4f" % (i+1, l, lr, ld))
        logger.debug("\n%s" % "\n".join(s))

    all_losses = all_losses / opts.n_ano_gan_test
    ordered_insts = np.argsort(-all_losses)
    logger.debug("Ordered on loss score:\n%s" % str(list(ordered_insts+1)))

    d = x_test.shape[1]
    if opts.plot and d == 2:
        # plot details of only the first test run
        colors = ["red", "blue", "green", "orange", "brown", "cyan", "magenta", "pink", "yellow"]
        pdfpath = "%s/%s_test_ano_%d%s%s.pdf" % \
                  (opts.results_dir, opts.get_opts_name_prefix(), int(opts.ano_gan_lambda*100),
                   "_dist" if opts.ano_gan_use_dist else "", "_indv" if opts.ano_gan_individual else "")
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        dp.plot_points(x, pl, defaultcol='grey', marker='.', s=10)
        dp.plot_points(x_test, pl, defaultcol='black', marker='x', s=25)
        dp.plot_points(all_gen_x[0], pl, defaultcol='red', marker='x', s=25)
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_range = np.subtract(x_max, x_min) / 100.
        for i in range(x_test.shape[0]):
            pl.text(x_test[i, 0], x_test[i, 1] + 2*x_range[1],
                    "%d" % (i+1), fontsize=20, color='black')
            gen_x = all_gen_x[0]
            pl.text(gen_x[i, 0] + 2*x_range[1], gen_x[i, 1] + 2*x_range[1],
                    "%d" % (i + 1), fontsize=20, color=colors[i % len(colors)])

        for i, xy in enumerate(all_traces[0]):
            pl.plot(xy[:, 0], xy[:, 1], '-', color=colors[i % len(colors)], linewidth=1, label=None)
        dp.close()


def test_gan(opts):
    x, _ = read_dataset(opts)

    d = x.shape[1]

    y = y_one_hot = class_codes = pvals = gmm = None
    n_classes = 0

    if opts.conditional or opts.info_gan:
        # We will assign pseudo-classes using an unsupervised technique.
        # This might provide a better structure to guide the GAN training.
        y_cls, gmm = get_cluster_labels(x, min_k=2, max_k=10)

        n_classes = len(np.unique(y_cls))
        class_codes = np.eye(n_classes, dtype=np.float32)

        y_one_hot = class_codes[y_cls]
        pvals = np.sum(y_one_hot, axis=0) * 1.0 / len(y_cls)
        logger.debug("pvals: %s" % (str(list(pvals))))

        if opts.conditional:
            y = y_cls  # pass on the class labels only for Conditional GAN, not InfoGAN

    opts.k = n_classes

    if opts.plot and d == 2:
        pdfpath = "%s/%s.pdf" % (opts.results_dir, opts.get_opts_name_prefix())
        plot_2D_gan_samples(x, y=None, gmm=gmm, pdfpath=pdfpath)

    gan_defs = get_gan_layer_defs(d)

    gan = GAN(data_dim=d,
              discr_layer_nodes=gan_defs['discr_layer_nodes'],
              discr_layer_activations=gan_defs['discr_layer_activations'],
              gen_input_dim=gan_defs['gen_input_dim'],
              gen_layer_nodes=gan_defs['gen_layer_nodes'],
              gen_layer_activations=gan_defs['gen_layer_activations'],
              label_smoothing=opts.label_smoothing, smoothing_prob=0.9,
              info_gan=opts.info_gan, info_gan_lambda=opts.info_gan_lambda,
              conditional=opts.conditional, n_classes=n_classes, pvals=pvals, l2_lambda=0.001,
              enable_ano_gan=True,
              n_epochs=opts.n_epochs, batch_size=25, shuffle=True,
              listener=GanListener(x, y=y, ll_freq=100, plot_freq=100, opts=opts))
    gan.init_session()

    model_path = "%s/%s_model" % (opts.results_dir, opts.get_opts_name_prefix())
    if not gan.load_session(model_path):
        gan.fit(x, y_one_hot)

    logger.debug("\n%s" % "\n".join(["%d,%f,%f" % (epoch+1, ll_mean, ll_sd) for epoch, ll_mean, ll_sd in gan.listener.lls]))

    if opts.plot:
        if gan.listener is not None and len(gan.listener.lls) > 0:
            plot_log_likelihood(gan.listener.lls, plot_sd=True, opts=opts)
        if d == 1:
            z, _ = gan.get_gen_input_samples(n=x.shape[0], gen_y=False)
            if not opts.conditional:
                gen_x = gan.get_gen_output_samples(z=z)
                pdfpath = "%s/%s_final.pdf" % (opts.results_dir, opts.get_opts_name_prefix())
                plot_1D_gan_samples(x=x, gen_x=gen_x, pdfpath=pdfpath)
            else:
                for c in range(n_classes):
                    zy_one_hot = class_codes[np.ones(z.shape[0], dtype=int) * c]
                    gen_x = gan.get_gen_output_samples(z=z, y=zy_one_hot)
                    pdfpath = "%s/%s_final_c%d.pdf" % (opts.results_dir, opts.get_opts_name_prefix(), c)
                    plot_1D_gan_samples(x=x, gen_x=gen_x, pdfpath=pdfpath)
        elif d == 2:
            z, gen_y = gan.get_gen_input_samples(n=x.shape[0], gen_y=opts.conditional)
            # logger.debug("y_one_hot:\n%s" % str(gen_y))
            gen_x = gan.get_gen_output_samples(z=z, y=gen_y)
            pdfpath = "%s/%s_final.pdf" % (opts.results_dir, opts.get_opts_name_prefix())
            plot_2D_gan_samples(x, gen_x=gen_x, pdfpath=pdfpath)

    if opts.ano_gan:
        test_ano_gan(x=x, gan=gan, opts=opts,
                     tol=gan_defs['ano_gan_tol'], max_iters=gan_defs['ano_gan_max_iters'])

    gan.save_session(model_path, overwrite=False)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/gan")

    args = get_command_args(debug=False, debug_args=["--debug",
                                                     "--plot",
                                                     "--dataset=2",
                                                     "--log_file=temp/gan/test_gan.log"],
                            parser=get_gan_option_list())
    dir_create(args.results_dir)

    configure_logger(args)

    opts = GanOpts(args)

    set_random_seeds(opts.randseed, opts.randseed + 1, opts.randseed + 2)

    test_gan(opts)
    # test_gan(opts)
