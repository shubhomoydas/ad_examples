from .gan import *
from ad.gmm_outlier import make_ellipses


def get_gan_sample_defs():
    sampledefs = list([
        MVNParams(
            mu=np.array([0.]),
            mcorr=np.array([
                [1]]),
            dvar=np.array([1.])
        ),
        MVNParams(
            mu=np.array([4.]),
            mcorr=np.array([
                [1]]),
            dvar=np.array([0.5])
        ),
        MVNParams(
            mu=np.array([6.]),
            mcorr=np.array([
                [1]]),
            dvar=np.array([0.5])
        ),
        MVNParams(
            mu=np.array([10.]),
            mcorr=np.array([
                [1]]),
            dvar=np.array([0.5])
        )
    ])
    return sampledefs


def get_normal_samples(sample_indexes=None, n=None):
    sampledefs = get_gan_sample_defs()
    d = 1
    s = np.zeros(shape=(0, d))
    for i, index in enumerate(sample_indexes):
        sampledef = sampledefs[index]
        si = generate_dependent_normal_samples(n[i], sampledef.mu, sampledef.mcorr, sampledef.dvar)
        if d == 1:
            si = np.reshape(si, newshape=(-1, d))
        s = np.append(s, si, axis=0)
    return s


def read_dataset(args):
    if args.dataset == "1":
        x = get_normal_samples([1], [200])
        y = np.zeros(200, dtype=int)
    elif args.dataset == "2":
        x = get_normal_samples([0, 1], [200, 200])
        y = np.append(np.zeros(200, dtype=int), np.ones(200, dtype=int))
    elif args.dataset == "3":
        x = get_normal_samples([0, 1, 2], [200, 200, 200])
        y = np.append(np.zeros(200, dtype=int), np.ones(200, dtype=int))
        y = np.append(y, 2*np.ones(200, dtype=int))
    elif args.dataset == "4":
        x = get_normal_samples([0, 1, 2, 3], [200, 200, 200, 200])
        y = np.append(np.zeros(200, dtype=int), np.ones(200, dtype=int))
        y = np.append(y, 2*np.ones(200, dtype=int))
        y = np.append(y, 3*np.ones(200, dtype=int))
    elif args.dataset == "donut":
        x, y = load_donut_data()
    elif args.dataset == "face":
        x, y = load_face_data()
    else:
        # raise ValueError("dataset '%s' not supported" % opts.dataset)
        x, y = read_anomaly_dataset(args.dataset)
    return x, y


def plot_sample_hist(x, y, pl, labels=None, n_buckets=50,
                     plot_samples=True, plot_hist=True,
                     color='black', legend="Original data", marker='o',
                     legend_loc="upper right", legend_size=8):
    plt.xlabel('x', fontsize=12)
    plt.ylabel("")

    colors = ["red", "blue", "green", "orange", "pink", "brown"]
    if plot_samples:
        if labels is not None:
            cls = np.unique(labels)
            for i, c in enumerate(cls):
                c_idx = np.where(labels == c)[0]
                legend_tmp = "class %d (%s)" % (c, legend)
                pl.scatter(x[c_idx], y[c_idx], c=colors[i], marker=marker, linewidths=2.0, s=24, label=legend_tmp)
        else:
            pl.scatter(x, y, c=color, marker=marker, linewidths=2.0, s=24, label=legend)

    if plot_hist:
        bins = np.arange(start=np.min(x), stop=np.max(x), step=(np.max(x) - np.min(x)) / n_buckets)
        n, bins1 = np.histogram(x, bins=bins, normed=True)
        center = (bins[:-1] + bins[1:]) / 2
        pl.plot(center, n, '-', color=color, linewidth=1, label=None)

    pl.legend(loc=legend_loc, prop={'size': legend_size})


def plot_1D_gan_samples(x, y=None, gen_x=None, gen_y=None, pdfpath=None):
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    plot_sample_hist(x[:, 0], np.zeros(x.shape[0], dtype=np.float32), pl, labels=y, n_buckets=20,
                     legend='Orig. data', color='black', marker='o')
    plot_sample_hist(gen_x[:, 0], 0.5*np.ones(gen_x.shape[0], dtype=np.float32), pl, labels=gen_y, n_buckets=20,
                     legend='Synth. data', color='red', marker='x')
    dp.close()


def plot_2D_gan_samples(x, y=None, gen_x=None, gen_y=None, gmm=None, pdfpath=None):
    colors = ['green', 'red', 'blue', 'brown', 'yellow', 'pink', 'navy', 'turquoise', 'darkorange', 'magenta']
    dp = DataPlotter(pdfpath=pdfpath,
                     rows=1 if gen_x is None else 2,
                     cols=1 if gen_x is None else 2)
    pl = dp.get_next_plot()
    if gmm is not None:
        make_ellipses(gmm, pl, colors)
    xlim = [np.min(x[:, 0]), np.max(x[:, 0])]
    ylim = [np.min(x[:, 1]), np.max(x[:, 1])]
    if gen_x is not None:
        xlim = [min(xlim[0], np.min(gen_x[:, 0])), max(xlim[1], np.max(gen_x[:, 0]))]
        ylim = [min(ylim[0], np.min(gen_x[:, 1])), max(ylim[1], np.max(gen_x[:, 1]))]
    plt.xlim(xlim)
    plt.ylim(ylim)
    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
    if gen_x is not None:
        pl = dp.get_next_plot()
        plt.xlim(xlim)
        plt.ylim(ylim)
        dp.plot_points(gen_x, pl, labels=gen_y, lbl_color_map={0: "grey", 1: "red"}, s=25)
    dp.close()


def plot_log_likelihood(lls, plot_sd=True, opts=None):
    llpdfpath = "%s/%s_log_likelihood.pdf" % (opts.results_dir, opts.get_opts_name_prefix())
    x = [epoch+1 for epoch, _, _ in lls]
    y = [ll_mean for _, ll_mean, _ in lls]
    sd = [ll_sd for _, _, ll_sd in lls]
    dp = DataPlotter(pdfpath=llpdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.plot(x, y, '-', color='black', linewidth=1, label=None)
    if plot_sd:
        pl.errorbar(x, y, yerr=sd, fmt='.', color='black')
    dp.close()


class GanListener(Listener):
    """ Callback class that gets called at the end of a GAN training epoch """
    def __init__(self, x, y=None, ll_freq=10, plot_freq=10, opts=None, ep=1e-1):
        Listener.__init__(self)
        self.x = x
        self.y = y
        self.ll_freq = ll_freq
        self.plot_freq = plot_freq
        self.opts = opts
        self.ep = ep

        self.lls = []

    def __call__(self, gan, epoch, epoch_start_tm):
        if (epoch + 1) % 1 == 0:
            logger.debug(epoch_start_tm.message("Completed epoch [%d/%d]" % (epoch + 1, gan.n_epochs)))

        if (epoch + 1) % self.ll_freq == 0:
            ll_tm = Timer()
            ll_mean, ll_sd = gan.get_log_likelihood(self.x)
            logger.debug(ll_tm.message("Data log-likelihood after epoch %d: %f(%f)" % (epoch, ll_mean, ll_sd)))
            self.lls.append((epoch, ll_mean, ll_sd))

            if False:
                # some debug metrics
                probs, n_low = self.get_discriminator_metrics(gan)
                logger.debug("fraction of very low probability real data: %d" % n_low)
                logger.debug("probability of real data:\n%s" % str(list(probs)))

        if self.opts.plot and (epoch + 1) % self.plot_freq == 0:
            pdfpath = "%s/%s_%d.pdf" % (self.opts.results_dir, self.opts.get_opts_name_prefix(), epoch + 1)
            z, y = gan.get_gen_input_samples(n=self.x.shape[0], gen_y=gan.conditional)
            gen_y = None
            if y is not None:
                # convert from one-hot to int labels
                gen_y = np.dot(y, np.arange(gan.n_classes, dtype=int).reshape((-1, 1))).reshape((-1,)).astype(int)
            gen_x = gan.get_gen_output_samples(z=z, y=y)
            if self.x.shape[1] == 1:  # plot only if 1D data
                val_x = gan.get_gen_output_samples(z=z, y=y)
                try:
                    gmm_tm = Timer()
                    gmm, best_bic, all_bics = fit_gmm(x=gen_x, val_x=val_x)
                    logger.debug(gmm_tm.message("best gmm k: %d, bic: %f, all_bics: %s" % (gmm.n_components, best_bic, str(list(all_bics)))))
                except:
                    logger.warning("GMM error: %s" % str(sys.exc_info()[0]))
                plot_1D_gan_samples(self.x, y=self.y, gen_x=gen_x, gen_y=gen_y, pdfpath=pdfpath)
            elif self.x.shape[1] == 2:  # plot only if 2D data
                plot_2D_gan_samples(x=self.x, gen_x=gen_x, pdfpath=pdfpath)

    def get_discriminator_metrics(self, gan):
        """ Returns some properties of the trained discriminator based on real data samples """
        probs = gan.get_discriminator_probability(self.x)
        threshold = self.ep/self.x.shape[0]
        # find instances which have very low probability
        low_prob_insts = np.where(probs < threshold)[0]
        return probs, len(low_prob_insts * 1. / self.x.shape[0])


def test_samples(opts):
    x, y = read_dataset(opts)

    d = x.shape[1]

    # logger.debug("samples: %s\n%s" % (str(x.shape), str(x)))
    logger.debug("samples: %s" % (str(x.shape)))

    if opts.plot and d == 1:
        pdfpath = "%s/%s_orig_samples.pdf" % (opts.results_dir, opts.dataset)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()
        plot_sample_hist(x[:, 0], np.zeros(x.shape[0], dtype=np.float32), pl, labels=y, n_buckets=20)
        dp.close()
    elif opts.plot and d == 2:
        _, gmm = get_cluster_labels(x)
        pdfpath = "%s/%s_orig_samples.pdf" % (opts.results_dir, opts.dataset)
        plot_2D_gan_samples(x, y, gmm=gmm, pdfpath=pdfpath)
