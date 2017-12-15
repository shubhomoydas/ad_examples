from forest_aad_detector import *
from classifier.perceptron import Perceptron
from common.data_plotter import *


logger = logging.getLogger(__name__)


def get_angles(x, w):
    n = x.shape[0]
    a = np.zeros(n, dtype=float)
    # logger.debug("x[0]:\n%s" % str(list(x[0].todense())))
    # logger.debug("x[0]:\n%s" % str(list(x[0].data)))
    for i in range(n):
        cos_theta = x[i].dot(w)
        a[i] = np.arccos(cos_theta)*180./np.pi
    return a


def plot_angle_hist(vals, labels, dp):
    nom_v = vals[np.where(labels==-1)[0]]
    anom_v = vals[np.where(labels==1)[0]]
    bins = np.arange(start=np.min(vals), stop=np.max(vals), step=(np.max(vals)-np.min(vals))/50)
    pl = dp.get_next_plot()
    n1, bins1 = np.histogram(nom_v, bins=bins, normed=True)
    n2, bins2 = np.histogram(anom_v, bins=bins, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, n1, align='center', width=width, facecolor='green', alpha=0.50)
    plt.bar(center, n2, align='center', width=width, facecolor='red', alpha=0.50)


def test_hyperplane_angles():

    dense = False  # DO NOT Change this!

    # PRODUCTION code
    args = get_aad_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)

    opts = AadOpts(args)
    logger.debug(opts.str_opts())

    if opts.streaming:
        raise ValueError("Streaming not supported")

    data = pd.read_csv(opts.datafile, header=0, sep=',', index_col=None)
    X_train = np.zeros(shape=(data.shape[0], data.shape[1]-1))
    for i in range(X_train.shape[1]):
        X_train[:, i] = data.iloc[:, i + 1]
    labels = np.array([1 if data.iloc[i, 0] == "anomaly" else -1 for i in range(data.shape[0])], dtype=int)

    # X_train = X_train[0:10, :]
    # labels = labels[0:10]

    logger.debug("loaded file: %s" % opts.datafile)
    logger.debug("results dir: %s" % opts.resultsdir)
    logger.debug("forest_type: %s" % detector_types[opts.detector_type])

    rng = np.random.RandomState(args.randseed)

    compute_angles = True
    compute_optimal_plane = True

    a = []
    first_run = True
    for runidx in opts.get_runidxs():

        # fit the model
        mdl = AadForest(n_estimators=opts.forest_n_trees,
                        max_samples=min(opts.forest_n_samples, X_train.shape[0]),
                        score_type=opts.forest_score_type, random_state=rng,
                        add_leaf_nodes_only=opts.forest_add_leaf_nodes_only,
                        max_depth=opts.forest_max_depth,
                        ensemble_score=opts.ensemble_score,
                        detector_type=opts.detector_type, n_jobs=opts.n_jobs)
        mdl.fit(X_train)

        logger.debug("total #nodes: %d" % (len(mdl.all_regions)))

        X_train_new = mdl.transform_to_region_features(X_train, dense=dense, norm_unit=opts.norm_unit)

        unif_w = np.ones(len(mdl.d), dtype=float)
        unif_w = unif_w / np.sqrt(unif_w.dot(unif_w))  # normalized uniform weights

        # logger.debug("w:%s" % str(w))
        # logger.debug("|w| = %f" % (np.sum(w * w)))

        n = X_train_new.shape[0]
        logger.debug("n: %d, anomalies: %d" % (n, len(np.where(labels==1)[0])))

        if compute_angles and first_run:
            # Plot angle histogram for only the first run
            # Computes the angles between each sample vector and the
            # uniform weight vector.
            dir_create("%s/percept" % args.resultsdir)
            pdfpath = "%s/percept/angles_%s.pdf" % (args.resultsdir, args.dataset)

            angles = get_angles(X_train_new, unif_w)
            # logger.debug("angles:\n%s" % str(list(angles)))

            dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
            plot_angle_hist(angles, labels, dp)
            dp.close()

        if compute_optimal_plane:
            # Runs a perceptron to compute the optimal hyperplane
            # and then computes the angle between this plane and
            # the uniform weight vector.
            perc = Perceptron(learning_rate=1.)
            perc.fit(X_train_new, labels, unif_w, epochs=200)
            cos_theta = unif_w.dot(perc.w)
            angle = np.arccos(cos_theta)*180./np.pi
            a.append(angle)
            logger.debug("Run %d, angle: %f (%f)" % (runidx, angle, cos_theta))

        # next time we will know that this is not the first run
        # and will not plot histogram of angles.
        first_run = False

    angles = np.array(a, dtype=float)
    logger.debug("Mean Angle: %f (%f)" % (np.mean(angles),
                                          1.95*np.std(angles)/np.sqrt(1.*len(angles))))


if __name__ == "__main__":
    test_hyperplane_angles()
