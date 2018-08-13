from aad.data_stream import *
from common.gen_samples import read_anomaly_dataset
from aad.anomaly_dataset_support import *
from aad.classifier_trees import RandomForestAadWrapper


"""
Check data drift with a Random Forest classifier.

NOTE: The classifier is trained only once in this example with the 
    first window of data. The drift is tested for the rest of the 
    windows *without* updating the model.

To run:
pythonw -m aad.test_concept_drift_classifier --debug --plot --log_file=temp/test_concept_drift_classifier.log --dataset=weather
"""


def test_kl_data_drift_classifier():
    logger = logging.getLogger(__name__)

    args = get_command_args(debug=False)
    configure_logger(args)

    dataset_config = dataset_configs[args.dataset]
    stream_window = dataset_config[2]
    alpha = 0.05
    n_trees = 100

    X_full, y_full = read_anomaly_dataset(args.dataset)
    logger.debug("dataset: %s (%d, %d), stream_window: %d, alpha: %0.3f" %
                 (args.dataset, X_full.shape[0], X_full.shape[1], stream_window, alpha))

    stream = DataStream(X_full, y_full, IdServer(initial=0))

    # get first window of data
    training_set = stream.read_next_from_stream(stream_window)
    x, y, ids = training_set.x, training_set.y, training_set.ids
    logger.debug("First window loaded (%s): %d" % (args.dataset, x.shape[0]))

    # train classifier with the window of data
    rf = RFClassifier.fit(x, y, n_estimators=n_trees)
    logger.debug("Random Forest classifier created with %d trees" % rf.clf.n_estimators)

    # prepare wrapper over the classifier which will compute KL-divergences
    # NOTE: rf.clf is the scikit-learn Random Forest classifier instance
    model = RandomForestAadWrapper(x=x, y=y, clf=rf.clf)
    logger.debug("Wrapper model created with %d nodes" % len(model.w))

    # compute KL replacement threshold *without* p
    ref_kls, kl_q_alpha = model.get_KL_divergence_distribution(x, p=None, alpha=alpha)
    # now initialize reference p
    p = model.get_node_sample_distributions(x)

    window = 0
    while not stream.empty():
        window += 1
        # get next window of data and check KL-divergence
        training_set = stream.read_next_from_stream(n=stream_window)
        x, y = training_set.x, training_set.y

        logger.debug("window %d loaded: %d" % (window, x.shape[0]))

        # compare KL-divergence of current data dist against reference dist p
        comp_kls, _ = model.get_KL_divergence_distribution(x, p=p)

        # find which trees exceed alpha-level threshold
        trees_exceeding_kl_q_alpha = model.get_trees_to_replace(comp_kls, kl_q_alpha)
        n_threshold = int(2 * alpha * n_trees)

        logger.debug("[%d] #trees_exceeding_kl_q_alpha: %d, threshold number of trees: %d\n%s" %
                     (window, len(trees_exceeding_kl_q_alpha), n_threshold, str(list(trees_exceeding_kl_q_alpha))))


if __name__ == "__main__":
    test_kl_data_drift_classifier()
