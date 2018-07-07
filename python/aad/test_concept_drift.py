import os
import numpy as np

import logging

from common.utils import *
from aad.aad_globals import *
from aad.aad_support import *

from aad.data_stream import *

"""
pythonw -m aad.test_concept_drift
"""


def get_afonso_featuresets():
    malware_list = [
        "malware-2010-afonso-adrd",
        "malware-2010-afonso-airpush",
        "malware-2010-afonso-artemis",
        "malware-2010-afonso-geinimi",
        "malware-2010-afonso-iconosys",
        "malware-2010-afonso-imlog",
        "malware-2010-afonso-kmin",
        "malware-2010-afonso-mobclick",
        "malware-2010-afonso-pjapps",
        "malware-2012-afonso-airpush",
        "malware-2013-afonso-copycat",
        "malware-2014-afonso-gappusin",
        "malware-2015-afonso-fakebank",
        "malware-2016-afonso-malap",
    ]
    return malware_list


def get_dynamic_featuresets():
    malware_list = [
        "malware-2010-dynamic-adrd",
        "malware-2010-dynamic-airpush",
        "malware-2010-dynamic-artemis",
        "malware-2010-dynamic-geinimi",
        "malware-2010-dynamic-iconosys",
        "malware-2010-dynamic-imlog",
        "malware-2010-dynamic-kmin",
        "malware-2010-dynamic-mobclick",
        "malware-2010-dynamic-pjapps",
        "malware-2012-dynamic-airpush",
        "malware-2013-dynamic-copycat",
        "malware-2014-dynamic-gappusin",
        "malware-2015-dynamic-fakebank",
        "malware-2016-dynamic-malap",
    ]
    return malware_list


def get_static_featuresets():
    malware_list = [
        "malware-2010-static-adrd",
        "malware-2010-static-airpush",
        "malware-2010-static-artemis",
        "malware-2010-static-geinimi",
        "malware-2010-static-iconosys",
        "malware-2010-static-imlog",
        "malware-2010-static-kmin",
        "malware-2010-static-mobclick",
        "malware-2010-static-pjapps",
        "malware-2012-static-airpush",
        "malware-2013-static-copycat",
        "malware-2014-static-gappusin",
        "malware-2015-static-fakebank",
        "malware-2016-static-malap",
    ]
    return malware_list


def read_datasets(root_dir, type="S"):
    malware_lists = {"S": get_static_featuresets(),
                     "D": get_dynamic_featuresets(),
                     "A": get_afonso_featuresets()}
    malware_list = malware_lists[type]
    file_list = list()
    datasets = list()
    for f in malware_list:
        file_path = os.path.join(root_dir, f, "fullsamples", "%s_1.csv" % f)
        file_list.append(file_path)

        data = read_csv(file_path, header=True, sep=',')
        x = np.array(data.iloc[:, 1:])
        y = data.iloc[:, 0]
        y = np.array([1 if v == "anomaly" else 0 for v in y], dtype=int)
        datasets.append((x, y))
        # logger.debug("File:\n%s\n%d" % (file_path, x.shape[0]))

    return datasets, malware_list


def get_iforest_model(x):
    model = AadForest(n_estimators=100,  # 100,
                      max_samples=256,
                      score_type=IFOR_SCORE_TYPE_NEG_PATH_LEN, random_state=42,
                      add_leaf_nodes_only=True,
                      max_depth=100,
                      ensemble_score=ENSEMBLE_SCORE_LINEAR,
                      detector_type=AAD_IFOREST, n_jobs=4,
                      tree_update_type=TREE_UPD_OVERWRITE,
                      forest_replace_frac=1.0,
                      feature_partitions=None)
    model.fit(x)
    return model


def test_concept_drift():
    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/test_concept_drift.log"])
    configure_logger(args)

    np.random.seed(42)

    root_dir = "/Users/moy/work/WSU/datasets/malware"
    # malware_type = "S"
    malware_type = "D"
    # malware_type = "A"
    datasets, malware_list = read_datasets(root_dir, type=malware_type)
    logger.debug("Datasets loaded (%s): %d" % (malware_type, len(datasets)))

    x, y = datasets[0]
    model = get_iforest_model(x)
    logger.debug("model created...")

    alpha = 0.1
    _, q_alpha = model.get_KL_divergence_distribution(x, alpha=alpha)
    p = model.get_node_sample_distributions(x)
    logger.debug("q_alpha: %f (alpha=%0.2f)" % (q_alpha, alpha))

    update_model = True

    for i in range(1, len(datasets)):
        x, y = datasets[i]
        kls, _ = model.get_KL_divergence_distribution(x, p=p)
        kl_mean = np.mean(kls)
        logger.debug("[%s] KL_mean: %f, q_alpha: %f" % (malware_list[i], np.mean(kls), q_alpha))
        if update_model and kl_mean > q_alpha:
            logger.debug("Updating model since %f > %f" % (kl_mean, q_alpha))
            model = get_iforest_model(x)
            _, q_alpha = model.get_KL_divergence_distribution(x, alpha=alpha)
            p = model.get_node_sample_distributions(x)
            logger.debug("q_alpha: %f (alpha=%0.2f)" % (q_alpha, alpha))


if __name__ == "__main__":
    test_concept_drift()
