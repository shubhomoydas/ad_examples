import os
import logging
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ..common.utils import dir_create, get_command_args, configure_logger
from .simple_gcn import SimpleGCNAttack, AdversarialUpdater, set_random_seeds, \
    get_gcn_option_list, GcnOpts, GraphAdjacency, create_gcn_default
from .gcn_test_support import read_datasets_for_illustration, get_target_and_attack_nodes, \
    plot_graph, gradients_to_arrow_texts, nodes_to_arrow_texts, test_create_gcn_default, \
    test_edge_sample, test_tensorflow_array_differentiation, test_marked_nodes, \
    test_neighbor_gradients, test_robust_training_helper, plot_model_diagnostics

"""
pythonw -m ad_examples.graph.test_gcn --debug --plot --results_dir=./temp/gcn --log_file=temp/test_gcn.log --dataset=face_top
pythonw -m ad_examples.graph.test_gcn --debug --plot --results_dir=./temp/gcn --log_file=temp/test_gcn.log --dataset=face_top --ensemble --n_estimators=10 --edge_sample_prob=0.6
"""


def test_gcn(opts):

    x, y, y_orig, A = read_datasets_for_illustration(opts)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y)+1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    # logger.debug("A:\n%s" % str(A))

    target_nodes, attack_nodes = get_target_and_attack_nodes(x, opts.dataset)
    target_nodes = target_nodes[:1]  # focus on just one target for illustration

    search_along_max_grad_feature = True

    gcn = create_gcn_default(input_shape=x.shape, n_classes=n_classes, opts=opts)

    gcn.fit(x, y, A)

    f1 = gcn.get_f1_score(y_orig)
    logger.debug("f1 score: %f" % f1)

    if len(target_nodes) > 0:
        attack_model = SimpleGCNAttack(gcn=gcn)
        best_attacks_for_each_target = attack_model.suggest_nodes(target_nodes, attack_nodes)

        if best_attacks_for_each_target is None or len(best_attacks_for_each_target) == 0:
            logger.debug("No suitable attack vectors were found...")
            print("No suitable attack vectors were found...")

        all_attack_details = []
        for best, feature_grads in best_attacks_for_each_target:
            target_node, old_label, attack_node, feature, grads = best
            if attack_node is not None:
                search_direction = np.zeros(grads.shape, dtype=grads.dtype)
                if search_along_max_grad_feature:
                    # as in nettack paper (Zugner et al., 2018)
                    search_direction[feature] = grads[feature]
                else:
                    search_direction[:] = grads[:]
                mod_val = attack_model.find_minimum_modification(target_node=target_node,
                                                                 mod_node=attack_node,
                                                                 old_label=old_label,
                                                                 search_direction=search_direction)
                mod_node = (target_node, old_label, attack_node, mod_val)
                if mod_val is not None:
                    logger.debug("Suggested node: %d, feature: %d, grads: %s" % (attack_node, feature, grads))
                all_attack_details.append((mod_node, feature_grads))
            else:
                logger.debug("No attack node found for a target node %d (%s)" % (target_node, str(x[target_node, :])))
                print("No attack node found for a target node %d (%s)" % (target_node, str(x[target_node, :])))

        if opts.plot and x.shape[1] == 2:  # plot only if 2D dataset
            fsig = opts.get_opts_name_prefix()
            pdfpath = "%s/%s.pdf" % (opts.results_dir, fsig)
            plot_model_diagnostics(attack_model, target_nodes=target_nodes,
                                   attack_nodes=attack_nodes,
                                   attack_details=all_attack_details,
                                   pdfpath=pdfpath, opts=opts)

    gcn.close_session()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    dir_create("./temp/gcn")

    args = get_command_args(debug=False, parser=get_gcn_option_list())

    dir_create(args.results_dir)
    configure_logger(args)

    opts = GcnOpts(args)
    set_random_seeds(opts.randseed, opts.randseed + 1, opts.randseed + 2)

    # test_tensorflow_array_differentiation()
    # test_marked_nodes(opts)
    # test_edge_sample(opts)
    # test_create_gcn_default(opts)
    test_neighbor_gradients(opts)
    # test_robust_training_helper(opts)
    # test_gcn(opts)
