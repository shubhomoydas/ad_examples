import numpy.random as rnd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from common.gen_samples import *
from .simple_gcn import SimpleGCN, AttackModel, set_random_seeds, get_f1_score
from .gcn_test_support import read_graph_dataset, get_target_and_attack_nodes, \
    plot_graph, test_tensorflow_array_differentiation, test_marked_nodes, \
    gradients_to_arrow_texts, nodes_to_arrow_texts

"""
pythonw -m graph.test_gcn --debug --plot --log_file=temp/test_gcn.log --dataset=synth_graph
"""


def plot_labels_with_modified_node(gcn, y_hat, target_node, old_label, modified_node, node_val,
                                   marked_nodes=None, marked_colors=None, title=None, dp=None):

    old_val = np.copy(gcn.fit_x[modified_node, :])  # save previous value
    gcn.fit_x[modified_node, :] = node_val

    changed = y_hat[target_node] != old_label
    scale = np.array([-0.05, 0.08], dtype=np.float32)
    start_pos = gcn.fit_x[target_node]
    text_pos = start_pos + scale + [-0.18, 0.04]
    arrow_texts = [{"start_pos": start_pos, "text_pos": text_pos,
                    "scale": scale,
                    "text": "label changed" if changed else "label unchanged",
                    "head_width": 0.01,
                    "head_length": 0.02}]

    plot_graph(gcn.fit_x, y_hat, gcn.fit_A,
               marked_nodes=marked_nodes, marked_colors=marked_colors,
               nodes=False, edges=True, edges_title=title, arrow_texts=arrow_texts, dp=dp)

    gcn.fit_x[modified_node, :] = old_val  # restore previous value


def modify_and_predict_gcn(gcn, node, node_val, retrain=False):
    """ Modifies the node in the graph and then predicts labels """

    x, y, A = gcn.fit_x, gcn.fit_y, gcn.fit_A
    old_val = np.copy(x[node, :])  # save previous value
    x[node, :] = node_val
    if retrain:
        mod_gcn = SimpleGCN(n_neurons=gcn.n_neurons, activations=gcn.activations,
                            n_classes=gcn.n_classes, max_epochs=gcn.max_epochs,
                            learning_rate=gcn.learning_rate, l2_lambda=gcn.l2_lambda)
        mod_gcn.fit(x, y, A)
    else:
        mod_gcn = gcn

    y_hat = mod_gcn.predict()

    x[node, :] = old_val  # restore previous value

    return y_hat


def find_minimum_modification(gcn, target_node, mod_node, old_label, search_direction):
    """ Search along search_direction for mod_node until label of target_node flips """
    min_val = 0.0
    max_val = 5.0
    max_iters = 15
    prod = 0.5
    orig_val = np.copy(gcn.fit_x[mod_node, :])
    mod_val = None
    for i in range(max_iters):
        node_val = orig_val + prod * search_direction
        y_hat = modify_and_predict_gcn(gcn, node=mod_node, node_val=node_val, retrain=False)
        if y_hat[target_node] != old_label:
            mod_val = node_val
            if prod < 0.01:
                break
            max_val = prod
        else:
            min_val = prod
        prod = (min_val + max_val) / 2
    logger.debug("prod: %f; mod_val: %s" % (prod, "" if mod_val is None else str(mod_val)))
    return mod_val


def plot_model_diagnostics(gcn, target_nodes=None, attack_nodes=None,
                           mod_node=None, attack_grads=None, pdfpath=None):
    m_nodes = [target_nodes, attack_nodes]
    m_colors = ['green', 'magenta']

    node_arrow_texts = nodes_to_arrow_texts(gcn.fit_x, nodes=target_nodes, scale=[-0.08, 0.04],
                                            text="target", head_width=0.02, head_length=0.01)
    node_arrow_texts.extend(nodes_to_arrow_texts(gcn.fit_x, nodes=attack_nodes, scale=[-0.08, 0.04],
                                                 text="attacker", head_width=0.02, head_length=0.01))

    y_hat = gcn.predict()
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2, save_tight=True)
    plot_graph(gcn.fit_x, gcn.fit_y, gcn.fit_A,
               marked_nodes=m_nodes, marked_colors=m_colors,
               nodes=True, edges=True, arrow_texts=node_arrow_texts,
               nodes_title=r"${\bf (a)}$ Synthetic Semi-supervised Dataset",
               edges_title=r"${\bf (b)}$ Graph by Joining Nearest Neighbors", dp=dp)
    node_arrow_texts = gradients_to_arrow_texts(gcn.fit_x, grads=attack_grads,
                                           scale=[-0.025, -0.07], head_width=0.02, head_length=0.01)
    plot_graph(gcn.fit_x, y_hat, gcn.fit_A,
               marked_nodes=m_nodes, marked_colors=m_colors,
               nodes=False, edges=True, edges_title=r"${\bf (c)}$ Predicted Labels",
               arrow_texts=node_arrow_texts, dp=dp)

    if mod_node is not None:
        target_node, old_label, modified_node, node_val = mod_node
        y_hat_mod = modify_and_predict_gcn(gcn, node=modified_node, node_val=node_val, retrain=False)
        plot_labels_with_modified_node(gcn, y_hat_mod, target_node, old_label, modified_node, node_val,
                                       marked_nodes=m_nodes, marked_colors=m_colors,
                                       title=r"${\bf (d)}$ Predicted Labels on Modified Graph", dp=dp)
    dp.close()


def test_gcn(args):
    dataset = args.dataset

    sub_sample = 0.3
    labeled_frac = 0.3
    n_neighbors = 5  # includes self
    euclidean = False

    x, y, y_orig, A = read_graph_dataset(dataset, sub_sample=sub_sample,
                                         labeled_frac=labeled_frac,
                                         n_neighbors=n_neighbors,
                                         euclidean=euclidean)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y)+1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    logger.debug("A:\n%s" % str(A))

    target_nodes, attack_nodes = get_target_and_attack_nodes(x, dataset)

    learning_rate = 0.1
    l2_lambda = 0.001

    # Two NN layers implies max two-hop information propagation
    # through graph by the GCN in each training epoch.
    n_neurons = [10, n_classes]
    activations = [tf.nn.leaky_relu, None]
    # activations = [None, None]
    # activations = [tf.nn.sigmoid, None]
    # activations = [tf.nn.tanh, None]
    gcn = SimpleGCN(n_neurons=n_neurons, activations=activations,
                    n_classes=n_classes, max_epochs=5000,
                    learning_rate=learning_rate, l2_lambda=l2_lambda,
                    rand_seed=args.randseed+2)

    gcn.fit(x, y, A)

    f1 = get_f1_score(gcn, y, y_orig)
    logger.debug("f1 score: %f" % f1)

    mod_node = None
    all_grads = None
    if len(target_nodes) > 0:
        atk = AttackModel(model=gcn, target_nodes=target_nodes, attack_nodes=attack_nodes)
        best, all_grads = atk.suggest_node()
        if best is not None:
            target_node, old_label, attack_node, feature, grads = best
            search_direction = np.zeros(grads.shape, dtype=grads.dtype)
            search_direction[feature] = grads[feature]
            mod_val = find_minimum_modification(gcn, target_node=target_node, mod_node=attack_node,
                                                old_label=old_label,
                                                search_direction=search_direction)
            if mod_val is not None:
                mod_node = (target_node, old_label, attack_node, mod_val)
                logger.debug("Suggested node: %d, feature: %d, grads: %s" % (attack_node, feature, grads))

    if args.plot:
        fsig = "%s_n%d_l%d%s" % (dataset, n_neighbors, len(n_neurons), "" if not euclidean else "_euc")
        pdfpath = "temp/test_gcn_%s.pdf" % (fsig)
        plot_model_diagnostics(gcn, target_nodes, attack_nodes,
                               mod_node=mod_node, attack_grads=all_grads, pdfpath=pdfpath)

    gcn.close_session()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_command_args(debug=False, debug_args=["--debug",
                                                     "--plot",
                                                     "--log_file=temp/test_adjacency.log"])
    configure_logger(args)

    set_random_seeds(args.randseed, args.randseed + 1, args.randseed + 2)

    # test_tensorflow_array_differentiation()
    # test_marked_nodes(args)
    test_gcn(args)

