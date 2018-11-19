import numpy.random as rnd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from common.gen_samples import *
from .simple_gcn import SimpleGCN, SimpleGCNAttack, set_random_seeds
from .gcn_test_support import read_graph_dataset, get_target_and_attack_nodes, \
    plot_graph, test_tensorflow_array_differentiation, test_marked_nodes, \
    gradients_to_arrow_texts, nodes_to_arrow_texts

"""
pythonw -m graph.test_gcn --debug --plot --log_file=temp/test_gcn.log --dataset=face_top
"""


def plot_labels_with_modified_node(gcn, y_hat, target_node, old_label, modified_node, node_val,
                                   lbl_color_map=None,
                                   marked_nodes=None, marked_colors=None, title=None, dp=None):
    """ Plots the annotated graph details

    The plotting constants such as arrow scale etc. are tuned
    for the 'face_top' dataset.
    """
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

    plot_graph(gcn.fit_x, y_hat, gcn.fit_A, lbl_color_map=lbl_color_map,
               marked_nodes=marked_nodes, marked_colors=marked_colors,
               nodes=False, edges=True, edges_title=title, arrow_texts=arrow_texts, dp=dp)

    gcn.fit_x[modified_node, :] = old_val  # restore previous value


def plot_model_diagnostics(attack_model, mod_node=None, attack_grads=None, pdfpath=None):
    """ Plots the annotated graph details

    The plotting constants such as arrow scale etc. are tuned
    for the 'face_top' dataset.
    """
    gcn = attack_model.gcn
    target_nodes = attack_model.target_nodes
    attack_nodes = attack_model.attack_nodes

    lbl_color_map = {-1: "grey", 0: "blue", 1: "red", 2: "green", 3: "orange"}
    m_nodes = [target_nodes, attack_nodes]
    m_colors = ['green', 'magenta']

    node_arrow_scale = np.array([-0.08, 0.04], dtype=np.float32)
    text_offset = node_arrow_scale + [-0.10, 0.02]
    node_arrow_texts = nodes_to_arrow_texts(gcn.fit_x, nodes=target_nodes, scale=node_arrow_scale,
                                            text="target", text_offset=text_offset,
                                            head_width=0.02, head_length=0.01)
    node_arrow_texts.extend(nodes_to_arrow_texts(gcn.fit_x, nodes=attack_nodes, scale=node_arrow_scale,
                                                 text="attacker", text_offset=text_offset,
                                                 head_width=0.02, head_length=0.01))

    y_hat = gcn.predict()
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2, save_tight=True)
    plot_graph(gcn.fit_x, gcn.fit_y, gcn.fit_A, lbl_color_map=lbl_color_map,
               marked_nodes=m_nodes, marked_colors=m_colors,
               nodes=True, edges=True, arrow_texts=node_arrow_texts,
               nodes_title=r"${\bf (a)}$ Synthetic Semi-supervised Dataset",
               edges_title=r"${\bf (b)}$ Graph by Joining Nearest Neighbors", dp=dp)

    grad_arrow_scale = np.array([-0.025, -0.07], dtype=np.float32)
    node_arrow_texts = gradients_to_arrow_texts(gcn.fit_x, grads=attack_grads,
                                                scale=grad_arrow_scale,
                                                text_offset=grad_arrow_scale + [-0.28, -0.04],
                                                head_width=0.02, head_length=0.01)
    plot_graph(gcn.fit_x, y_hat, gcn.fit_A, lbl_color_map=lbl_color_map,
               marked_nodes=m_nodes, marked_colors=m_colors,
               nodes=False, edges=True, edges_title=r"${\bf (c)}$ Predicted Labels",
               arrow_texts=node_arrow_texts, dp=dp)

    if mod_node is not None:
        target_node, old_label, modified_node, node_val = mod_node
        y_hat_mod = attack_model.modify_gcn_and_predict(node=modified_node, node_val=node_val, retrain=False)
        plot_labels_with_modified_node(gcn, y_hat_mod, target_node, old_label, modified_node, node_val,
                                       lbl_color_map=lbl_color_map, marked_nodes=m_nodes, marked_colors=m_colors,
                                       title=r"${\bf (d)}$ Predicted Labels on Modified Graph", dp=dp)
    dp.close()


def test_gcn(args):
    dataset = args.dataset

    # Changing the below will change the output plots as well
    sub_sample = 0.3
    labeled_frac = 0.3
    n_neighbors = 5  # includes self

    x, y, y_orig, A = read_graph_dataset(dataset, sub_sample=sub_sample,
                                         labeled_frac=labeled_frac,
                                         n_neighbors=n_neighbors,
                                         euclidean=False)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y)+1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    logger.debug("A:\n%s" % str(A))

    target_nodes, attack_nodes = get_target_and_attack_nodes(x, dataset)

    learning_rate = 0.1
    l2_lambda = 0.001
    search_along_max_grad_feature = True

    # Two NN layers implies max two-hop information propagation
    # through graph by the GCN in each forward/backward propagation.
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

    f1 = gcn.get_f1_score(y_orig)
    logger.debug("f1 score: %f" % f1)

    mod_node = None
    if len(target_nodes) > 0:
        attack_model = SimpleGCNAttack(gcn=gcn, target_nodes=target_nodes,
                                       attack_nodes=attack_nodes)
        best, all_grads = attack_model.suggest_node()
        if best is not None:
            target_node, old_label, attack_node, feature, grads = best
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
            if mod_val is not None:
                mod_node = (target_node, old_label, attack_node, mod_val)
                logger.debug("Suggested node: %d, feature: %d, grads: %s" % (attack_node, feature, grads))

        if args.plot and x.shape[1] == 2:  # plot only if 2D dataset
            fsig = "%s_n%d_l%d" % (dataset, n_neighbors, len(n_neurons))
            pdfpath = "temp/test_gcn_%s.pdf" % (fsig)
            plot_model_diagnostics(attack_model, mod_node=mod_node, attack_grads=all_grads, pdfpath=pdfpath)

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

