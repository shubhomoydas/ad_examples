import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
import tensorflow as tf

from ..common.utils import Timer, logger
from ..common.gen_samples import get_demo_samples
from ..common.data_plotter import DataPlotter
from .simple_gcn import GraphAdjacency, create_gcn_default, SimpleGCNAttack, AdversarialUpdater


def find_insts(x, x_range, y_range):
    """ Returns indexes of instances which lie within the input ranges"""
    insts = np.where(np.all(np.vstack([x[:, 0] >= x_range[0],
                                       x[:, 0] <  x_range[1],
                                       x[:, 1] >= y_range[0],
                                       x[:, 1] <  y_range[1]]), axis=0))[0]
    return insts


def read_face_dataset_with_labels():
    """ Reads the synthetic 'face' dataset which has two eyes, mouth, and nose

    :return: np.ndarray, np.array
    """
    x, y = get_demo_samples("face_")
    # label the parts of face as different classes:
    # the nose is labeled '1'
    nose = np.where(y == 1)[0]
    # right eye of face
    r_eye = np.where(np.all(np.vstack([y == 0, x[:, 0] < 0.5, x[:, 1] > 0.6]), axis=0))
    # left eye of face
    l_eye = np.where(np.all(np.vstack([y == 0, x[:, 0] > 0.5, x[:, 1] > 0.6]), axis=0))
    # mouth
    mouth = np.where(np.all(np.vstack([y == 0, x[:, 1] < 0.6]), axis=0))
    y[r_eye] = 0
    y[l_eye] = 1
    y[mouth] = 2
    y[nose] = 3
    return x, y


def read_synth_graph_dataset_with_labels(small=False):
    """ A simple connected graph with three classes of nodes

      2|         0(2)  0(3)
       |           \  /
      1|   2(11)    0(1)  1(7)
       |   |       /     /
    y 0|   2(10) 0(0)  1(6)
       |    \  /   \  /
     -1|      2(8)  1(4)
       |     /      |
     -2|   2(9)     1(5)
       |---------------------
       |  -2 -1  0  1  2  3
                  x

    """
    x = np.array([
        [ 0,  0],  # 0 (0)
        [ 1,  1],  # 0 (1)
        [ 0,  2],  # 0 (2)
        [ 2,  2],  # 0 (3)
        [ 1, -1],  # 1 (4)
        [ 1, -2],  # 1 (5)
        [ 2,  0],  # 1 (6)
        [ 3,  1],  # 1 (7)
        [-1, -1],  # 2 (8)
        [-2, -2],  # 2 (9)
        [-2,  0],  # 2 (10)
        [-2,  1],  # 2 (11)
    ], dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    if small:
        idxs = [0, 1, 4, 5, 8, 10]
        x, y = x[idxs], y[idxs]
    return x, y


def get_synth_graph_adjacency(small=False):
    """ Returns the adjacency matrix for the simple connected graph defined above """

    # first define only the upper triangular matrix of adjacency
    A = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    if small:
        idxs = [0, 1, 4, 5, 8, 10]
        A = A[idxs][:, idxs]

    # finally, make the matrix symmetric
    A += A.T

    return A


def sample_indexes(n, frac=1.0, random_state=None):
    random_state = check_random_state(random_state)
    n_hat = int(n * frac)
    rnd_idxs = np.arange(n, dtype=np.int32)
    random_state.shuffle(rnd_idxs)
    return rnd_idxs[:n_hat]


def read_dataset(dataset, sub_sample=1.0, labeled_frac=1.0):
    random_state = None
    if dataset == "face":
        x, y = read_face_dataset_with_labels()
        # since we use this dataset for illustrative purposes,
        # we should make sure that we always sample the same instances
        random_state = 40
    elif dataset == "face_top":
        x, y = read_face_dataset_with_labels()
        idxs = np.where(y != 2)  # exclude mouth
        x, y = x[idxs], y[idxs]
        # since we use this dataset for illustrative purposes,
        # we should make sure that we always sample the same instances
        random_state = 40
    elif dataset == "synth_graph":
        x, y = read_synth_graph_dataset_with_labels()
    elif dataset == "synth_graph_small":
        x, y = read_synth_graph_dataset_with_labels(small=True)
    else:
        raise ValueError("dataset '%s' not supported" % dataset)

    random_state = check_random_state(random_state)
    n_orig = x.shape[0]
    if dataset == "synth_graph" and labeled_frac < 1.0:
        y_orig = y
        labeled_idxs = [0, 4, 8]
        y_new = -1 * np.ones(len(y))
        y_new[labeled_idxs] = y[labeled_idxs]
        y = y_new
    elif dataset == "synth_graph_small" and labeled_frac < 1.0:
        y_orig = y
        labeled_idxs = [0, 2, 4]
        y_new = -1 * np.ones(len(y))
        y_new[labeled_idxs] = y[labeled_idxs]
        y = y_new
    else:
        if sub_sample < 1.0:
            rnd_idxs = sample_indexes(x.shape[0], sub_sample, random_state=random_state)
            x, y = x[rnd_idxs], y[rnd_idxs]
            logger.debug("sub-sampled %d/%d" % (x.shape[0], n_orig))

        y_orig = y
        if labeled_frac < 1.0:
            rnd_idxs = sample_indexes(x.shape[0], labeled_frac)
            y_new = -1 * np.ones(len(y))
            y_new[rnd_idxs] = y[rnd_idxs]
            y = y_new

    if dataset == "face" or dataset == "face_top":
        nose = np.where(y == 3)[0]
        y[nose] = -1

    return x, np.asarray(y, dtype=int), np.asarray(y_orig, dtype=int)


def read_graph_dataset(dataset, sub_sample=1.0, labeled_frac=1.0, n_neighbors=5, euclidean=False):
    x, y, y_orig = read_dataset(dataset, sub_sample=sub_sample,
                                labeled_frac=labeled_frac)

    if dataset == "face" or dataset == "face_top":
        ga = GraphAdjacency(n_neighbors=n_neighbors, euclidean=euclidean, self_loops=True)
        A = ga.build_adjacency(x)
    elif dataset == "synth_graph" or dataset == "synth_graph_small":
        A = get_synth_graph_adjacency(small=dataset == "synth_graph_small")
        A += np.eye(A.shape[0])  # add self-loops
    else:
        raise ValueError("dataset '%s' not defined" % dataset)
    return x, y, y_orig, A


def read_datasets_for_illustration(opts):
    """ Reads datasets used to create illlustrative figures """
    # Changing the below will change the output plots as well
    sub_sample = 0.3
    labeled_frac = 0.3

    x, y, y_orig, A = read_graph_dataset(opts.dataset, sub_sample=sub_sample,
                                         labeled_frac=labeled_frac,
                                         n_neighbors=opts.n_neighbors,
                                         euclidean=False)
    return x, y, y_orig, A


def gradients_to_arrow_texts(x, grads=None, scale=None, text_offset=None, head_width=0.02, head_length=0.01):
    """ Annotates gradients with arrow pointers """
    if grads is None or len(grads) == 0:
        return []
    arrow_texts = []
    if text_offset is None:
        text_offset = scale
    for node, best_feature, f_grads in grads:
        start_pos = x[node]
        text_pos = start_pos + text_offset
        arrow_text = {"start_pos": start_pos, "text_pos": text_pos,
                      "scale": scale,
                      "text": "grad: [%0.3f, %0.3f]" % (f_grads[0], f_grads[1]),
                      "head_width": head_width,
                      "head_length": head_length}
        arrow_texts.append(arrow_text)
    return arrow_texts


def nodes_to_arrow_texts(x, nodes, scale, text, text_offset=None, head_width=0.02, head_length=0.01):
    arrow_texts = []
    if nodes is None:
        return arrow_texts
    if text_offset is None:
        text_offset = scale
    for node in nodes:
        start_pos = x[node]
        text_pos = start_pos + text_offset
        arrow_text = {"start_pos": start_pos, "text_pos": text_pos,
                      "scale": scale,
                      "text": text,
                      "head_width": head_width,
                      "head_length": head_length}
        arrow_texts.append(arrow_text)
    return arrow_texts


def plot_arrow_texts(arrow_texts=None, fc='k', ec='k', pl=None):
    if arrow_texts is None:
        return

    for arrow_text in arrow_texts:
        start_pos = arrow_text.get("start_pos")
        text_pos = arrow_text.get("text_pos")
        scale = arrow_text.get("scale")
        text = arrow_text.get("text")
        head_width = arrow_text.get("head_width", 0.01)
        head_length = arrow_text.get("head_length", 0.02)
        pl.arrow(start_pos[0], start_pos[1], scale[0], scale[1],
                 head_width=head_width, head_length=head_length, fc=fc, ec=ec)
        pl.text(text_pos[0], text_pos[1], text, fontsize=6)


def plot_nodes(x, y, pl, dp, lbl_color_map=None,
               marked_nodes=None, marked_colors=None, arrow_texts=None,
               marker='x', s=10, facecolors='none', edgecolor=None, defaultcol='grey'):
    dp.plot_points(x, pl, labels=y, lbl_color_map=lbl_color_map,
                   marker=marker, s=s, facecolors=facecolors,
                   edgecolor=edgecolor, defaultcol=defaultcol)
    if marked_nodes is not None:
        for i, m_list in enumerate(marked_nodes):
            dp.plot_points(x[m_list], pl, labels=None, marker='o', s=38,
                           facecolors='none', edgecolor=marked_colors[i], defaultcol='none')
    plot_arrow_texts(arrow_texts=arrow_texts, pl=pl)


def plot_edges(x, A, pl):
    for i in range(x.shape[0]):
        neighbors = np.where(A[i, :] > 0)[0]
        for j in neighbors:
            pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], "-", color="gray", linewidth=0.5)


def plot_graph(x, y, A, lbl_color_map=None,
               marked_nodes=None, marked_colors=None,
               nodes_title=None, edges_title=None,
               nodes=True, edges=False, arrow_texts=None, dp=None):
    if nodes:
        pl = dp.get_next_plot()
        if nodes_title is not None:
            plt.title(nodes_title, fontsize=8)
        plot_nodes(x, y, pl, dp, lbl_color_map=lbl_color_map,
                   marked_nodes=marked_nodes, marked_colors=marked_colors,
                   arrow_texts=arrow_texts)
    if edges:
        pl = dp.get_next_plot()
        if edges_title is not None:
            plt.title(edges_title, fontsize=8)
        plot_nodes(x, y, pl, dp, lbl_color_map=lbl_color_map,
                   marked_nodes=marked_nodes, marked_colors=marked_colors,
                   arrow_texts=arrow_texts)
        plot_edges(x, A, pl)


def get_target_and_attack_nodes(x, dataset):
    """ Returns list of target and attack nodes for synthetic test datasets """
    if dataset == "face":
        target_nodes = find_insts(x, [0.75, 0.8], [0.7, 0.8])[:1]
        attack_nodes = find_insts(x, [0.6, 0.95], [0.8, 0.95])
    elif dataset == "face_top":
        target_nodes = find_insts(x, [0.4, 0.5], [0.9, 1.0])
        target_nodes = np.append(target_nodes, find_insts(x, [0.4, 0.6], [0.6, 0.7])[0])
        attack_nodes = find_insts(x, [0.6, 0.95], [0.8, 0.95])
    elif dataset == "synth_graph":
        target_nodes = []
        attack_nodes = []
    elif dataset == "synth_graph_small":
        target_nodes = []
        attack_nodes = []
    else:
        raise ValueError("dataset %s not supported" % dataset)
    logger.debug(target_nodes)
    logger.debug(attack_nodes)
    logger.debug("[%s] #target nodes: %d; #attack nodes: %d" % (dataset, len(target_nodes), len(attack_nodes)))
    return target_nodes, attack_nodes


def test_tensorflow_array_differentiation():
    """ Shows how the derivative can be computed wrt only one row of input matrix in TensorFlow"""
    X_above = tf.placeholder(dtype=tf.float32, shape=(None, 5))  # treat this as fixed
    X_row = tf.Variable(tf.zeros([1, 5], dtype=tf.float32), name="varX")  # gradient wrt this
    X_below = tf.placeholder(dtype=tf.float32, shape=(None, 5))  # treat this as fixed
    A = tf.placeholder(dtype=tf.float32, shape=(5, 2))
    X = tf.concat([X_above, X_row, X_below], axis=0)
    XA = tf.matmul(X, A)  # function whose gradient is required
    grad_XA_all = np.zeros((0, 5), dtype=np.float32)
    with tf.Session() as session:
        X_above_val = np.zeros(shape=(0, 5), dtype=np.float32)  # empty just for illustration
        X_row_val = np.array([2, 3, 5, 1, 3], dtype=np.float32).reshape((1, -1))
        assign_X_row = X_row.assign(X_row_val)
        session.run(assign_X_row)
        X_below_val = np.arange(20, dtype=np.float32).reshape((-1, 5))
        A_val = np.arange(10, dtype=np.float32).reshape((5, -1))
        X_val = X.eval(feed_dict={X_above: X_above_val, X_below: X_below_val, A: A_val})
        XA_val = XA.eval(feed_dict={X_above: X_above_val, X_below: X_below_val, A: A_val})
        for i in range(A.shape[1]):
            grad_XA = tf.gradients([XA[:, i]], [X_row])
            grad_XA_val = session.run([grad_XA], feed_dict={X_above: X_above_val, X_below: X_below_val, A: A_val})[0]
            logger.debug("grad_XA_val:\n%s" % str(grad_XA_val))
            grad_XA_all = np.vstack([grad_XA_all, grad_XA_val[0]])
    logger.debug("X_val:\n%s" % str(X_val))
    logger.debug("A_val:\n%s" % str(A_val))
    logger.debug("XA_val:\n%s" % str(XA_val))
    logger.debug("grad_XA_all:\n%s" % str(grad_XA_all))


def test_marked_nodes(opts):
    x, y, y_orig, A = read_datasets_for_illustration(opts)

    n_classes = len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    target_nodes, attack_nodes = get_target_and_attack_nodes(x, opts.dataset)

    fsig = opts.get_opts_name_prefix()
    lbl_color_map = {-1: "grey", 0: "blue", 1: "red", 2: "green", 3: "orange"}
    pdfpath = "%s/%s_marked_nodes.pdf" % (opts.results_dir, fsig)
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2)
    plot_graph(x, y, A=None, lbl_color_map=lbl_color_map,
               marked_nodes=[target_nodes, attack_nodes],
               marked_colors=['green', 'magenta'], edges=False, dp=dp)
    dp.close()


def test_edge_sample(opts):
    x, y, y_orig, A = read_datasets_for_illustration(opts)

    n_classes = len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    target_nodes, attack_nodes = get_target_and_attack_nodes(x, opts.dataset)

    lbl_color_map = {-1: "grey", 0: "blue", 1: "red", 2: "green", 3: "orange"}

    # logger.debug("\n%s" % str(A))
    ga = GraphAdjacency(opts.n_neighbors, self_loops=True)

    fsig = opts.get_opts_name_prefix()
    pdfpath = "%s/%s_edge_samples.pdf" % (opts.results_dir, fsig)
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2)
    for i in range(4):
        A_new = ga.sample_edges(A, prob=0.75)
        # logger.debug("\n%s" % str(A_new))
        plot_graph(x, y, A=A_new, lbl_color_map=lbl_color_map,
                   marked_nodes=[target_nodes, attack_nodes],
                   marked_colors=['green', 'magenta'], nodes=False, edges=True, dp=dp)
    dp.close()


def plot_labels_with_modified_node(gcn, y_hat, target_node, old_label, modified_node, node_val,
                                   lbl_color_map=None, marked_nodes=None, marked_colors=None,
                                   title=None, dp=None):
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


def plot_model_diagnostics(attack_model, target_nodes, attack_nodes, attack_details=None, pdfpath=None, opts=None):
    """ Plots the annotated graph details

    The plotting constants such as arrow scale etc. are tuned
    for the 'face_top' dataset.
    """
    gcn = attack_model.gcn

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
    gcn_sig = "%s GCN, layers: %d (%s)" % ("Adversarial" if opts.adversarial_train else
                                           "Ensemble (%d)" % opts.n_estimators if opts.ensemble else "Single",
                                           opts.n_layers, opts.activation_type)
    dp = DataPlotter(pdfpath=pdfpath, rows=2, cols=2, save_tight=True)
    plot_graph(gcn.fit_x, gcn.fit_y, gcn.fit_A, lbl_color_map=lbl_color_map,
               marked_nodes=m_nodes, marked_colors=m_colors,
               nodes=True, edges=True, arrow_texts=node_arrow_texts,
               nodes_title=r"${\bf (a)}$ Synthetic Semi-supervised Dataset",
               edges_title=r"${\bf (b)}$ Graph by Joining Nearest Neighbors", dp=dp)

    if attack_details is not None:
        for mod_node, feature_grads in attack_details:
            target_node, old_label, modified_node, node_val = mod_node
            target_gcn_sig = "[%d] %s" % (target_node, gcn_sig)
            grad_arrow_scale = np.array([-0.025, -0.07], dtype=np.float32)
            node_arrow_texts = gradients_to_arrow_texts(gcn.fit_x, grads=feature_grads,
                                                        scale=grad_arrow_scale,
                                                        text_offset=grad_arrow_scale + [-0.28, -0.04],
                                                        head_width=0.02, head_length=0.01)

            m_nodes = [[target_node], attack_nodes]
            plot_graph(gcn.fit_x, y_hat, gcn.fit_A, lbl_color_map=lbl_color_map,
                       marked_nodes=m_nodes, marked_colors=m_colors,
                       nodes=False, edges=True, edges_title=r"${\bf (c)}$ Predicted Labels" + "\n" + target_gcn_sig,
                       arrow_texts=node_arrow_texts, dp=dp)

            if node_val is not None:
                y_hat_mod = attack_model.modify_gcn_and_predict(node=modified_node, node_val=node_val, retrain=False)
                plot_labels_with_modified_node(gcn, y_hat_mod, target_node, old_label, modified_node, node_val,
                                               lbl_color_map=lbl_color_map, marked_nodes=m_nodes, marked_colors=m_colors,
                                               title=r"${\bf (d)}$ Predicted Labels on Modified Graph" + "\n" + target_gcn_sig, dp=dp)
    dp.close()


def test_create_gcn_default(opts):
    x, y, y_orig, A = read_datasets_for_illustration(opts)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y) + 1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    gcn = create_gcn_default(input_shape=x.shape, n_classes=n_classes, opts=opts)

    if True:
        gcn.fit(x, y, A)

        f1 = gcn.get_f1_score(y_orig)
        logger.debug("f1 score: %f" % f1)


def test_neighbor_gradients(opts):

    x, y, y_orig, A = read_datasets_for_illustration(opts)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y)+1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    gcn = create_gcn_default(input_shape=x.shape, n_classes=n_classes, opts=opts)

    gcn.fit(x, y, A)

    f1 = gcn.get_f1_score(y_orig)
    logger.debug("f1 score: %f" % f1)

    ga = GraphAdjacency(n_neighbors=opts.n_neighbors, self_loops=True)
    attack_model = SimpleGCNAttack(gcn)

    test_nodes = attack_model.get_top_uncertain_nodes(opts.n_vulnerable)
    logger.debug(test_nodes)

    attack_model = SimpleGCNAttack(gcn=gcn)

    neighbor_cache = dict()
    for i, test_node in enumerate(test_nodes):
        tm = Timer()
        neighbor_nodes = ga.sample_neighbors(test_node, gcn.fit_A, hops=opts.n_layers,
                                             n_neighbors=opts.n_sample_neighbors,
                                             neighbor_cache=neighbor_cache)
        # logger.debug(neighbor_nodes)
        best_attacks_for_each_target = attack_model.suggest_nodes([test_node], neighbor_nodes)

        all_attack_details = []
        for best, feature_grads in best_attacks_for_each_target:
            target_node, old_label, attack_node, feature, grads = best
            if attack_node is not None:
                mod_node = (target_node, old_label, attack_node, None)
                all_attack_details.append((mod_node, feature_grads))
            else:
                logger.debug("No attack node found for a target node %d (%s)" % (target_node, str(x[target_node, :])))
        logger.debug(tm.message("[%d/%d] node: %d, #neighbors: %d; time" %
                                (i+1, len(test_nodes), test_node, len(neighbor_nodes))))
        fsig = opts.get_opts_name_prefix()
        pdfpath = "%s/%s_vulnerable_nodes_%d.pdf" % (opts.results_dir, fsig, i+1)
        plot_model_diagnostics(attack_model, target_nodes=[test_node],
                               attack_nodes=neighbor_nodes,
                               attack_details=all_attack_details,
                               pdfpath=pdfpath, opts=opts)


def test_robust_training_helper(opts):
    x, y, y_orig, A = read_datasets_for_illustration(opts)

    # Number of classes includes the '0' class and excludes all marked '-1' i.e., unlabeled.
    n_classes = np.max(y) + 1  # len(np.unique(y[y >= 0]))
    logger.debug("n_classes: %d" % n_classes)

    gcn = create_gcn_default(input_shape=x.shape, n_classes=n_classes, opts=opts)

    gcn.fit(x, y, A)

    f1 = gcn.get_f1_score(y_orig)
    logger.debug("f1 score: %f" % f1)

    rh = AdversarialUpdater(attack_model=SimpleGCNAttack(gcn=gcn), opts=opts)

    prev_node_values = rh.select_and_update_nodes(gcn)
    if prev_node_values is not None:
        logger.debug("#updates: %d" % len(prev_node_values[0]))

    nodes = [node for node in prev_node_values[0]]
    logger.debug("before update:\n%s" % str(prev_node_values[1]))

    logger.debug("after update:\n%s" % str(gcn.fit_x[nodes, :]))

    rh.restore_values(gcn, node_values=prev_node_values)
    logger.debug("after restore:\n%s" % str(gcn.fit_x[nodes, :]))

