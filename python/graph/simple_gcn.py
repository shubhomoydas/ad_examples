import numpy.random as rnd
from sklearn.metrics import f1_score
import tensorflow as tf
from common.gen_samples import *


def set_random_seeds(py_seed=42, np_seed=42, tf_seed=42):
    """ Set random seeds for Python, numpy, TensorFlow graph """
    random.seed(py_seed)
    rnd.seed(np_seed)
    tf.set_random_seed(tf_seed)


def get_nn_layer(layers, layer_from_top=1):
    return layers[len(layers) - layer_from_top]


def euclidean_dist(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist


def sign(x):
    return -1. if x < 0 else 0. if x == 0 else 1.


class GraphAdjacency(object):
    """ Encapsulates methods for computing the adjacency matrix with nearest neighbors

    Attributes:
        n_neighbors: int
            Number of nearest neighbors *including* self
        euclidean: bool
            If False, populates output matrix with 1s.
            If True, populates output matrix with euclidean
                similarity (1=identical, 0=very different)
        sig2: float
            Variance for euclidean similarity
        self_loops: bool
            True: set the diagonal elements to 1
            False: set diagonal elements to 0
    """

    def __init__(self, n_neighbors=10, euclidean=False, sig2=1.0, self_loops=True):
        self.n_neighbors = n_neighbors
        self.euclidean = euclidean
        self.sig2 = sig2
        self.self_loops = self_loops

    def build_adjacency(self, x_in):
        n = nrow(x_in)
        x = normalize_and_center_by_feature_range(x_in)
        dists = np.zeros(shape=(n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                dists[i, j] = euclidean_dist(x[i, :], x[j, :])
                dists[j, i] = dists[i, j]

        neighbors = np.zeros(shape=(n, self.n_neighbors), dtype=int)
        for i in range(n):
            neighbors[i, :] = np.argsort(dists[i, :])[0:self.n_neighbors]

        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in neighbors[i, :]:
                # ignore diagonal elements of W
                if i != j:
                    if self.euclidean:
                        A[i, j] = np.exp(-(dists[i, j]**2) / self.sig2)
                    else:
                        A[i, j] = 1
                    A[j, i] = A[i, j]  # undirected edge

        if self.self_loops:
            A += np.eye(n)  # adding self-loops

        return A

    def sample_edges(self, A, prob=1.0):
        r, c = np.where(A > 0)  # row, column
        # logger.debug("r:\n%s" % str(r))
        # logger.debug("c:\n%s" % str(c))
        # Assume that A is symmetric and get all edges excluding self-loops.
        upper_diag_indexes = np.where(r < c)[0]
        # logger.debug("r upper:\n%s" % str(r[upper_diag_indexes]))
        # logger.debug("c upper:\n%s" % str(c[upper_diag_indexes]))
        all_upper = np.arange(len(upper_diag_indexes), dtype=np.int32)
        np.random.shuffle(all_upper)
        all_upper = all_upper[0:int(prob*len(all_upper))]
        sampled_edges = upper_diag_indexes[all_upper]
        A_new = np.zeros(A.shape, dtype=A.dtype)
        for i, j in zip(r[sampled_edges], c[sampled_edges]):
            A_new[i, j] = 1
            A_new[j, i] = 1  # symmetric matrix
        if self.self_loops:
            A_new += np.eye(A_new.shape[0])
        return A_new


class SimpleGCN(object):
    """ Implementation of a simple Graph Convolutional Network and APIs to support attack

    Reference(s):
        [1] Semi-Supervised Classification with Graph Convolutional Networks
            by Thomas N. Kipf and Max Welling, ICLR 2017
        [2] Adversarial Attacks on Neural Networks for Graph Data
            by Daniel Zugner, Amir Akbarnejad, and Stephan Gunnemann, KDD 2018
    """
    def __init__(self, input_shape, n_neurons, activations, n_classes,
                 name="gcn", graph=None,
                 learning_rate=0.005, l2_lambda=0.001, train_batch_size=25,
                 max_epochs=1, tol=1e-4, rand_seed=42):
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.activations = activations
        self.n_classes = n_classes
        self.name = name
        self.graph = graph
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.train_batch_size = train_batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.rand_seed = rand_seed

        self.layer_names = None
        self.n_features = self.input_shape[1]
        self.X = None
        self.y_labeled = None
        self.network = None
        self.A_hat = None
        self.D = None
        self.iDroot = None

        self.labeled_indexes = None
        self.z = None
        self.xentropy_loss = None
        self.train_loss_op = None
        self.preds = None

        self.class_enc = np.eye(self.n_classes, dtype=np.float32)

        self.fit_x = self.fit_y = self.fit_labeled_indexes = self.fit_A = self.fit_A_hat = None
        self.target = self.label_1 = self.label_2 = None
        self.X_above_attack_node = self.X_attack_node = self.X_below_attack_node = None
        self.attack_network = None
        self.attack_grad = None

        if self.graph is None:
            self.reset_graph()

        with self.graph.as_default():
            self.init_network()
            self.init_session()

    def dnn_layer(self, x, A, n_neurons, name, activation=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            n_inputs = int(x.get_shape()[1])
            stddev = 2. / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.get_variable("W", initializer=init)
            Z = tf.matmul(A, tf.matmul(x, W))
            if activation is not None:
                return activation(Z)
            else:
                return Z

    def dnn_construct(self, x, A, n_neurons, names, activations, reuse=False):
        layer_input = x
        layers = list()
        for i, name in enumerate(names):
            hidden = self.dnn_layer(layer_input, A, n_neurons=n_neurons[i],
                                    name=names[i], activation=activations[i], reuse=reuse)
            layers.append(hidden)
            layer_input = hidden
        return layers

    def init_network(self):
        n = self.input_shape[0]
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_features), name="%s_X" % self.name)
        self.y_labeled = tf.placeholder(tf.float32, shape=(None, self.n_classes), name="%s_y" % self.name)
        self.A_hat = tf.placeholder(tf.float32, shape=(n, n), name="%s_A" % self.name)
        self.iDroot = tf.placeholder(tf.float32, shape=(n, n), name="%s_iD" % self.name)
        self.layer_names = ['%s_layer_%d' % (self.name, i) for i, _ in enumerate(self.n_neurons)]
        with tf.variable_scope("%s_GCN" % self.name):
            self.network = self.dnn_construct(self.X, self.A_hat, self.n_neurons,
                                              self.layer_names, self.activations, reuse=False)
        # loss
        self.labeled_indexes = tf.placeholder(tf.int32, shape=(None), name="li")
        self.z = get_nn_layer(self.network, layer_from_top=1)
        logits_tensor = tf.gather(self.z, self.labeled_indexes, axis=0)
        self.xentropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_tensor,
                                                                                       labels=self.y_labeled))
        vars = tf.trainable_variables()
        params = [v for v in vars if v.name.startswith("%s" % self.name) and v.name.endswith('/W:0')]
        if len(params) == 0:
            raise ValueError("No trainable parameters")

        if self.l2_lambda > 0.0:
            self.xentropy_loss += self.l2_lambda * self.get_l2_regularizers(params)
        self.train_loss_op = self.training_op(self.xentropy_loss, params)

        # probability predictions
        self.preds = tf.nn.softmax(self.z)

        self.init_attack_network()

        vars = tf.trainable_variables()
        for v in vars: logger.debug(v.name)

    def init_attack_network(self):
        """
        Setup attack variables so that we can compute gradient of difference
        in logits wrt input attacker variables for arbitrary GCN networks.

        'label_1' would be set to the new label for the node (usually its
            second-best-predicted label)
        'label_2' would be set to the current label for the node (usually
            its best-predicted label)

        The attack matrix X_attack is setup such that the attack node's row is a
        TensorFlow variable for which we can compute the gradient. Other rows
        which are above it (i.e., X_above_attack_node) and below it
        (i.e., X_below_attack_node) are setup as fixed placeholders for which
        TensorFlow will not compute gradients.

        The full attack matrix would be row-wise concatenation:
                    [ self.X_above_attack_node ]  <- fixed placeholder
        X_attack =  [    self.X_attack_node    ]  <- Variable
                    [ self.X_below_attack_node ]  <- fixed placeholder
        """
        self.target = tf.placeholder(tf.int32, shape=(), name="%s_target" % self.name)
        self.label_1 = tf.placeholder(tf.int32, shape=(), name="%s_label_1" % self.name)
        self.label_2 = tf.placeholder(tf.int32, shape=(), name="%s_label_2" % self.name)

        self.X_above_attack_node = tf.placeholder(tf.float32, shape=(None, self.n_features), name="%s_x_pre" % self.name)
        self.X_attack_node = tf.Variable(tf.zeros([1, self.n_features]), name="%s_attacker" % self.name)
        self.X_below_attack_node = tf.placeholder(tf.float32, shape=(None, self.n_features), name="%s_x_pos" % self.name)
        self.X_attack = tf.concat([self.X_above_attack_node, self.X_attack_node, self.X_below_attack_node], axis=0)

        with tf.variable_scope("%s_GCN" % self.name, reuse=True):
            self.attack_network = self.dnn_construct(self.X_attack, self.A_hat, self.n_neurons,
                                                     self.layer_names, self.activations, reuse=True)
        # get the logits for the target node
        attack_logits = get_nn_layer(self.attack_network, layer_from_top=1)
        target_logits = attack_logits[self.target, :]

        # compute gradient of the difference of target logits wrt attack node
        self.attack_grad = tf.gradients(target_logits[self.label_1] - target_logits[self.label_2], [self.X_attack_node])

    def get_l2_regularizers(self, params):
        """ Returns L2 regularizer

        :param params: list of tf.Variable
            The model parameters

        :return: L2 regularizer loss
        """
        l2_loss = 0.0
        for v in params:
            l2_loss += tf.nn.l2_loss(v)
        return l2_loss

    def training_op(self, loss, var_list=None, use_adam=False):
        if use_adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                       200, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        return optimizer.minimize(loss, var_list=var_list)

    def init_session(self):
        self.session = tf.Session(graph=self.graph)

    def close_session(self):
        if self.session is not None:
            self.session.close()

    def reset_session(self):
        """ close existing session if any and start a new one """
        self.close_session()
        self.init_session()

    def reset_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.rand_seed)

    def get_loss(self):
        """ Use only for DEBUG """
        labeled_indexes = np.where(self.fit_y >= 0)[0]
        y_labeled = np.asarray(self.fit_y[labeled_indexes], dtype=int)
        y_labeled_enc = self.class_enc[y_labeled]
        feed_dict = {self.X: self.fit_x, self.A_hat: self.fit_A_hat, self.y_labeled: y_labeled_enc,
                     self.labeled_indexes: labeled_indexes}
        loss = self.session.run([self.xentropy_loss], feed_dict=feed_dict)[0]
        return loss

    def get_logits(self):
        """ Use only for DEBUG """
        with self.graph.as_default():
            labeled_indexes = np.where(self.fit_y >= 0)[0]
            feed_dict = {self.X: self.fit_x, self.A_hat: self.fit_A_hat,
                         self.labeled_indexes: labeled_indexes}
            logits_tensor = tf.gather(self.z, self.labeled_indexes, axis=0)
            logits = self.session.run([logits_tensor], feed_dict=feed_dict)[0]
        return logits

    def get_iDroot(self, A):
        D = A.sum(axis=1)
        iDroot = np.diag(np.sqrt(D) ** (-1))
        return D, iDroot

    def fit(self, x, y, A):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())
            self._fit(x, y, A)

    def _fit(self, x, y, A):
        """ Prepare the computation graph and train the network

        Note(s):
            By this point the default graph has already set to self.graph
            (see self.fit() above)
        """
        D, iDroot = self.get_iDroot(A)
        A_hat = np.dot(np.dot(iDroot, A), iDroot)
        self.fit_x = x
        self.fit_y = y
        self.fit_A = A
        self.fit_A_hat = A_hat
        labeled_indexes = np.where(self.fit_y >= 0)[0]
        y_labeled = np.asarray(self.fit_y[labeled_indexes], dtype=int)
        # logger.debug("y_labeled: %d\n%s" % (len(y_labeled), str(list(y_labeled))))
        y_labeled_enc = self.class_enc[y_labeled]
        # logger.debug("y_labeled_enc: %s" % (str(y_labeled_enc.shape)))

        fit_tm = Timer()
        feed_dict = {self.X: self.fit_x, self.y_labeled: y_labeled_enc,
                     self.labeled_indexes: labeled_indexes, self.A_hat: self.fit_A_hat}
        prev_loss = -np.infty
        for epoch in range(self.max_epochs):
            tm = Timer()
            _, loss = self.session.run([self.train_loss_op, self.xentropy_loss], feed_dict=feed_dict)
            if epoch > 0 and abs(loss - prev_loss) < self.tol:
                logger.debug("Exiting at epoch %d/%d (diff=%f)" % (epoch, self.max_epochs, abs(loss - prev_loss)))
                break
            if (epoch + 1) % 10 == 0:
                err = self.get_prediction_error()
                logger.debug(tm.message("[%d] loss: %f, pred_err: %f" % (epoch+1, loss, err)))
            prev_loss = loss

        logger.debug(fit_tm.message("SimpleGCN fitted (max epochs: %d)" % self.max_epochs))

    def get_x(self):
        return self.fit_x

    def decision_function(self):
        feed_dict = {self.X: self.fit_x, self.A_hat: self.fit_A_hat}
        preds = self.session.run([self.preds], feed_dict=feed_dict)[0]
        return preds

    def predict(self):
        preds = self.decision_function()
        y_hat = np.argmax(preds, axis=1)
        return y_hat

    def get_prediction_error(self):
        y_hat = self.predict()
        labeled_indexes = np.where(self.fit_y >= 0)[0]
        tp = np.where(self.fit_y[labeled_indexes] != y_hat[labeled_indexes])[0]
        # logger.debug("%d/%d" % (len(tp), len(labeled_indexes)))
        err = (1.0 * len(tp)) / len(labeled_indexes)
        return err

    def get_f1_score(self, y_orig):
        """ Returns F1 score

        :param y_orig: np.array
            Original labels for each node in the graph
        :return: float
        """
        unlabeled_indexes = np.where(self.fit_y < 0)[0]
        y_hat = self.predict()
        if False:
            logger.debug(y_orig)
            logger.debug("\ntrue:\n%s\npredicted:\n%s" % (str(list(y_orig[unlabeled_indexes])),
                                                          str(list(y_hat[unlabeled_indexes]))))
        f1 = f1_score(y_true=y_orig[unlabeled_indexes], y_pred=y_hat[unlabeled_indexes],
                      average='micro')
        return f1

    def grad_logit_diff_wrt_attacker(self, target, attacker, label_1, label_2):
        """ Computes the gradient of diff of target node logits wrt attacker node values

        :param target: int
        :param attacker: int
        :param label_1: int
        :param label_2: int
        :return: np.array
            grad(target_logits[label_1] - target_logits[label_2]) wrt attacker
        """
        # setup the attack X matrix where only the row corresponding to
        # the attack node is set as a variable and the rest are placeholders/fixed
        X_above_attack_node = self.fit_x[:attacker, :]
        assign_X_attack_node = self.X_attack_node.assign(self.fit_x[[attacker], :])
        self.session.run(assign_X_attack_node)
        if attacker < self.fit_x.shape[0]-1:
            X_below_attack_node = self.fit_x[(attacker+1):, :]
        else:
            # attacker node is the last node in the instance list; hence empty matrix
            X_below_attack_node = np.zeros((0, self.n_features), dtype=np.float32)

        g_v = self.session.run([self.attack_grad], feed_dict={self.A_hat: self.fit_A_hat,
                                                              self.target: target,
                                                              self.label_1: label_1,
                                                              self.label_2: label_2,
                                                              self.X_above_attack_node: X_above_attack_node,
                                                              self.X_below_attack_node: X_below_attack_node})[0]
        # logger.debug(g_v)
        g_v = g_v[0][0]
        return g_v

    def best_feature_wrt_attacker(self, target, attacker, old_label, new_label):
        """ Returns index of the feature along which the logit difference at the target is highest

        Treats the attacker values as the variables w.r.t which
        derivatives will be computed.

        :param target: int
            Index of target node
        :param attacker: int
            Inex of attack node
        :param old_label: int
            Old class label of target node
        :param new_label: int
            New class label for target node
        :return: int, np.array
            feature index, gradients for all features
        """
        grad_diffs = self.grad_logit_diff_wrt_attacker(target, attacker, new_label, old_label)
        abs_grad_diffs = np.abs(grad_diffs)
        most_change_feature = np.argmax(abs_grad_diffs)
        if False:
            logger.debug("grad_diffs:\n%s" % str(grad_diffs))
            logger.debug("abs_grad_diffs:\n%s" % str(abs_grad_diffs))
            logger.debug("most_change_feature: %d" % most_change_feature)
        return most_change_feature, grad_diffs


class SimpleGCNAttack(object):
    """ Implementation of feature modification-based attack on GCNs

    Reference(s):
        [1] Adversarial Attacks on Neural Networks for Graph Data
            by Daniel Zugner, Amir Akbarnejad, and Stephan Gunnemann, KDD 2018
    """

    def __init__(self, gcn, target_nodes, attack_nodes, min_prod=0.0, max_prod=5.0, max_iters=20):
        self.gcn = gcn
        self.target_nodes = target_nodes
        self.attack_nodes = attack_nodes
        self.min_prod = min_prod
        self.max_prod = max_prod
        self.max_iters = max_iters

    def suggest_node_feature(self, target_node, attack_node, old_label, new_label):
        """
        Suggest best feature of attacker node that can be modified such that
        the target node's label changes from old_label to new_label.

        :param target_node: int
        :param attack_node: int
        :param old_label: int
        :param new_label: int
        :return: int, np.array
            feature index, gradients wrt all input features
        """
        if len(self.target_nodes) > 0 and len(self.attack_nodes) > 0:
            best_feature, feature_grads = self.gcn.best_feature_wrt_attacker(target=target_node,
                                                                             attacker=attack_node,
                                                                             old_label=old_label,
                                                                             new_label=new_label)
            return best_feature, feature_grads

    def suggest_node(self):
        """ Suggests which is the best attack node and feature for modification

        Returns gradient wrt features so that the features can be modified
        such that label of target node changes to its next-best label.

        Note: Since this is just a simple demo, we only consider one target node.
        """
        probs = self.gcn.decision_function()
        sorted_probs = np.argsort(-probs, axis=1)
        y_hat = sorted_probs[:, 0]  # predicted best for all nodes
        y_hat_2 = sorted_probs[:, 1]  # predicted second-best for all nodes
        target_node = self.target_nodes[0]
        best_grad = 0.0
        best = None
        all_grads = []
        for attack_node in self.attack_nodes:
            old_label = y_hat[target_node]  # current predicted best label for target node
            new_label = y_hat_2[target_node]  # second-best predicted label for target node
            best_feature, feature_grads = self.suggest_node_feature(target_node, attack_node, old_label, new_label)
            all_grads.append((attack_node, best_feature, feature_grads))
            logger.debug("\nattack_node: %d, old_label: %d; new_label: %d; best_feature: %d; feature_grads: %s" %
                         (attack_node, old_label, new_label, best_feature, str(list(feature_grads))))
            if np.abs(feature_grads[best_feature]) > best_grad:
                best_grad = np.abs(feature_grads[best_feature])
                best = (target_node, old_label, attack_node, best_feature, feature_grads)
        return best, all_grads

    def modify_gcn_and_predict(self, node, node_val, retrain=False):
        """ Modifies the node in the graph and then predicts labels

        :param node: int
            The index of the node whose value will be modified
        :param node_val: np.array
            Value of the modified node
        :param retrain: bool
            Whether to retrain the GCN after modifying the instance before prediction
        :return: np.array
            Predicted labels for every node in the graph
        """

        x, y, A = self.gcn.fit_x, self.gcn.fit_y, self.gcn.fit_A
        old_val = np.copy(x[node, :])  # save previous value
        x[node, :] = node_val
        if retrain:
            mod_gcn = SimpleGCN(input_shape=self.gcn.input_shape, n_neurons=self.gcn.n_neurons,
                                activations=self.gcn.activations,
                                n_classes=self.gcn.n_classes, max_epochs=self.gcn.max_epochs,
                                learning_rate=self.gcn.learning_rate, l2_lambda=self.gcn.l2_lambda)
            mod_gcn.fit(x, y, A)
        else:
            mod_gcn = self.gcn

        y_hat = mod_gcn.predict()

        x[node, :] = old_val  # restore previous value

        return y_hat

    def find_minimum_modification(self, target_node, mod_node, old_label, search_direction):
        """ Search along search_direction for mod_node until label of target_node flips

        The goal is to find the smallest modification to the mod_node
        along the search_direction that results in the target node's label
        getting changed.

        :param target_node: int
            Index of target_node whose predicted label should be changed
        :param mod_node: int
            Index of node whose value will be changed
        :param old_label: int
            Old label of the target_node
        :param search_direction: np.array
            The direction of change -- usually determined by computing gradient of a function
        :return: np.array
            Modified value for mod_node that flips target node's label
            None is returned of no such value is found
        """
        if np.sum(search_direction ** 2) == 0.:
            # if search direction has zero magnitude, we cannot search at all
            return None
        min_prod = self.min_prod
        max_prod = self.max_prod
        prod = 0.5
        orig_val = np.copy(self.gcn.fit_x[mod_node, :])
        mod_val = None
        for i in range(self.max_iters):
            node_val = orig_val + prod * search_direction
            y_hat = self.modify_gcn_and_predict(node=mod_node, node_val=node_val, retrain=False)
            if y_hat[target_node] != old_label:
                mod_val = node_val
                if max_prod - prod < 1e-2 and mod_val is not None:
                    break
                max_prod = prod
            else:
                min_prod = prod
            prod = (min_prod + max_prod) / 2
        logger.debug("prod: %f; (max_prod - prod): %f; mod_val: %s" %
                     (prod, max_prod-prod, "" if mod_val is None else str(mod_val)))
        return mod_val

    def modify_structure(self):
        """ Attack by changing the graph adjacency matrix """
        raise NotImplementedError("modify_structure() not implemented yet")

