"""
Active Feature Space Suppression (AFSS)

This is used by the GLAD which puts a local flavor on global anomaly detectors
with analyst feedback.

Reference(s):
    Das, S. and Doppa, J.R. (2018). GLAD: GLocalized Anomaly Detection via
            Active Feature Space Suppression. (https://arxiv.org/abs/1810.01403)
"""
import numpy as np
import tensorflow as tf
import numpy.random as rnd
from common.gen_samples import *


def suppression_layer(x, n_neurons, name, activation=tf.nn.sigmoid):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        limit = np.sqrt(6 / (n_inputs + n_neurons))  # somewhat like glorot...
        init = tf.random_uniform((n_inputs, n_neurons), minval=0, maxval=limit, dtype=tf.float32)
        W = tf.Variable(init, name="W")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(x, W) + b
        if activation is not None:
            return activation(Z), W, b
        else:
            return Z, W, b


def construct_network(x, n_neurons, names, activations):
    layer_input = x
    layers = list()
    weights = list()
    biases = list()
    with tf.name_scope("dnn"):
        for i, name in enumerate(names):
            hidden, W, b = suppression_layer(layer_input, n_neurons=n_neurons[i], name=names[i],
                                             activation=activations[i])
            layers.append(hidden)
            weights.append(W)
            biases.append(b)
            layer_input = hidden
    return layers, weights, biases


def partition_instances(x, y, ensemble_scores, hf):
    mask = np.ones(x.shape[0], dtype=int)
    mask[hf] = 0
    ids = np.where(mask == 1)[0]
    x_unlabeled = x[ids]
    y_unlabeled = y[ids]
    scores_unlabeled = ensemble_scores[ids]
    x_labeled = x[hf]
    y_labeled = y[hf]
    scores_labeled = ensemble_scores[hf]
    return x_unlabeled, y_unlabeled, scores_unlabeled, x_labeled, y_labeled, scores_labeled


def get_unlabeled_batches(x, batch_size=-1, shuffle=False):
    n = x.shape[0]
    if batch_size < 0:
        batch_size = n
    indxs = np.arange(n)
    if shuffle:
        rnd.shuffle(indxs)
    for i in range(0, n, batch_size):
        et = min(i+batch_size, n)
        yield x[indxs[i:et], :]


def get_afss_batches(x_labeled, y_labeled, scores_labeled,
                     x_unlabeled, scores_unlabeled,
                     batch_size=-1, n_labeled_reps=1, lambda_prior=1.0):
    """ Returns batches of a mix of labeled and unlabeled instances

    :param x_labeled: np.ndarray
        Labeled instances
    :param y_labeled: np.array
        Labels for labeled instances
    :param scores_labeled: np.ndarray
        Unsupervised ensemble scores for labeled instances
    :param x_unlabeled: np.ndarray
        Unlabeled instances
    :param scores_unlabeled: np.ndarray
        Unsupervised ensemble scores for unlabeled instances
    :param batch_size: int
    :param n_labeled_reps: int
        Number of times labeled instances will be repeated. When the number of
        labeled examples is *much* lower than the number of unlabeled instances,
        the AAD loss due to the labeled instances might get smudged out by the
        prior loss of the labeled instances. Hence we try to overcome this by
        over sampling the labeled examples. Over sampling and randomly mixing
        labeled instances with unlabeled instances will hopefully help the network
        do a better job in optimization with SGD.
    :param lambda_prior: float
        If lambda_prior is 0.0, then no unlabeled instances will be included.
        In AFSS, the unlabeled instances only enforce the prior loss; they are
        not required if lambda_prior=0
    :return: np.ndarray

    Note:
         1. The labeled and unlabeled instances are randomly mixed.
         2. The returned batch will have the labeled instances (if at all)
            at the top, and the unlabeled (if at all) next.
         3. The returned y will be empty array in case there are no labeled
            instances in the batch, else it will have the labels for only
            the labeled instances in the batch in the corresponding order.
         4. Score matrices must be compatible with the instance matrices.
    """
    if x_labeled is not None and x_labeled.shape[0] > 0 and n_labeled_reps > 1 and lambda_prior > 0:
        # here we are up-sampling the labeled examples. We might as well down-sample
        # the unlabeled instances. Something to do later.
        # logger.debug("labeled instances too few; repeating them %d times" % n_labeled_reps)
        x_labeled = np.repeat(x_labeled, n_labeled_reps, axis=0)
        y_labeled = np.repeat(y_labeled, n_labeled_reps, axis=0)
        scores_labeled = np.repeat(scores_labeled, n_labeled_reps, axis=0)

    n_labeled = x_labeled.shape[0]
    if lambda_prior > 0:
        n_unlabeled = x_unlabeled.shape[0]
        n = n_labeled + n_unlabeled
        x = np.vstack([x_labeled, x_unlabeled])
        scores = np.vstack([scores_labeled, scores_unlabeled])
    else:
        # do not add unlabeled instance of lambda_prior=0
        logger.debug("Not adding unlabeled instances to batch; lambda_prior: %0.2f" % lambda_prior)
        n = n_labeled
        x = x_labeled
        scores = scores_labeled
    y = np.ones(n, dtype=y_labeled.dtype) * -1
    y[0:n_labeled] = y_labeled

    if batch_size < 0:
        batch_size = n
    indxs = np.arange(n)
    rnd.shuffle(indxs)
    for i in range(0, n, batch_size):
        et = min(i+batch_size, n)
        batch_x = x[indxs[i:et], :]
        batch_y = y[indxs[i:et]]
        batch_scores = scores[indxs[i:et], :]
        y_unl_inds = np.where(batch_y == -1)[0]
        b_n_unl = len(y_unl_inds)
        b_n_lbl = len(batch_y) - b_n_unl
        sorted_y = np.argsort(-batch_y)
        yield batch_x[sorted_y], batch_y[sorted_y[0:b_n_lbl]], batch_scores[sorted_y], b_n_lbl, b_n_unl


class AFSS(object):
    """ Active Feature Space Suppression (AFSS)

    Setup a network that outputs a value close to <bias_prob> initially for any part
    of the feature space. We then add additional loss objectives to the output of this
    network in order to suppress parts of the feature space.
    """
    def __init__(self, n_neurons, names, activations,
                 feature_ranges=None, bias_prob=0.50, prime=True,
                 c_q_tau=1.0, c_x_tau=1.0, lambda_prior=1.0,
                 l2_penalty=True, l2_lambda=0.001, train_batch_size=25,
                 max_init_epochs=5, max_afss_epochs=5, init_tol=1e-4, max_labeled_reps=1):
        if activations[len(activations)-1] is not None:
            raise ValueError("The last layer should not have any activation function.")
        self.n_neurons = n_neurons
        self.names = names
        self.activations = activations
        self.bias_prob = bias_prob
        self.prime = prime
        self.c_q_tau = c_q_tau
        self.c_x_tau = c_x_tau
        self.lambda_prior = lambda_prior
        self.l2_penalty = l2_penalty
        self.l2_lambda = l2_lambda
        self.max_init_epochs = max_init_epochs
        self.max_afss_epochs = max_afss_epochs
        self.init_tol = init_tol
        self.feature_ranges = feature_ranges
        self.train_batch_size = train_batch_size
        self.max_labeled_reps = max_labeled_reps
        self.n_inputs = None
        self.X = None
        self.Y = None
        self.layers = None  # The Feature Space Suppression Network
        self.weights = None
        self.biases = None
        self.session = None
        self.feature_space_prior_loss = None
        self.suppression_network_l2_loss = None
        self.feature_space_prior_training_op = None
        self.n_labeled = None
        self.n_unlabeled = None
        self.q_tau = None
        self.ensemble_scores = None
        self.integrated_loss = None
        self.aad_loss = None
        self.integrated_training_op = None

        self.y_mod = self.afss_output = self.a_score_all = self.a_score = None

        if self.bias_prob <= 0.0 or self.bias_prob >= 1.0:
            raise ValueError("bias_prob must be between 0 and 1. It cannot be 0.0 or 1.0. "
                             "E.g., try something like 0.99 instead of 1.0.")

    def get_param_values(self):
        out_params = list()
        out_params.extend(self.weights)
        out_params.extend(self.biases)
        with tf.name_scope("get_weights"):
            out_values = self.session.run(out_params, feed_dict={})
        return out_values

    def set_initial_bias_weight(self, p):
        """ Set the bias term so that output is self.bias_prob """
        values = self.get_param_values()
        last_W_val = values[len(self.layers) - 1]
        last_b = self.biases[len(self.layers) - 1]
        """
        1 / (1 + exp(-(b + Wx))) = p
        => (1 - p) = p * exp(-(b + Wx))
        => log((1 - p) / p) = -b - Wx
        => b = log(p / (1 - p)) - Wx
        """
        last_b_val = np.log(p / (1 - p)) - np.sum(last_W_val, axis=0)
        # logger.debug(last_W_val)
        # logger.debug("last_b_val (%s): %s" % (str(last_b_val.shape), str(last_b_val)))

        assign_op = last_b.assign(last_b_val)
        self.session.run(assign_op)
        # logger.debug("bias: %s" % (str(last_b.eval(session=self.session))))
        # z = last_b_val + np.sum(last_W_val, axis=0)
        # logger.debug("z when all nodes are 1.0: %s" % str(1. / (1 + np.exp(-z))))
        # logger.debug("prob when all nodes are 1.0: %s" % str(1. / (1 + np.exp(-z))))

    def get_suppression_network_l2_loss(self):
        l2_loss = 0.0
        for W in self.weights:
            l2_loss += tf.nn.l2_loss(W)
        return self.l2_lambda * l2_loss

    def init_network(self, x, prime_network=False):
        """ Constructs the computational graph for the Feature Space Suppression Network

        :param x: np.ndarray
        :param prime_network: bool
            whether to prime the suppression network after construction
        :return:
        """
        self.n_inputs = x.shape[1]
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X")

        # construct the suppression network
        self.layers, self.weights, self.biases = construct_network(self.X, self.n_neurons, self.names, self.activations)

        self.feature_space_prior_loss = self.prepare_sigmoid_cross_entropy_loss()
        self.suppression_network_l2_loss = self.get_suppression_network_l2_loss()
        if self.l2_penalty:
            logger.debug("Adding L2 loss to suppression network")
            self.feature_space_prior_loss += self.suppression_network_l2_loss

        self.feature_space_prior_training_op = self.prepare_training_op(self.feature_space_prior_loss)

        self.n_labeled, self.n_unlabeled, self.q_tau, self.Y, self.ensemble_scores, self.integrated_loss, self.aad_loss, \
        self.y_mod, self.afss_output, self.a_score_all, self.a_score = self.prepare_integrated_loss()
        self.aad_training_op = self.prepare_training_op(self.aad_loss)
        self.integrated_training_op = self.prepare_training_op(self.integrated_loss)

        self.init_session()

        if prime_network and self.prime:
            self.prime_suppression_network(x)
        elif prime_network and not self.prime:
            logger.debug("Skipping priming suppression network since self.prime is False")

    def init_session(self):
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        self.set_session(session)

    def set_session(self, session):
        self.session = session

    def close_session(self):
        if self.session is not None:
            self.session.close()

    def prime_suppression_network(self, x):
        """ Train the network to predict the bias_prob initially """

        # set the bias term so that output is self.bias_prob
        self.set_initial_bias_weight(self.bias_prob)

        # loss = self.session.run([self.feature_space_prior_loss], feed_dict={self.X: x})[0]
        # logger.debug("initial prior loss: %f" % loss)

        tm = Timer()
        self._train_feature_space_prior(x, self.max_init_epochs, batch_size=self.train_batch_size)
        logger.debug(tm.message("Completed prime_suppression_network() with max %d epochs:" % self.max_init_epochs))

    def output_params(self):
        """ Returns the top-most layer of the suppression network """
        return self.layers[len(self.layers)-1]

    def decision_function(self, x):
        """ Returns the probability outputs from the top layer of the network

        The probabilities correspond to the feature space relevance.
        """
        with tf.name_scope("decision_function"):
            z = self.output_params()
            logits = tf.sigmoid(z)
            out = self.session.run([logits], feed_dict={self.X: x})[0]
            # logger.debug("out: %s" % (str(out.shape)))
            # out = np.mean(out, axis=1).reshape((-1, 1))
        return out

    def prepare_sigmoid_cross_entropy_loss(self):
        """ Computes sigmoid cross entropy loss relative to the bias_prob

        similar to: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        p = self.bias_prob
        z = self.output_params()

        loss = p * -log(sigmoid(z)) + (1 - p) * -log(1 - sigmoid(z))
             = p * -log(1 / (1 + exp(-z))) + (1 - p) * -log(exp(-z) / (1 + exp(-z)))
             = (1 - p) * z + log(1 + exp(-z))
             = z - z * p + log(1 + exp(-z))
             = log(exp(z)) - x * p + log(1 + exp(-z))  # to avoid overflow in exp(-z) when z < 0
             = - z * p + log(1 + exp(z))
             = max(z, 0) - z * p + log(1 + exp(-abs(z)))

        :param x:
        :return:
        """
        with tf.name_scope("feature_space_prior_loss"):
            p = self.bias_prob
            z = self.output_params()
            # loss = tf.reduce_mean(- z * p + tf.log(1 + tf.exp(z)))
            loss = tf.reduce_mean(tf.maximum(z, 0) - z * p + tf.log(1 + tf.exp(-tf.abs(z))))
        return loss

    def prepare_output_sigmoid(self):
        with tf.name_scope("output_sigmoid"):
            s = tf.sigmoid(self.output_params())
        return s

    def prepare_integrated_loss(self):
        """ Prepare the integrated cross-entropy and AAD loss functions

        Assumes:
            1. the last instance in a batch is the tau-th ranked instance
            2. the number of neurons in the output layer of suppression network
               is same as the ensemble size
            3. length of y in a batch is equal to the number of labeled instances in it
        """
        m = self.n_neurons[len(self.n_neurons)-1]
        with tf.name_scope("integrated_loss"):
            # n = tf.shape(self.X)[0]-1  # since the last is the tau-th ranked instance

            # the following values will help us dynamically determine whether the
            # AAD loss applies to the current batch
            n_labeled = tf.placeholder(tf.int32, shape=(), name="n_labeled")
            n_unlabeled = tf.placeholder(tf.int32, shape=(), name="n_unlabeled")

            q_tau = tf.placeholder(tf.float32, shape=(), name="q_tau")
            y = tf.placeholder(tf.float32, shape=(None), name="y")
            ensemble_scores = tf.placeholder(tf.float32, shape=(None, m), name="ensemble_scores")

            # use tf.cond() because when n_labeled == 0, then there will be no values in 'y'
            y_ = tf.cond(tf.greater(n_labeled, 0), lambda: 2*y - 1, lambda: tf.constant([0], dtype=tf.float32))  # {0, 1} -> {-1, 1}

            afss_output = tf.sigmoid(self.output_params())
            a_score_all = tf.reduce_sum(ensemble_scores * afss_output, axis=1)
            a_score_labeled = a_score_all[0:n_labeled]

            # the AAD loss applies only if there are labeled data.
            # we handle this dynamically for a batch using tf.cond().
            aad_loss = tf.cond(tf.equal(n_labeled, 0),
                               lambda: 0.0,  # no labeled data
                               lambda: self.c_q_tau * tf.reduce_mean(tf.maximum(0.0, y_ * (q_tau - a_score_labeled))) + \
                                       self.c_x_tau * tf.reduce_mean(tf.maximum(0.0, y_ * (a_score_all[n_labeled+n_unlabeled] - a_score_labeled)))
                               )

            if self.l2_penalty:
                aad_loss += self.suppression_network_l2_loss

            if self.lambda_prior > 0:
                integrated_loss = aad_loss + self.lambda_prior * self.feature_space_prior_loss
            else:
                integrated_loss = aad_loss

        return n_labeled, n_unlabeled, q_tau, y, ensemble_scores, integrated_loss, aad_loss, y_, afss_output, a_score_all, a_score_labeled

    def prepare_training_op(self, loss, learning_rate=0.01):
        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
        return training_op

    def _train_feature_space_prior(self, x, max_epochs=1, batch_size=25):
        """ Trains only the softmax entropy loss *without* accounting for the AAD label loss """
        i = 0
        loss_window = 20
        prev_avg = 0.0
        losses = np.zeros(max_epochs, dtype=np.float32)
        for epoch in range(max_epochs):
            for batch in get_unlabeled_batches(x, batch_size=batch_size):
                if False and batch.shape[0] < batch_size:
                    logger.debug("smaller batch received of size: %d" % batch.shape[0])  # DEBUG only
                self.session.run([self.feature_space_prior_training_op], feed_dict={self.X: batch})
            i += 1
            # out = self.eval_output(x)
            # logger.debug("[%d] train init:\n%s" % (i, str(out)))
            loss = self.session.run([self.feature_space_prior_loss], feed_dict={self.X: x})[0]
            losses[i-1] = loss
            # logger.debug("[%d/%d] prior loss: %f" % (i, self.max_init_epochs, loss))
            if i >= loss_window:
                curr_avg = np.mean(losses[(i-loss_window):i])
                if np.abs(curr_avg - prev_avg) < self.init_tol:
                    logger.debug("terminating training at %d epoch\n(prev/curr): %f / %f" %
                                 (i, prev_avg, curr_avg))
                    break
                prev_avg = curr_avg
            # logger.debug("Trained initial weights for %d/%d epochs" % (i, self.max_init_epochs))
        return losses[-1]

    def _train_integrated_loss(self, x_labeled, y_labeled, scores_labeled,
                               x_unlabeled, scores_unlabeled,
                               x_tau, scores_tau, q_tau, max_epochs=1, batch_size=25):
        """ Trains the complete AFSS loss (softmax entropy loss + the AAD label loss) """
        loss_window = 20
        prev_avg = 0.0
        losses = np.zeros(max_epochs, dtype=np.float32)

        n_labeled = x_labeled.shape[0]
        n_unlabeled = x_unlabeled.shape[0]
        max_labeled_reps = 1
        if n_labeled < 50 and n_labeled * 10 < n_unlabeled:
            max_labeled_reps = self.max_labeled_reps

        i = 0
        for epoch in range(max_epochs):
            i += 1
            for batch_x, batch_y, batch_scores, b_n_lbl, b_n_unl in get_afss_batches(x_labeled,
                                                                                     y_labeled,
                                                                                     scores_labeled,
                                                                                     x_unlabeled,
                                                                                     scores_unlabeled,
                                                                                     batch_size=batch_size,
                                                                                     n_labeled_reps=max_labeled_reps,
                                                                                     lambda_prior=self.lambda_prior):
                if False and batch_x.shape[0] < batch_size:
                    logger.debug("smaller batch received of size: %d" % batch_x.shape[0])  # DEBUG only
                batch_x_ = np.vstack([batch_x, x_tau])
                batch_scores_ = np.vstack([batch_scores, scores_tau])
                self.session.run([self.integrated_training_op],
                                 feed_dict={self.n_labeled: b_n_lbl,
                                            self.n_unlabeled: b_n_unl,
                                            self.X: batch_x_, self.Y: batch_y,
                                            self.q_tau: q_tau, self.ensemble_scores: batch_scores_})

            x_ = np.vstack([x_labeled, x_tau])
            scores_ = np.vstack([scores_labeled, scores_tau])
            loss = self.session.run([self.integrated_loss],
                                    feed_dict={self.n_labeled: x_labeled.shape[0],
                                               self.n_unlabeled: 0,
                                               self.X: x_, self.Y: y_labeled, self.q_tau: q_tau,
                                               self.ensemble_scores: scores_})[0]
            losses[i-1] = loss

            if i >= loss_window:
                curr_avg = np.mean(losses[(i-loss_window):i])
                if np.abs(curr_avg - prev_avg) < self.init_tol:
                    logger.debug("terminating training at %d epoch\n(prev/curr): %f / %f" %
                                 (i, prev_avg, curr_avg))
                    break
                prev_avg = curr_avg
            # logger.debug("Trained initial weights for %d/%d epochs" % (i, self.max_init_epochs))
        return losses[i-1]

    def get_weighted_scores(self, x, ensemble_scores):
        """ Multiplies the scores with feature space relevance

        Note: Scores must be such that higher is more anomalous
        """
        afss_scores = self.decision_function(x)
        scores = np.multiply(ensemble_scores, afss_scores)
        scores = np.sum(scores, axis=1)

        return scores

    def get_aad_loss(self, x, y, q_tau, ensemble_scores):
        """ Returns the aad component of the loss

        Note: Use only for debug.

        loss_prev = self.get_aad_loss(x, y_labeled, q_tau, scores)
        where:
            1. x.shape[0] == (len(y) + 1) because the last instance in x is the
               tau-th ranked element which we consider unlabeled.
            2. x.shape[0] == ensemble_scores.shape[0]

        """
        return self.session.run([self.aad_loss],
                                feed_dict={self.n_labeled: x.shape[0]-1,
                                           self.n_unlabeled: 0,
                                           self.X: x, self.Y: y, self.q_tau: q_tau,
                                           self.ensemble_scores: ensemble_scores})[0]

    def get_y_mod(self, x, y, q_tau, ensemble_scores):
        """ Returns the values of y in {-1, 1} instead of {0, 1}

        Note: Use only for debug.

        y_mod = self.get_y_mod(x_labeled, y_labeled, q_tau, scores_labeled)
        logger.debug("y_mod: %s" % (str(list(y_mod))))
        where:
            1. x.shape[0] == (len(y) + 1) because the last instance in x is the
               tau-th ranked element which we consider unlabeled.
            2. x.shape[0] == ensemble_scores.shape[0]
        """
        return self.session.run([self.y_mod],
                                feed_dict={self.n_labeled: x.shape[0]-1,
                                           self.n_unlabeled: 0,
                                           self.X: x, self.Y: y, self.q_tau: q_tau,
                                           self.ensemble_scores: ensemble_scores})[0]

    def update_afss(self, x, y, hf, ensemble_scores, tau=0.03):
        """ Update network by incorporating labels

        Note: Scores must be such that higher is more anomalous

        :param x: np.ndarray
        :param y: np.array
        :param hf: np.array
        :param ensemble_scores: np.ndarray
        :param tau: float
        :return: None
        """
        tm = Timer()
        n = x.shape[0]
        n_tau = int(tau * n)
        # logger.debug("tau: %f, n_tau: %d" % (tau, n_tau))

        x_unlabeled, y_unlabeled, scores_unlabeled, x_labeled, y_labeled, scores_labeled = \
            partition_instances(x, y, ensemble_scores, hf)

        df_scores = self.get_weighted_scores(x, ensemble_scores)
        sorted_indexes = np.argsort(-df_scores)
        i_tau = sorted_indexes[n_tau]
        x_tau = x[i_tau]
        scores_tau = ensemble_scores[i_tau]
        q_tau = df_scores[i_tau]

        # logger.debug("i_tau: %d, q_tau: %f" % (i_tau, q_tau))

        loss_integrated = self._train_integrated_loss(x_labeled, y_labeled, scores_labeled,
                                                      x_unlabeled, scores_unlabeled,
                                                      x_tau, scores_tau, q_tau,
                                                      max_epochs=self.max_afss_epochs,
                                                      batch_size=self.train_batch_size)
        # logger.debug("integrated loss: %f" % loss_integrated)

        # logger.debug(tm.message("Completed update_aafs() with max %d epochs:" % self.max_afss_epochs))

    def log_probability_ranges(self, x):
        """ Useful for debug only

        Logs the max/min probability values observed among x
        """
        out = self.decision_function(x)
        out_min = np.min(out, axis=0)
        out_max = np.max(out, axis=0)
        out_mean = np.mean(out, axis=0)
        out_sd = np.std(out, axis=0)
        logger.debug("output len: %s\nmin/max:\n%s / %s\nmean probs:\n%s\nstdev:\n%s" %
                     (out.shape[0], str(list(out_min)), str(list(out_max)), str(list(out_mean)), str(list(out_sd))))


def get_glad_option_list():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy2", required=False,
                        help="Dataset name")
    parser.add_argument("--datafile", type=str, default="", required=False,
                        help="Original data in CSV format.")
    parser.add_argument("--results_dir", action="store", default="",
                        help="Folder where the generated metrics will be stored")
    parser.add_argument("--ensemble_type", type=str, default="loda", required=False,
                        help="Type of ensemble to create")
    parser.add_argument("--op", type=str, default="", required=False,
                        help="Operation to apply")
    parser.add_argument("--reruns", action="store", type=int, default=1,
                        help="Number of times the experiment should be rerun with randomization")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--budget", type=int, default=0, required=False,
                        help="Budget for feedback")
    parser.add_argument("--n_epochs", type=int, default=200, required=False,
                        help="Max training epochs")
    parser.add_argument("--train_batch_size", type=int, default=25, required=False,
                        help="Batch size for stochastic gradient descent based training methods")
    parser.add_argument("--n_anoms", type=int, default=10, required=False,
                        help="Number of top anomalies to report")
    parser.add_argument("--loda_mink", type=int, default=2, required=False,
                        help="Minimum number of random LODA projections")
    parser.add_argument("--loda_maxk", type=int, default=15, required=False,
                        help="Maximum number of random LODA projections")
    parser.add_argument("--loda_debug", action="store_true", default=False,
                        help="If specified, this will signal the use of pre-set projection vectors. "
                             "This is relevant only if the dataset is 2D.")
    parser.add_argument("--afss_tau", type=float, default=0.03, required=False,
                        help="Tau (a guess for the fraction of anomalies, just needs to be small)")
    parser.add_argument("--afss_nodes", type=int, default=0, required=False,
                        help="Number of nodes in first layer of AFSS")
    parser.add_argument("--afss_max_labeled_reps", type=int, default=5, required=False,
                        help="Number of times the labeled instances may be repeated when performing SGD for AFSS")
    parser.add_argument("--afss_c_tau", type=float, default=1.0, required=False,
                        help="Penalty factor for aad loss with AFSS")
    parser.add_argument("--afss_lambda_prior", type=float, default=1.0, required=False,
                        help="Penalty factor for feature space prior with AFSS")
    parser.add_argument("--afss_bias_prob", type=float, default=0.50, required=False,
                        help="Bias probability for AFSS")
    parser.add_argument("--afss_no_prime", action="store_true", default=False,
                        help="The suppression network for AFSS will NOT be primed if this option is set")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Whether to plot figures")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--ensemble_only", action="store_true", default=False,
                        help="Whether to run AFSS on the generated ensemble")
    parser.add_argument("--compare_aad", action="store_true", default=False,
                        help="Whether to compare AFSS against AAD")
    return parser


class GladOpts(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.datafile = args.datafile
        self.results_dir = args.results_dir
        self.ensemble_type = args.ensemble_type
        self.op = args.op
        self.reruns = args.reruns
        self.randseed = args.randseed
        self.budget = args.budget
        self.n_epochs = args.n_epochs
        self.train_batch_size = args.train_batch_size
        self.n_anoms = args.n_anoms
        self.loda_mink = args.loda_mink
        self.loda_maxk = args.loda_maxk
        self.loda_debug = args.loda_debug
        self.afss_tau = args.afss_tau
        self.afss_nodes = args.afss_nodes
        self.afss_max_labeled_reps = args.afss_max_labeled_reps
        self.afss_c_tau = args.afss_c_tau
        self.afss_lambda_prior = args.afss_lambda_prior
        self.afss_bias_prob = args.afss_bias_prob
        self.afss_no_prime = args.afss_no_prime
        self.debug = args.debug
        self.plot = args.plot
        self.log_file = args.log_file
        self.ensemble_only = args.ensemble_only
        self.compare_aad = args.compare_aad

        self.fid = 1  # this attributed has been retained for historical reasons only
        self.runidx = 0  # number of reruns

    def get_alad_metrics_name_prefix(self):
        tau_str = ("-tau%0.2f" % self.afss_tau).replace('.', '_')
        budget_str = "-bd%d" % self.budget if self.budget > 0 else ""
        if self.ensemble_type == "loda":
            ensemble_str = "-loda_%d_%d" % (self.loda_mink, self.loda_maxk)
        else:
            ensemble_str = "-%s" % self.ensemble_type
        bias_str = ("-bias%0.2f" % self.afss_bias_prob).replace('.', '_')
        c_tau_str = ("-c%0.2f" % self.afss_c_tau).replace('.', '_')
        nodes_str = "-nodes%d" % self.afss_nodes
        reps_sig = "-amr%d" % self.afss_max_labeled_reps
        prior_sig = "-no_prior" if self.afss_lambda_prior == 0 else ""
        name = "%s%s%s%s%s%s%s%s%s-r%d" % (self.dataset, ensemble_str, nodes_str, budget_str,
                                         tau_str, bias_str, c_tau_str, reps_sig, prior_sig, self.reruns)
        return name

    def str_opts(self):
        name = self.get_alad_metrics_name_prefix()
        s = "%s" % name
        return s


def get_glad_command_args(debug=False, debug_args=None):
    parser = get_glad_option_list()

    if debug:
        unparsed_args = debug_args
    else:
        unparsed_args = sys.argv
        if len(unparsed_args) > 0:
            unparsed_args = unparsed_args[1:len(unparsed_args)]  # script name is first arg

    args = parser.parse_args(unparsed_args)
    return args


