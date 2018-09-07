import random
import numpy as np
import numpy.random as rnd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold
import collections
import math
import tensorflow as tf
from common.utils import *
from common.timeseries_datasets import *
from common.data_plotter import *
from common.nn_utils import *


"""
This is a simplified version of TensorFlow word2vec_basic.py

The primary purpose is pedagogical. Instead of calling some tensorflow functions
such as tf.nn.nce_loss, we directly sample uniformly at random for negative samples.

The other reason to use this is for the activity modeling example in activity_word2vec.py
where the 'vocabulary' is limited to the total number of sensors (few) such that a customized
implementation might be more efficient. 
"""


class CustomWord2vec(object):

    def __init__(self,
                 sensors=None, sensor2code=None, code2sensor=None,
                 dims=100, window_size=3, neg_samples=3, n_epochs=1,
                 learning_rate=0.001, debug=False):
        self.sensors = sensors
        self.sensor2code = sensor2code
        self.code2sensor = code2sensor

        self.dims = dims
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.debug = debug

        self.X = self.Y = self.Z = self.W = self.embedding = self.weights = None
        self.normalized_embeddings = None
        self.similarity = None
        self.training_op = None

    def fit(self, seq):

        tf.set_random_seed(42)

        self.X = tf.placeholder(tf.int32, shape=[None])  # input 'word'
        self.Y = tf.placeholder(tf.int32, shape=[None])  # predicted 'word'
        self.Z = tf.placeholder(tf.float32, shape=[None])  # multiplier {1, -1}
        self.W = tf.placeholder(tf.float32, shape=[None])  # weight [0, 1.0]

        vocab_size = len(self.sensors)

        valid_examples = np.arange(0, vocab_size)
        valid_size = len(valid_examples)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        top_k = 4  # number of nearest neighbors for validation of similarity

        init = tf.random_uniform((vocab_size, self.dims),
                                 minval=-1.0, maxval=1.0, dtype=tf.float32)

        # the encoding matrix
        self.embedding = tf.Variable(init, name="embedding")

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True))
        self.normalized_embeddings = self.embedding / norm
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                       valid_dataset)
        self.similarity = tf.matmul(
            self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

        w_i = tf.nn.embedding_lookup(self.embedding, self.X)

        # the 'output' matrix, or the coefficients of logistic regression
        # for each class (words). This will be ignored once the embeddings
        # have been computed
        self.weights = tf.Variable(init, name="weights")  # weights
        self.b = tf.Variable(tf.zeros(vocab_size), name="b", dtype=tf.float32)  # biases

        w_o = tf.nn.embedding_lookup(self.weights, self.Y)
        w_b = tf.nn.embedding_lookup(self.b, self.Y)

        with tf.name_scope("loss"):
            """
            Refer to Equation 4 in:
                Distributed Representations of Words and Phrases and their Compositionality, 
                by Mikolov et. al., 2014

            loss = log(sigmoid(W_i.W_pos)) + E[log(sigmoid(-W_i.W_neg))]

            Note: The second term above (E[.]) is an 'expectation'.
            To compute the expectation, we multiply by the self.W.
            To distinguish between pos/neg examples, we multiply by self.Z
            """
            sim = tf.reduce_sum(tf.multiply(w_i, w_o), axis=1) + w_b
            sim_sigmoids = tf.log(tf.nn.sigmoid(tf.multiply(sim, self.Z)))
            log_lik_loss = -tf.reduce_mean(tf.multiply(sim_sigmoids, self.W))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training_op = optimizer.minimize(log_lik_loss)

        init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(init)
        timer = Timer()
        i = 0
        for epoch in range(self.n_epochs):
            for x, y, z, w in self.get_batches_skip_gram(seq, window_size=self.window_size,
                                                         neg_samples=self.neg_samples):
                # logger.debug(np.hstack([y, x, z, w]))
                sim_v, log_lik_loss_v, _ = self.session.run([sim, log_lik_loss, self.training_op],
                                                            feed_dict={self.X: x, self.Y: y, self.Z: z, self.W: w})
                if self.debug and (i == 0 or (i + 1) % 5000 == 0):
                    # the original word2vec code for logging the most similar
                    # words for a particular word
                    logger.debug("i: %d, log_lik_loss_v: %f" % (i, log_lik_loss_v))
                    sim_valid = self.session.run(self.similarity)
                    for j in range(valid_size):
                        valid_word = self.code2sensor[valid_examples[j]]
                        nearest = (-sim_valid[j, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = self.code2sensor[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        logger.debug(log_str)
                if (i + 1) % 5000 == 0:
                    # logger.debug("sim_v: %s\n%s" % (str(sim_v.shape), str(sim_v)))
                    logger.debug("processed %d" % (i + 1))
                    # break  # early terminate for DEBUG only
                i += 1
            logger.debug(timer.message("Completed epoch %d in" % epoch))

    def get_embeddings(self, normalized=True):
        return self.session.run(self.normalized_embeddings) if normalized \
            else self.session.run(self.embedding)

    def get_batches_skip_gram(self, seq, window_size=3, skip_size=1, n_contexts=10, neg_samples=3):
        """ Skip-gram model for word2vec

        The max #samples per batch will be:
            n_contexts x ((window_size - 1) + neg_samples)

        :param window_size: int
            length of each context window. Must be > 1 and must be an odd number.
        :param skip_size: int
        :param n_contexts: int
            Number of context windows per batch.
        :param neg_samples: int
            Number of negative samples per window
        :return:
        """
        if window_size <= 1 or window_size % 2 == 0:
            raise ValueError("window_size must be greater than 1 and must be odd")

        n = len(seq)
        s = window_size // 2
        all_sensors = set(self.code2sensor.keys())
        st = 0
        sz = (window_size - 1) + neg_samples  # number of samples per context window
        batch_size = n_contexts * sz
        x = y = z = w = None
        for i in range(s, n - s, skip_size):

            if i + skip_size >= n:
                logger.debug("i: %d, n: %d, s: %d, sz: %d" % (i, n, s, sz))

            if st == 0:
                x = np.zeros(batch_size, dtype=np.int32)
                y = np.zeros(batch_size, dtype=np.int32)
                z = np.zeros(batch_size, dtype=np.float32)
                w = np.zeros(batch_size, dtype=np.float32)

            w_in = seq[i]

            # w_in will be same for both positive and negative samples
            x[st:(st + sz)] = w_in
            z[st:(st + 2 * s)] = 1
            z[(st + 2 * s):(st + sz)] = -1
            w[st:(st + 2 * s)] = 1  # weights for positive samples
            w[(st + 2 * s):(st + sz)] = 1. / neg_samples  # weights for negative samples

            # first, populate the positive examples
            y[st:(st + s)] = seq[(i - s):i]
            y[(st + s):(st + 2 * s)] = seq[(i + 1):(i + s + 1)]

            # Now, sample a few negative examples...
            # sample a few sensor ids uniformly at random from those
            # which do not occur in the current context
            curr = set(seq[(i - s):(i + s)])  # sensors in current context window
            non_context = list(all_sensors - curr)  # sensors *not* in current context window
            np.random.shuffle(non_context)  # random subsample
            y[(st + 2 * s):(st + sz)] = non_context[0:neg_samples]

            st += sz
            if st >= batch_size:
                yield x, y, z, w
                st = 0


