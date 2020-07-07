import collections
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from ..common.utils import logger


"""
A copy of Tensorflow's word2vec_basic.py. The functionality is encapsulated in a class.

Original TenorFlow code:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""


class Word2vec(object):
    def __init__(self, sensors=None, sensor2code=None, code2sensor=None,
                 dims=100, window_size=3, n_epochs=1,
                 batch_size=128, num_skips=2, num_sampled=64, num_steps=100000,
                 learning_rate=0.001, debug=False):
        self.sensors = sensors
        self.dictionary = sensor2code
        self.reverse_dictionary = code2sensor

        self.window_size = window_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.debug = debug

        if window_size <= 1 or window_size % 2 == 0:
            raise ValueError("window_size must be greater than 1 and must be odd")

        self.vocabulary_size = len(self.sensors)
        self.batch_size = batch_size
        self.embedding_size = dims  # Dimension of the embedding vector.
        self.skip_window = window_size // 2  # How many words to consider left and right.
        self.num_skips = num_skips  # How many times to reuse an input to generate a label.
        self.num_sampled = num_sampled  # Number of negative examples to sample.
        self.num_steps = num_steps

        self.vocab_size = len(self.sensors)

        self.graph = None
        self.session = None

        self.final_embeddings = None
        self.final_normalized_embeddings = None

    # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self, data, data_index, batch_size, num_skips, skip_window):
        # global data_index
        # assert batch_size % num_skips == 0
        # assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels, data_index

    def fit(self, data):

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                valid_examples = np.arange(0, self.vocab_size)
                valid_size = len(valid_examples)
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
                top_k = 4  # number of nearest neighbors for validation of similarity

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [self.vocabulary_size, self.embedding_size],
                            stddev=1.0 / math.sqrt(self.embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm

            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

        # Step 5: Begin training.
        # num_steps = 100001

        self.session = tf.Session(graph=self.graph)

        with self.session as session:
            # We must initialize all variables before we use them.
            init.run()
            logger.debug('Initialized')
            data_index = 0
            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels, data_index = self.generate_batch(data, data_index,
                                                                             self.batch_size, self.num_skips,
                                                                             self.skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = None  # tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    logger.debug('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if self.debug and step % 5000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = self.reverse_dictionary[valid_examples[i]]
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        logger.debug(log_str)
            self.final_embeddings = embeddings.eval()
            self.final_normalized_embeddings = normalized_embeddings.eval()

    def get_embeddings(self, normalized=True):
        return self.final_normalized_embeddings if normalized else self.final_embeddings


