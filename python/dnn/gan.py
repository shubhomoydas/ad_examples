import numpy as np
import tensorflow as tf
import numpy.random as rnd
from sklearn import mixture
from common.gen_samples import *
from common.nn_utils import get_train_batches
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
A simple [conditional|Info] GAN with fully connected layers for both generator and discriminator.
Also supports AnoGAN.

See dnn/test_gan.py for usage.

References:
[1] Generative Adversarial Nets by Ian J. Goodfellow, Jean Pouget-Abadi, et al., NIPS 2014
[2] Conditional Generative Adversarial Nets by Mehdi Mirza and Simon Osindero, 2014
[3] Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery
    by Thomas Schlegl, Philipp Seebock, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs, IPMI 2017
[4] InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
    by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
"""

TINY = 1e-8  # as in virtually every InfoGAN implementation on the internet


def set_random_seeds(py_seed=42, np_seed=42, tf_seed=42):
    random.seed(py_seed)
    rnd.seed(np_seed)
    tf.set_random_seed(tf_seed)


class Listener(object):
    def __init__(self):
        pass

    def __call__(self, gan, epoch, epoch_start_tm):
        pass


def fit_gmm(x, val_x, min_k=1, max_k=10):
    cv_type = 'diag'  # ['spherical', 'tied', 'diag', 'full']
    lowest_bic = np.infty
    bic = []
    best_gmm = None
    for k in range(min_k, max_k+1):
        gmm = mixture.GaussianMixture(n_components=k, covariance_type=cv_type)
        gmm.fit(x)
        bic.append(gmm.bic(val_x))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    return best_gmm, lowest_bic, bic


def get_cluster_labels(x, min_k=1, max_k=10):
    """ Fits data to a Gaussian Mixture Model and assigns clusters """
    gmm, _, _ = fit_gmm(x, x, min_k=min_k, max_k=max_k)
    logger.debug("best GMM k: %d" % (gmm.n_components))
    y = gmm.predict(x)
    # logger.debug("y:\n%s" % (str(y)))
    return y, gmm


def get_nn_layer(layers, layer_from_top=1):
    return layers[len(layers) - layer_from_top]


class GAN(object):
    """ A GAN or a conditional GAN for simple i.i.d data """
    def __init__(self, data_dim=1, discr_layer_nodes=None, discr_layer_activations=None,
                 gen_input_dim=None, gen_layer_nodes=None, gen_layer_activations=None,
                 label_smoothing=False, smoothing_prob=0.9, info_gan=False, info_gan_lambda=1.0,
                 conditional=False, n_classes=0, pvals=None, enable_ano_gan=False,
                 n_epochs=10, batch_size=25, shuffle=False, learning_rate=0.005,
                 l2_lambda=0.001, listener=None, use_adam=False):
        """ Create the generator-discriminator networks

        :param data_dim: int
            number of input dimensions in original data
        :param discr_layer_nodes: list of int
            number of nodes in each discriminator layer (excluding input)
        :param discr_layer_activations: list
            list of activation functions for each discriminator layer (excluding input)
        :param gen_input_dim: int
            number of input dimensions in input generator samples
        :param gen_layer_nodes: list of int
            number of nodes in each generator layer (excluding input)
        :param gen_layer_activations: list
            list of activation functions for each generator layer (excluding input)
        :param label_smoothing: bool
            if True, then use one-sided label smoothing for discriminator loss
        :param smoothing_prob: float
            label-smoothing probability
        :param info_gan: bool
            if True, then use InfoGAN, else simple or conditional GAN
        :param info_gan_lambda: float
            InfoGAN regularization penalty
        :param conditional: bool
            if True, then use Conditional GAN, else simple or InfoGAN
        :param n_classes:
            number of class labels in conditional mode
        :param pvals: np.array(dtype=np.float32)
            probability of each class
        :param enable_ano_gan: bool
            whether to enable AnoGAN network (for anomaly detection)
        :param n_epochs: int
            max number of epochs for training
        :param batch_size: int
            mini-batch size for training
        :param shuffle: bool
            whether to shuffle the data in each epoch during training
        :param learning_rate: float
        :param l2_lambda: float
        :param listener: Listener
            call-back function that gets called at the end of each training epoch
        :param use_adam: bool
            whether to use ADAM. The default is GradientDescent
        """
        self.label_smoothing = label_smoothing
        self.smoothing_prob = smoothing_prob
        self.info_gan = info_gan
        self.info_gan_lambda = info_gan_lambda
        self.conditional = conditional
        self.n_classes = n_classes
        self.pvals = pvals
        self.enable_ano_gan = enable_ano_gan

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dim = data_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.listener = listener
        self.use_adam = use_adam

        # first create the generator network
        self.gen_input_dim = gen_input_dim
        self.gen_layer_nodes = gen_layer_nodes
        self.gen_layer_activations = gen_layer_activations
        self.z = self.gen = None

        # now, create the discriminator network
        self.discr_layer_nodes = discr_layer_nodes
        self.discr_layer_activations = discr_layer_activations

        self.x = self.y = None
        self.discr_data = self.discr_gen = None
        self.discr_loss = self.gen_loss = self.discr_training_op = self.gen_training_op = None

        # InfoGAN variables and losses
        self.q_network = self.q_pred = None
        self.info_gan_loss = None

        # AnoGAN variables and losses
        self.ano_gan_lambda = None
        self.ano_z = self.ano_gan_net_G = self.ano_gan_net_D = None
        self.ano_gan_training_op = self.ano_gan_loss = None
        self.ano_gan_loss_R = self.ano_gan_loss_D = self.ano_gan_info_loss = None
        self.ano_gan_q_network = None

        # Tensoflow session object
        self.session = None

        self.unif_lo = 0.0  # -1.0
        self.unif_hi = 1.0

        if self.conditional and self.info_gan:
            raise ValueError("Only one of conditional or info_gan should be true")

        if (self.conditional or self.info_gan) and self.pvals is None:
            raise ValueError("pvals is required for ConditionalGAN and InfoGAN")

        self.init_network()

    def init_network(self):

        self.x = tf.placeholder(tf.float32, shape=(None, self.data_dim), name="x")
        self.z = tf.placeholder(tf.float32, shape=(None, self.gen_input_dim), name="z")

        if self.conditional:
            if self.n_classes <= 0:
                raise ValueError("n_classes must be greater than 1 for conditional GAN")
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_classes), name="y")

        with tf.variable_scope("GAN"):
            # here will create the generator and discriminator networks with initial reuse=False
            self.gen = self.generator(z=self.z, y=self.y, reuse_gen=False)
            self.discr_data, self.discr_gen = self.discriminator(x=self.x, y=self.y, reuse_discr=False)

        if not self.label_smoothing:
            discr_loss_data = -tf.log(tf.nn.sigmoid(get_nn_layer(self.discr_data, layer_from_top=1)))
        else:
            logger.debug("Label smoothing enabled with smoothing probability: %f" % self.smoothing_prob)
            discr_logit = get_nn_layer(self.discr_data, layer_from_top=1)
            discr_loss_data = tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_logit,
                                                                      labels=tf.ones(shape=tf.shape(discr_logit)) * self.smoothing_prob)

        discr_gen_logit = get_nn_layer(self.discr_gen, layer_from_top=1)
        discr_gen_probs = tf.nn.sigmoid(discr_gen_logit)
        self.discr_loss = tf.reduce_mean(discr_loss_data - tf.log(1 - discr_gen_probs))
        self.gen_loss = tf.reduce_mean(-tf.log(discr_gen_probs))

        self.info_gan_loss = tf.constant(0.0)
        if self.info_gan:
            logger.debug("Adding InfoGAN regularization")
            with tf.variable_scope("InfoGAN"):
                # The last-but-one layer of the discriminator (when the input is from
                # fake generated data) will be the input to category prediction layer.
                # The expectation is w.r.t generator output.
                self.q_network = self.init_info_gan_network(get_nn_layer(self.discr_gen, layer_from_top=2),
                                                            reuse=False)

                # the below will be used to predict category for debug; it is not required for training
                self.q_pred = self.init_info_gan_network(get_nn_layer(self.discr_data, layer_from_top=2),
                                                         reuse=True)

            # get softmax output layer of q_network that predicts class
            q_out = get_nn_layer(self.q_network, layer_from_top=1)
            # compute entropy of class predictions
            self.info_gan_loss = self.marginal_mutual_info(q_out, self.pvals)

        vars = tf.trainable_variables()
        for v in vars: logger.debug(v.name)
        g_params = [v for v in vars if v.name.startswith('GAN/G/')]
        d_params = [v for v in vars if v.name.startswith('GAN/D/')]
        q_params = [v for v in vars if v.name.startswith('InfoGAN/')]
        if self.info_gan and len(q_params) == 0:
            # Just to be sure we do not have programmatic errors
            raise ValueError("No q_params found for InfoGAN")

        if self.l2_lambda > 0:
            # add L2 regularization loss
            logger.debug("Adding L2 regularization")
            l2_loss_g, l2_loss_d, l2_loss_q = self.get_l2_regularizers(g_params, d_params, q_params)
            self.gen_loss += self.l2_lambda * l2_loss_g
            self.discr_loss += self.l2_lambda * l2_loss_d

            if self.info_gan:
                self.info_gan_loss += self.l2_lambda * l2_loss_q
                g_params.extend(q_params)
                d_params.extend(q_params)

        self.gen_training_op = self.training_op(self.gen_loss + self.info_gan_lambda * self.info_gan_loss,
                                                var_list=g_params, use_adam=self.use_adam)
        self.discr_training_op = self.training_op(self.discr_loss + self.info_gan_lambda * self.info_gan_loss,
                                                  var_list=d_params, use_adam=self.use_adam)

        if self.enable_ano_gan:
            # Prepare variables required for AnoGAN functionality
            #
            # Note: AnoGAN functionality will come in use only *after* the
            # GAN (simple|conditional|InfoGAN) has been fully trained.
            self.ano_gan_lambda = tf.placeholder(tf.float32, shape=(), name="ano_gan_lambda")
            self.ano_z = tf.Variable(initial_value=tf.zeros([1, self.gen_input_dim]), trainable=True, name="ano_z")
            with tf.variable_scope("GAN", reuse=True):
                self.ano_gan_net_G, self.ano_gan_net_D = self.init_ano_gan_network(x=self.x, y=self.y, z=self.ano_z)

            ano_gan_G, ano_gan_D, ano_gan_D_features = self.ano_gan_outputs()

            # reconstruction loss: generate synthetic data in original
            # feature space that is close to input data
            self.ano_gan_loss_R = tf.reduce_sum(tf.abs(tf.subtract(self.x, ano_gan_G)))
            # ano_gan_loss_R = tf.nn.l2_loss(tf.subtract(self.x, ano_gan_G))

            # discrimination loss: encourage generated data to be
            # similar to real data
            self.ano_gan_loss_D = tf.reduce_sum(-tf.log(tf.nn.sigmoid(ano_gan_D)))

            self.ano_gan_info_loss = tf.constant(0.0)
            if self.info_gan:
                # apply appropriate variable scope for reuse
                with tf.variable_scope("InfoGAN"):
                    # The last-but-one layer of the discriminator will be the input to
                    # category prediction layer. The expectation is w.r.t generator output.
                    self.ano_gan_q_network = self.init_info_gan_network(ano_gan_D_features, reuse=True)

                    # Compute the InfoGAN entropy regularization loss for
                    # AnoGAN with the output of ano_gan_q_network
                    self.ano_gan_info_loss = self.marginal_mutual_info(get_nn_layer(self.ano_gan_q_network,
                                                                                    layer_from_top=1),
                                                                       self.pvals)

            self.ano_gan_loss = (1 - self.ano_gan_lambda) * self.ano_gan_loss_R + \
                                self.ano_gan_lambda * (self.ano_gan_loss_D + self.ano_gan_info_loss)

            self.ano_gan_training_op = self.training_op(self.ano_gan_loss, var_list=[self.ano_z], use_adam=self.use_adam)

    def marginal_mutual_info(self, q_c_x, c, include_h_c=False):
        """ Compute avg. entropy of probability distributions arcoss all rows of q_c_x

        Each row of q_c_x contains one probability distribution (likely computed with softmax)
        """
        mi = -tf.reduce_mean(tf.reduce_sum(tf.multiply(c, tf.log(q_c_x + TINY)), axis=1))
        if include_h_c:
            # usually this is constant; hence add this only if asked for
            mi += -tf.reduce_mean(tf.reduce_sum(c * tf.log(c + TINY), axis=1))
        return mi

    def get_l2_regularizers(self, g_params, d_params, q_params=None):
        """ Returns L2 regularizers for generator and discriminator variables

        :param g_params: list of tf.Variable
            The generator parameters
        :param d_params: list of tf.Variable
            The discriminator parameters
        :param q_params: list of tf.Variable
            The InfoGAN regularization parameters
        :return: generator, discriminator, InfoGAN L2 regularizer losses
        """
        l2_loss_g = 0.0
        l2_loss_d = 0.0
        l2_loss_q = 0.0
        for v in g_params:
            l2_loss_g += tf.nn.l2_loss(v)
        for v in d_params:
            l2_loss_d += tf.nn.l2_loss(v)
        if q_params is not None:
            for v in q_params:
                l2_loss_q += tf.nn.l2_loss(v)
        return l2_loss_g, l2_loss_d, l2_loss_q

    def generator(self, z, y=None, reuse_gen=False):
        inp = z
        if y is not None:
            inp = tf.concat(values=[z, y], axis=1)
        with tf.variable_scope('G'):
            gen_layer_names = ["g_%d" % (i+1) for i in range(len(self.gen_layer_nodes))]
            gen = self.gan_construct(inp, self.gen_layer_nodes, names=gen_layer_names,
                                     activations=self.gen_layer_activations, reuse=reuse_gen)
        return gen

    def discriminator(self, x, y=None, reuse_discr=False, prep_gen_input=True):
        """ Prepares the discriminator network

        Note: Assumes that the generator network has already been created so that it
            can be reused. The discriminator network is reused if reuse_discr=True.

        :param x: np.ndarray
        :param y: np.ndarray
            TensorFlow Variable that expects one-hot encoded labels
        :param reuse_discr: bool
            Whether to reuse previously declared discriminator variables in the scope
        :param prep_gen_input: bool
            Whether to return the network that takes generator output as input to discriminator
        :return: tf.Variable, tf.Variable
        """
        with tf.variable_scope('D'):
            discr_layer_names = ["d_%d" % (i+1) for i in range(len(self.discr_layer_nodes))]

            inp = x if y is None else tf.concat(values=[x, y], axis=1)
            discr_data = self.gan_construct(inp, self.discr_layer_nodes, names=discr_layer_names,
                                            activations=self.discr_layer_activations, reuse=reuse_discr)

            discr_gen = None
            if prep_gen_input:
                # the discriminator's loss for the generated data needs to back-propagate through
                # the same network as that for the real data; hence reuse_discr=True
                gen_out = get_nn_layer(self.gen, layer_from_top=1)
                inp = gen_out if y is None else tf.concat(values=[gen_out, y], axis=1)
                discr_gen = self.gan_construct(inp, self.discr_layer_nodes, names=discr_layer_names,
                                               activations=self.discr_layer_activations, reuse=True)
        return discr_data, discr_gen

    def init_info_gan_network(self, x, reuse=False):
        return self.gan_construct(x, n_neurons=[self.n_classes], names=["q_out"],
                                  activations=[tf.nn.softmax], reuse=reuse)

    def init_session(self):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def training_op(self, loss, var_list=None, use_adam=False):
        if use_adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                       200, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        return optimizer.minimize(loss, var_list=var_list)

    def init_ano_gan_network(self, x=None, y=None, z=None):
        # here we assume that all networks have already been created
        # and hence we will set reuse=True.
        # Might be redundant if reuse=True before entering this method.
        ano_gan_net_G = self.generator(z=z, y=y, reuse_gen=True)
        ano_gan_G = ano_gan_net_G[len(ano_gan_net_G) - 1]
        ano_gan_net_D, _ = self.discriminator(x=ano_gan_G, y=y, reuse_discr=True, prep_gen_input=False)
        return ano_gan_net_G, ano_gan_net_D

    def ano_gan_outputs(self):
        """ Returns layers of generator and discrminator which will be used by AnoGAN
        Returns the last layers of discriminator and generator,
        and last-but-one of discriminator. The last-but-one layer of
        discriminator is used for the entropy regularization if the GAN is InfoGAN variety.
        """
        return self.ano_gan_net_G[len(self.ano_gan_net_G) - 1], \
               self.ano_gan_net_D[len(self.ano_gan_net_D) - 1], \
               self.ano_gan_net_D[len(self.ano_gan_net_D) - 2] if self.info_gan else None

    def get_gen_input_samples(self, n=1, gen_y=False):
        if gen_y and self.pvals is None:
            raise ValueError("pvals is required")
        y = None
        if gen_y:
            y = np.random.multinomial(1, pvals=self.pvals, size=n).astype(float)
        return np.random.uniform(low=self.unif_lo, high=self.unif_hi, size=(n, self.gen_input_dim)), y

    def get_gen_output_samples(self, z, y=None):
        feed_dict = {self.z: z}
        if self.conditional: feed_dict.update({self.y: y})
        x = self.session.run([get_nn_layer(self.gen, layer_from_top=1)], feed_dict=feed_dict)[0]
        return x

    def gan_layer(self, x, n_neurons, name, activation=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            n_inputs = int(x.get_shape()[1])
            stddev = 2. / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.get_variable("W", initializer=init)
            b = tf.get_variable("b", initializer=tf.zeros([n_neurons]))
            Z = tf.matmul(x, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z

    def gan_construct(self, x, n_neurons, names, activations, reuse=False):
        layer_input = x
        layers = list()
        for i, name in enumerate(names):
            hidden = self.gan_layer(layer_input, n_neurons=n_neurons[i], name=names[i],
                                    activation=activations[i], reuse=reuse)
            layers.append(hidden)
            layer_input = hidden
        return layers

    def fit(self, x, y=None):
        if self.session is None:
            self.init_session()

        fit_tm = Timer()
        for epoch in range(self.n_epochs):
            tm = Timer()
            i = 0
            for x_batch, y_batch in get_train_batches(x, y=y, batch_size=self.batch_size, shuffle=self.shuffle):
                # for the discriminator, use the true y labels
                z, _ = self.get_gen_input_samples(n=x_batch.shape[0], gen_y=False)
                feed_dict_discr = {self.x: x_batch, self.z: z}
                if self.conditional: feed_dict_discr.update({self.y: y_batch})
                self.session.run([self.discr_training_op], feed_dict=feed_dict_discr)
                if i % 1 == 0:  # train gen_loss only half as frequently as discr_loss
                    # z, y_ = self.get_gen_input_samples(n=x_batch.shape[0], gen_y=False)
                    feed_dict_gen = {self.z: z}
                    if self.conditional: feed_dict_gen.update({self.y: y_batch})
                    self.session.run([self.gen_training_op], feed_dict=feed_dict_gen)
                i += 1

            if self.listener is not None:
                self.listener(self, epoch=epoch, epoch_start_tm=tm)
        logger.debug(fit_tm.message("GAN fitted (max epochs: %d)" % self.n_epochs))

    def get_discriminator_probability(self, x, y=None):
        """ Returns the probability of the input under the current discriminator model

        :param x: np.ndarray
        :param y: np.array
            This is like a list of integers. Should contain the class labels (*not* one-hot-encoded).
        :return: np.array
            Probability of each input data
        """
        discr_data_out = get_nn_layer(self.discr_data, layer_from_top=1)
        if not self.conditional:
            feed_dict_discr = {self.x: x}
            probs = self.session.run([discr_data_out], feed_dict=feed_dict_discr)[0]
            probs = probs.reshape(-1)
        else:
            feed_dict_discr = {self.x: x}
            if y is not None:
                y_one_hot = np.zeros(shape=(x.shape[0], self.n_classes), dtype=np.float32)
                for i, c in enumerate(y):
                    y_one_hot[i, c] = 1.
                feed_dict_discr.update({self.y: y_one_hot})
                probs = self.session.run([discr_data_out], feed_dict=feed_dict_discr)[0]
                probs = probs.reshape(-1)
            else:
                # marginalize over all classes
                probs = np.zeros(x.shape[0], dtype=np.float32)
                for c  in range(self.n_classes):
                    y_one_hot = np.zeros(shape=(x.shape[0], self.n_classes), dtype=np.float32)
                    y_one_hot[:, c] = 1.
                    feed_dict_discr.update({self.y: y_one_hot})
                    probs_c = self.session.run([discr_data_out], feed_dict=feed_dict_discr)[0]
                    probs += self.pvals[c] * probs_c.reshape(-1)
        return probs

    def get_log_likelihood(self, x, n_samples=None, n_reps=10, gmm_min_k=2, gmm_max_k=10):
        """ Returns the avg. and std. dev. of log-likelihood of samples in x under the trained GAN model

        This is a simple but rough technique, and might not be very accurate.

        In the original GAN paper (Goodfellow et al. 2014), the authors
        employed a parzen-windows based technique. The Gaussian Mixture Model
        is a coarse approximation to it.
        """
        if n_samples is None:
            n_samples = x.shape[0]
        ll = []
        for i in range(n_reps):
            z, y = self.get_gen_input_samples(n=n_samples, gen_y=self.conditional)
            x_gen = self.get_gen_output_samples(z=z, y=y)
            try:
                gmm, _, _ = fit_gmm(x_gen, x_gen, min_k=gmm_min_k, max_k=gmm_max_k)
                ll.append(np.mean(gmm.score_samples(x)))
            except:
                logger.warning("Exception in iter %d/%d of gmm: %s" % (i+1, n_reps, str(sys.exc_info()[0])))

        ll = np.array(ll, dtype=np.float32)
        return np.mean(ll), np.std(ll)

    def get_anomaly_score_z(self, x, y_one_hot=None, z=None, ano_gan_lambda=0.1):
        """ Get the anomaly score with an initialized z

        This corresponds to one back-prop step in AnoGAN for computing
        a reconstructed image, for the input test point x, starting from an initial z

        :param x: np.ndarray (one row-vector)
            Test instance whose image needs to be reconstructed
        :param y_one_hot: np.ndarray (one row-vector)
        :param z: np.ndarray (one row-vector)
            If this is None, a random z will be sampled, else the input z will be use
        :param ano_gan_lambda: float
        :return: gen_x, ano_z, loss, loss_R, loss_D
            gen_x: the reconstructed image for 'x' starting from latent representation 'z'
            ano_z: the optimal computed by back-propagation
            loss: AnoGAN loss
            loss_R: reconstruction loss component of the AnoGAN loss
            loss_D: descrimination loss component of the AnoGAN loss
        """
        if not self.enable_ano_gan:
            raise RuntimeError("AnoGAN not enabled for this network")

        if z is None:
            z, _ = self.get_gen_input_samples(n=1)
        assign_z = self.ano_z.assign(z)
        self.session.run(assign_z)

        ano_gan_G, ano_gan_D, _ = self.ano_gan_outputs()
        feed_dict = {self.x: x, self.ano_gan_lambda: ano_gan_lambda}
        if self.conditional:
            feed_dict.update({self.y: y_one_hot})
        self.session.run([self.ano_gan_training_op], feed_dict=feed_dict)
        rets = self.session.run([ano_gan_G, self.ano_gan_loss, self.ano_z,
                                 self.ano_gan_loss_R, self.ano_gan_loss_D, self.ano_gan_info_loss], feed_dict=feed_dict)
        gen_x = rets[0]
        loss = rets[1]
        ano_z = rets[2]
        loss_R = rets[3]
        loss_D = rets[4] + rets[5]

        # make z values in [lo, hi]
        ano_z = self.clip(ano_z, lo=self.unif_lo, hi=self.unif_hi)

        return gen_x, ano_z, loss, loss_R, loss_D

    def get_anomaly_score_xy(self, x, y=None, z=None, ano_gan_lambda=0.1, tol=1e-3, max_iters=100):
        """ Computes anomaly score per instance and y (if conditional)

        :param x: np.ndarray
        :param y: int
            if y is None, and self.conditional==True, then pvals will be used
        :param z: np.ndarray
        :param tol: float
        :param max_iters: int
        :return: gen_x, z, loss, trace
        """
        tm = Timer()
        y_one_hot = None
        if self.conditional:
            if y is None:
                y_one_hot = np.array(self.pvals, dtype=np.float32).reshape((1, -1))
            else:
                y_one_hot = np.zeros(shape=(1, self.n_classes), dtype=np.float32)
                y_one_hot[0, y] = 1
        gen_x, z, loss, loss_R, loss_D = self.get_anomaly_score_z(x, y_one_hot=y_one_hot, z=z, ano_gan_lambda=ano_gan_lambda)
        losses = [loss]
        losses_R = [loss_R]
        losses_D = [loss_D]
        trace = []
        i = 0
        prev_loss = np.inf
        while i < max_iters and abs(loss - prev_loss) > tol:
            prev_loss = loss
            gen_x, z, loss, loss_R, loss_D = self.get_anomaly_score_z(x, y_one_hot=y_one_hot, z=z, ano_gan_lambda=ano_gan_lambda)
            losses.append(loss)
            losses_R.append(loss_R)
            losses_D.append(loss_D)
            trace.append(gen_x)
            i += 1
        logger.debug(tm.message("AnoGAN loss (iters: %d, final loss: %f)" % (i, losses[-1])))
        # logger.debug("losses:\n%s" % (str(losses)))
        return gen_x, z, loss, loss_R, loss_D, np.vstack(trace)

    def clip(self, z, lo, hi):
        z = np.minimum(np.maximum(z, lo), hi)
        return z

    def get_anomaly_score_x(self, x, ano_gan_lambda=0.1, tol=1e-3, max_iters=100, use_loss=True, mode_avg=True):
        """ Try each label and return the generated instance with best metrics (loss or distance)

        :param x: np.ndarray
        :param tol: float
        :param max_iters: int
        :param use_loss: bool
            if use_loss==True, then use the composite loss, else use the
            euclidean distance to find best regenerated point when the GAN is conditional
        :param mode_avg: bool
            If self.conditional==True and mode_avg==True, then soft-membership
            as defined by self.pvals will be used instead of individual
            one-hot-encoding membership.
        :return:
        """
        if mode_avg or not self.conditional:
            return self.get_anomaly_score_xy(x, y=None, z=None, ano_gan_lambda=ano_gan_lambda,
                                             tol=tol, max_iters=max_iters)

        gen_x = z = loss = loss_R = loss_D = trace = None
        best_dist = np.inf
        best_loss = np.inf
        for y in range(self.n_classes):
            gen_x_y, z_y, loss_y, loss_R_y, loss_D_y, trace_y = self.get_anomaly_score_xy(x, y=y, z=None,
                                                                                          ano_gan_lambda=ano_gan_lambda,
                                                                                          tol=tol, max_iters=max_iters)
            if use_loss:
                if loss_y < best_loss:
                    best_loss = loss_y
                    gen_x, z, loss, loss_R, loss_D, trace = (gen_x_y, z_y, loss_y, loss_R_y, loss_D_y, trace_y)
            else:
                dist = np.sum(np.square(np.subtract(x, gen_x_y)))
                if dist < best_dist:
                    best_dist = dist
                    gen_x, z, loss, loss_R, loss_D, trace = (gen_x_y, z_y, loss_y, loss_R_y, loss_D_y, trace_y)

        return gen_x, z, loss, loss_R, loss_D, trace

    def get_anomaly_score(self, x, ano_gan_lambda=0.1, tol=1e-3, max_iters=100, use_loss=True, mode_avg=True):
        """ Returns the anomaly score of test instance x

        :param x: np.ndarray (one row-vector)
        :param ano_gan_lambda: float
        :param tol: float
            loss tolerance to check for termination of back-propagation
            steps when computing reconstruction image
        :param max_iters: int
        :param use_loss: bool
            (applies only to conditional GAN and when mode_avg is False, default: True)
            If true, then employs the AnoGAN loss when selecting the best category for test instance
        :param mode_avg: bool
            (applies only to conditional GAN, default: True)
        :return:
        """
        losses = np.zeros(x.shape[0], dtype=np.float32)
        losses_R = np.zeros(x.shape[0], dtype=np.float32)
        losses_D = np.zeros(x.shape[0], dtype=np.float32)
        traces = []
        new_x = np.zeros(shape=x.shape, dtype=x.dtype)
        for i in range(x.shape[0]):
            gen_x, z, loss, loss_R, loss_D, trace = self.get_anomaly_score_x(x[[i]], ano_gan_lambda=ano_gan_lambda,
                                                                             tol=tol, max_iters=max_iters,
                                                                             use_loss=use_loss, mode_avg=mode_avg)
            new_x[i, :] = gen_x[0, :]
            losses[i] = loss
            losses_R[i] = loss_R
            losses_D[i] = loss_D
            traces.append(trace)
        return new_x, losses, losses_R, losses_D, traces

    def save_session(self, file_path, overwrite=False):
        if tf.train.checkpoint_exists(file_path):
            if overwrite:
                logger.debug("Overwriting existing checkpoint for prefix %s" % file_path)
            else:
                logger.debug("Checkpoint already exists for prefix %s" % file_path)
                return None
        saver = tf.train.Saver()
        save_path = saver.save(self.session, file_path)
        logger.debug("Saved session to path %s" % save_path)
        return save_path

    def load_session(self, file_path):
        if not tf.train.checkpoint_exists(file_path):
            logger.debug("Checkpoint does not exist for prefix %s" % file_path)
            return False
        if self.session is None:
            self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, file_path)
        logger.debug("Loaded saved session from path %s" % file_path)
        return True

    def close_session(self):
        if self.session is not None:
            self.session.close()
        self.session = None


def get_gan_option_list():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airline", required=False,
                        help="Dataset name")
    parser.add_argument("--results_dir", action="store", default="./temp",
                        help="Folder where the generated metrics will be stored")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--label_smoothing", action="store_true", default=False,
                        help="Whether to use one-sided label smoothing")
    parser.add_argument("--smoothing_prob", action="store", type=float, default=0.9,
                        help="Probability to use for one-sided label smoothing")
    parser.add_argument("--ano_gan_lambda", action="store", type=float, default=0.1,
                        help="The AnoGAN penalty term that balances reconstruction loss and discriminative loss")
    parser.add_argument("--info_gan", action="store_true", default=False,
                        help="Whether to use simple GAN or InfoGAN")
    parser.add_argument("--info_gan_lambda", action="store", type=float, default=1.0,
                        help="The InfoGAN penalty term")
    parser.add_argument("--conditional", action="store_true", default=False,
                        help="Whether to use simple GAN or Conditional GAN")
    parser.add_argument("--ano_gan", action="store_true", default=False,
                        help="Whether to enable AnoGAN functionality")
    parser.add_argument("--ano_gan_individual", action="store_true", default=False,
                        help="Whether to use each class individually for Conditional AnoGAN. "
                             "By default the pval metric will be used instead of one-hot-encoding during test evaluation")
    parser.add_argument("--ano_gan_use_dist", action="store_true", default=False,
                        help="Whether to use euclidean dist-based reconstruction error for Conditional AnoGAN. "
                             "By default, the composite loss will be used")
    parser.add_argument("--n_ano_gan_test", type=int, default=1, required=False,
                        help="Number of times AnoGAN loss will be computed for each test instance")
    parser.add_argument("--budget", type=int, default=1, required=False,
                        help="Budget for feedback")
    parser.add_argument("--n_epochs", type=int, default=200, required=False,
                        help="Max training epochs")
    parser.add_argument("--train_batch_size", type=int, default=25, required=False,
                        help="Batch size for stochastic gradient descent based training methods")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Whether to plot figures")
    return parser


class GanOpts(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.results_dir = args.results_dir
        self.randseed = args.randseed
        self.label_smoothing = args.label_smoothing
        self.smoothing_prob = args.smoothing_prob
        self.ano_gan_lambda = args.ano_gan_lambda
        self.ano_gan_individual = args.ano_gan_individual
        self.ano_gan_use_dist = args.ano_gan_use_dist
        self.info_gan = args.info_gan
        self.info_gan_lambda = args.info_gan_lambda
        self.conditional = args.conditional
        self.ano_gan = args.ano_gan
        self.ano_gan_individual = args.ano_gan_individual
        self.ano_gan_use_dist = args.ano_gan_use_dist
        self.n_ano_gan_test = args.n_ano_gan_test
        self.budget = args.budget
        self.n_epochs = args.n_epochs
        self.train_batch_size = args.train_batch_size
        self.log_file = args.log_file
        self.debug = args.debug
        self.plot = args.plot
        self.k = 0

    def get_opts_name_prefix(self):
        # ano_gan_sig = "_ano" if self.ano_gan else ""
        info_gan_sig = "_info" if self.info_gan else ""
        info_gan_lambda_sig = "" if self.info_gan_lambda == 1.0 else "_il%d" % int(self.info_gan_lambda*10)
        cond_sig = "_cond" if self.conditional else ""
        algo_sig = "%s%s_gan" % (cond_sig, info_gan_sig)
        k_sig = "_k%d" % self.k if self.k > 0 else ""
        smoothing_sig = "_ls%d" % (int(self.smoothing_prob*10)) if self.label_smoothing else ""
        name = "%s%s%s%s%s_%d" % (self.dataset, algo_sig, k_sig, smoothing_sig, info_gan_lambda_sig, self.n_epochs)
        return name

    def get_alad_metrics_name_prefix(self):
        return self.get_opts_name_prefix()

    def str_opts(self):
        name = self.get_alad_metrics_name_prefix()
        s = "%s" % name
        return s