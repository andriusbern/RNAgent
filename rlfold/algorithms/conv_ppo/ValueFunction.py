
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import rusher.settings as settings
import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, learningParameters, networkParameters):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        self.lParameters = learningParameters
        self.nParameters = networkParameters
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        if self.nParameters['network_type'] == 'convolutional':
            self._build_conv_graph()
        else:
            self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    
    def _build_conv_graph(self):
        """
        Builds a convolutional network for value approximation
        """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name='obs_valfunc')
            self.reshaped = tf.reshape(self.obs_ph, [-1, 64, 64, 1])
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self.lr = self.lParameters['lr_critic'] #/ np.sqrt(hid2_size)  # 1e-3 empirically determined
            # 3 hidden layers with tanh activations
            out = tf.layers.conv2d(inputs=self.reshaped,
                                strides=self.nParameters['strides'][0],
                                filters=self.nParameters['filters'][0],
                                kernel_size=self.nParameters['kernel'][0],
                                activation=getattr(tf.nn, self.nParameters['activation_h_c']), name='s')

            for i, layer in enumerate(self.nParameters['filters'][1:]):
                print(i, layer)
                out = tf.layers.conv2d(inputs=out,
                                    strides=self.nParameters['strides'][i+1],
                                    filters=layer,
                                    kernel_size=self.nParameters['kernel'][i+1],
                                    activation=getattr(tf.nn, self.nParameters['activation_h_c']))  
            out = tf.layers.flatten(out)
            print('Flattened policy: {}'.format(tf.shape(out)))

            # Hidden layers
            for layer in self.nParameters['hidden_layers_a']:
                out = tf.layers.dense(out, layer, activation=getattr(tf.nn, self.nParameters['activation_h_c']))

            # out = tf.layers.dense(out, 16, activation=tf.nn.relu)
            out = tf.layers.dense(out, 1)

            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        """
        Construct TensorFlow graph, including loss function, init op and train op
        """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            hid1_size = self.obs_dim * 5  
            hid3_size = 16  
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
           
            self.lr = 1e-3
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y):
        """ 

        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         
        exp_var = 1 - np.var(y - y_hat) / np.var(y) 

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def save(self):
        """
        Save the tensorflow weights file
        """
        path = '{}_{}_VALUE.cpkt'.format('BP', random.random.randint(100000000))
        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1)
        saver.save(self.sess, path)
        print('Trained model saved at {}'.format(path))