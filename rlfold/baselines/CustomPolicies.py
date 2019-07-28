from stable_baselines.common.policies import ActorCriticPolicy, register_policy, RecurrentActorCriticPolicy
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
import rlfold.settings as settings
import numpy as np


def conv1d(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 1d convolutional layer for TensorFlow

    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)


        bias + tf.nn.conv1d(input_tensor, weight, stride=None, padding='VALID')
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)


def custom_cnn(scaled_images, params, **kwargs):
    """
    Custom CNN architecture 
    """
    s = params
    activ = getattr(tf.nn, s.activ)
    init_scale = s.conv_init_scale
    print(s.kernel_size)
    # First layer
    # odb = True if s.kernel_size[0][1] == 1 else False
    odb = False
    print('First: {}, odb: {}'.format(s.kernel_size[0], odb))
    out = activ(conv(scaled_images, 'c0', n_filters=s.filters[0], filter_size=s.kernel_size[0], stride=s.stride[0], init_scale=init_scale, one_dim_bias=odb, **kwargs))
    # Following layers
    for i, layer in enumerate(s.filters[1:]):
        print('Loop: {}'.format(s.kernel_size[i+1]))
        out = activ(conv(out, 'c{}'.format(i+1), n_filters=layer, filter_size=s.kernel_size[i+1], stride=s.stride[i+1], init_scale=init_scale, one_dim_bias=odb, **kwargs))

    out = conv_to_fc(out)
    return out

class CustomCnnPolicy(ActorCriticPolicy):
    """
    Custom CNN policy, requires a params dictionary (ParameterContainer) as an argument
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 params=None, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        params = settings.ParameterContainer(**params)
        print(params)
        init_scale = params.pd_init_scale
        activ = getattr(tf.nn, params.activ)
        initializer = getattr(tf, params.kernel_initializer)

        with tf.variable_scope('model', reuse=reuse):
            extracted_features = custom_cnn(self.processed_obs, params, )
            flattened = tf.layers.flatten(extracted_features)
            
            # Shared layers
            shared = flattened
            for i, layer in enumerate(params.shared):
                shared = activ(tf.layers.dense(shared, layer, name='fc_shared'+str(i), kernel_initializer=initializer))

            # Policy net
            pi_h = shared
            for i, layer in enumerate(params.h_actor):
                pi_h = activ(tf.layers.dense(pi_h, layer, name='pi_fc'+str(i), kernel_initializer=initializer))
            pi_latent = pi_h

            # Value net
            vf_h = shared
            for i, layer in enumerate(params.h_critic):
                vf_h = activ(tf.layers.dense(vf_h, layer, name='vf_fc'+str(i), kernel_initializer=initializer))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, init_bias=params.init_bias)

        self._value_fn = value_fn
        self._setup_init()


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

# register_policy('CustomCnnPolicy', CustomCnnPolicy)

class CustomMixedCnnPolicy(ActorCriticPolicy):
    """
    1D Custom CNN
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 params=None, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        params = settings.ParameterContainer(**params)
        init_scale = params.pd_init_scale
        activ = getattr(tf.nn, params.activ)
        initializer = getattr(tf, params.kernel_initializer)

        with tf.variable_scope('model', reuse=reuse):
            pass

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CustomLstmPolicy():
    pass

class CustomFcPolicy():
    pass


class CustomDDPGPolicy():
    pass
