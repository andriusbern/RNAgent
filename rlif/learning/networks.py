import os, sys
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, RecurrentActorCriticPolicy, FeedForwardPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, seq_to_batch, batch_to_seq, lstm
from rlif.settings import ConfigManager as settings
from rlif.settings import ParameterContainer
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
    # First layer
    odb = True if s.kernel_size[0][1] == 1 else False
    # odb = True
    out = activ(conv(
        scaled_images, 
        'c0', 
        n_filters=s.filters[0], 
        filter_size=s.kernel_size[0], 
        stride=s.stride[0], 
        init_scale=init_scale, 
        one_dim_bias=odb, 
        **kwargs))

    # Following layers
    for i, layer in enumerate(s.filters[1:]):
        out = activ(conv(
            out, 
            'c{}'.format(i+1), 
            n_filters=layer, 
            filter_size=s.kernel_size[i+1], 
            stride=s.stride[i+1], 
            one_dim_bias=odb, 
            **kwargs))#, init_scale=initinit_scale, 

    out = conv_to_fc(out)
    return out

class CustomCnnPolicy(ActorCriticPolicy):
    """
    Custom CNN policy, requires a params dictionary (ParameterContainer) as an argument
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 params=None, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        params = ParameterContainer(**params)
        init_scale = params.pd_init_scale
        activ = getattr(tf.nn, params.activ)
        initializer = getattr(tf, params.kernel_initializer)

        with tf.variable_scope('model', reuse=reuse):
            extracted_features = custom_cnn(self.processed_obs, params, )
            flattened = tf.layers.flatten(extracted_features)
            
            # Shared layers
            shared = flattened
            for i, layer in enumerate(params.shared):
                shared = activ(tf.layers.dense(shared, layer, name='fc_shared'+str(i)))

            # Policy net
            pi_h = shared
            for i, layer in enumerate(params.h_actor):
                pi_h = activ(tf.layers.dense(pi_h, layer, name='pi_fc'+str(i)))
            pi_latent = pi_h

            # Value net
            vf_h = shared
            for i, layer in enumerate(params.h_critic):
                vf_h = activ(tf.layers.dense(vf_h, layer, name='vf_fc'+str(i)))
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


class CustomMixedCnnPolicy(ActorCriticPolicy):
    """
    1D Custom CNN
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 params=None, **kwargs):
        super(CustomMixedCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        params = ParameterContainer(**params)
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

class CustomMlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, params=None, **kwargs):
        config = params
        net_architecture = config['shared']
        net_architecture.append(dict(pi=config['h_actor'],
                                     vf=config['h_critic']))

        print(net_architecture)
        super(CustomMlpPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, net_arch=net_architecture, feature_extraction="mlp")


class CustomCnnLnLstmPolicy(RecurrentActorCriticPolicy):
    
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=32, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=custom_cnn, layer_norm=True, feature_extraction="cnn", params=None,
                 **kwargs):
        super(CustomCnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))
        config = params
        params = ParameterContainer(**params)
        init_scale = params.pd_init_scale
        activ = getattr(tf.nn, params.activ)
        initializer = getattr(tf, params.kernel_initializer)
        self._kwargs_check(feature_extraction, kwargs)
        net_arch = config['shared']
        net_arch.append(dict(pi=config['h_actor'],
                             vf=config['h_critic']))

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, params)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            # if feature_extraction == "cnn":
            #     raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                extracted_features = cnn_extractor(self.processed_obs, params)
                
                latent = tf.layers.flatten(extracted_features)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
