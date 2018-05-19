from keras import backend as K
from keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                          Layer, Reshape, Lambda, Conv2D, TimeDistributed)
from keras.initializers import get as get_initializer


class Predictor(Layer):
    DEFAULT_LAYER_PARAMS = {
        'interaction_dense0': {
            'units': 256,
            'activation': 'selu',
            'batchnorm': False
        },
        'interaction_dense1': {
            'units': 256,
            'activation': 'selu',
            'batchnorm': False
        },
        'interaction_dense2': {
            'units': 256,
            'activation': 'tanh',
            'batchnorm': False
        },
        'atomwise_dense0': {
            'units': 1,
            'activation': 'linear'
        }
    }

    def __init__(self, layer_params, **kwargs):
        """Property predictor

        Args:
            layer_params (dict): layer parameters
        """
        super(Predictor, self).__init__(**kwargs)
        self.layer_params = layer_params

    @staticmethod
    def dense_block(dense_dict, x, name):
        activation = dense_dict.get('activation')
        if activation == 'selu':
            kernel_initializer = 'lecun_normal'
        elif activation == 'relu':
            kernel_initializer = 'he_uniform'

        kernel_initializer = dense_dict.get('kernel_initializer', 'glorot_uniform')
        if isinstance(kernel_initializer, dict):
            kernel_initializer = get_initializer(kernel_initializer)

        bias_initializer = dense_dict.get('bias_initializer', 'zeros')
        if isinstance(bias_initializer, dict):
            bias_initializer = get_initializer(bias_initializer)

        x = Dense(
            units=dense_dict['units'],
            activation=dense_dict.get('activation'),
            name=name,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )(x)

        if dense_dict.get('dropout_rate'):
            if activation == 'selu':
                x = AlphaDropout(rate=dense_dict['dropout_rate'],
                                 name='{0}_dropout'.format(name))(x)
            else:
                x = Dropout(rate=dense_dict['dropout_rate'],
                            name='{0}_dropout'.format(name))(x)
        if dense_dict.get('batchnorm'):
            x = BatchNormalization(
                axis=-1,
                name='{0}_norm'.format(name)
            )(x)
        return x


    @staticmethod
    def convolutional_block(conv_dict, x, name):
        activation = conv_dict.get('activation')
        if activation == 'selu':
            kernel_initializer = 'lecun_normal'
        elif activation == 'relu':
            kernel_initializer = 'he_uniform'

        kernel_initializer = conv_dict.get('kernel_initializer', 'glorot_uniform')
        if isinstance(kernel_initializer, dict):
            kernel_initializer = get_initializer(kernel_initializer)

        bias_initializer = conv_dict.get('bias_initializer', 'zeros')
        if isinstance(bias_initializer, dict):
            bias_initializer = get_initializer(bias_initializer)

        x = TimeDistributed(
            Conv2D(
                filters=conv_dict['filters'],
                kernel_size=conv_dict['kernel_size'],
                activation=conv_dict.get('activation'),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            ),
            name=name
        )(x)

        if conv_dict.get('dropout_rate'):
            if activation == 'selu':
                x = AlphaDropout(rate=conv_dict['dropout_rate'],
                                 name='{0}_dropout'.format(name))(x)
            else:
                x = Dropout(rate=conv_dict['dropout_rate'],
                            name='{0}_dropout'.format(name))(x)
        return x

    def call(self, inputs=None):
        x = inputs
        input_shape = K.int_shape(x)
        atoms_shape = input_shape[1]
        gaussians_shape = input_shape[3]
        numbers_shape = input_shape[4]

        # Interaction-wise Convolutional layers
        i = 0
        for i in range(100):
            conv_key = 'interactionwise_conv{i}'.format(i=i)
            if conv_key not in self.layer_params:
                break
            if i == 0:
                x = Reshape(
                    (atoms_shape * atoms_shape, gaussians_shape, numbers_shape, 1),
                    name='interactionwise_conv_expand'
                )(x)
            conv_dict = self.layer_params[conv_key]
            x = Predictor.convolutional_block(conv_dict, x, name=conv_key)

        x_shape = K.int_shape(x)
        if i > 0:
            gaussians_shape = x_shape[-3]
            numbers_shape = x_shape[-2]
            filters_shape = x_shape[-1]
            new_shape = (atoms_shape, atoms_shape, gaussians_shape * numbers_shape * filters_shape)
        else:
            gaussians_shape = x_shape[-2]
            numbers_shape = x_shape[-1]
            filters_shape = 1
            new_shape = (atoms_shape, atoms_shape, gaussians_shape * numbers_shape)
        x = Reshape(new_shape, name='interactionwise_conv_flatten')(x)

        # Interaction-wise Dense layers
        for i in range(100):
            dense_key = 'interaction_dense{i}'.format(i=i)
            if dense_key not in self.layer_params:
                break
            dense_dict = self.layer_params[dense_key]
            x = Predictor.dense_block(dense_dict, x, name=dense_key)

        x = Lambda(
            lambda x_: K.sum(x_, axis=2),
            name='interactions_to_atomwise'
        )(x)

        # Atom-wise Convolutional layers
        i = 0
        for i in range(100):
            conv_key = 'atomwise_conv{i}'.format(i=i)
            if conv_key not in self.layer_params:
                break
            if i == 0:
                new_shape = (atoms_shape, gaussians_shape, numbers_shape, filters_shape)
                x = Reshape(
                    new_shape,
                    name='atomwise_conv_expand'
                )(x)
            conv_dict = self.layer_params[conv_key]
            x = Predictor.convolutional_block(conv_dict, x, name=conv_key)

        if i > 0:
            x_shape = K.int_shape(x)
            gaussians_shape = x_shape[-3]
            numbers_shape = x_shape[-2]
            filters_shape = x_shape[-1]
            new_shape = (atoms_shape, gaussians_shape * numbers_shape * filters_shape)
            x = Reshape(
                new_shape,
                name='atomwise_conv_flatten'
            )(x)

        # Atom-wise Dense Layers
        for i in range(100):
            dense_key = 'atomwise_dense{i}'.format(i=i)
            if dense_key not in self.layer_params:
                break
            dense_dict = self.layer_params[dense_key]
            x = Predictor.dense_block(dense_dict, x, name=dense_key)

        return x

    def compute_output_shape(self, inputs):
        output_shape = inputs

        # Atom-wise Dense Layers
        for i in range(100):
            dense_key = 'atomwise_dense{i}'.format(i=i)
            if dense_key not in self.layer_params:
                break
            dense_dict = self.layer_params[dense_key]
            output_shape = output_shape[0:2] + (dense_dict['units'],)

        return output_shape

    def get_config(self):
        config = {
            'layer_params': self.layer_params
        }
        base_config = super(Predictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
