import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Lambda
from keras.models import Model

from atomic_images.layers import (AtomicNumberBasis, AtomRefOffset,
                                  DistanceMatrix, GaussianBasis, OneHot,
                                  DummyAtomMasking)
from keras_model_loader import kml

predictor_mod = kml.import_module('predictor.py', name='predictor')
Predictor = predictor_mod.Predictor
kml.register_class(OneHot, AtomicNumberBasis, GaussianBasis, DistanceMatrix,
                   Predictor, AtomRefOffset, DummyAtomMasking)


class AtomicImagesModel(kml.KerasModel):
    def __init__(self, *args, **kwargs):
        super(AtomicImagesModel, self).__init__(*args, **kwargs)

    def get_default_hyperparameters(self, **kwargs):
        defaults = {
            'batch_size': 32,
            'optimizer': {
                'optimizer': 'sgd',
                'config': {
                    'lr': 1e-2
                }
            },
            'inputs': {
                'coordinates_shape': (29, 3),
                'atomic_numbers_shape': (29,),
                'max_z': 9
            },
            'gaussian_basis': {
                'width': 0.2,
                'spacing': 0.2,
                'min_value': -1.0,
                'max_value': 15.0
            },
            'predictor': {
                'loss': 'mse',
                'metrics': ['mae', 'mse'],
                'standardize_outputs': True,
                'offset_by_ref': False,
                'layers': {
                    'atomwise_conv0': {
                        'kernel_size': [7, 1],
                        'filters': 1,
                        'activation': 'relu',
                        'bias_initializer': 'ones'
                    },
                    'atomwise_dense0': {
                        'units': 1,
                        'activation': 'linear',
                        'batchnorm': False
                    },
                    # 'atomwise_dense1': {
                    #     'units': 256,
                    #     'activation': 'elu',
                    #     'batchnorm': True
                    # },
                    # 'atomwise_dense1': {
                    #     'units': 1,
                    #     'activation': 'linear'
                    # }
                }
            }
        }
        return defaults

    def build(self, **kwargs):
        hps = self.hyperparameters

        #
        # Input
        #
        if self.data_loader is not None:
            input_shapes = self.data_loader.X_shape
            max_z = np.max(self.data_loader.X_train[0])
            atom_ref = self.data_loader.atom_ref

            standardize = hps['predictor/standardize_outputs']
            valid_offset = hps['predictor/offset_by_ref'] and atom_ref is not None

            if standardize:
                if valid_offset:
                    y_mu = float(self.data_loader.y_diff_mu_per_atom)
                    y_sigma = float(self.data_loader.y_diff_sigma_per_atom)
                else:
                    y_mu = float(self.data_loader.y_mu_per_atom)
                    y_sigma = float(self.data_loader.y_sigma_per_atom)
                unstandardization = Lambda(
                    lambda x: x * y_sigma + y_mu,
                    name='unstandardization'
                )
        else:
            input_shapes = [hps['inputs/atomic_numbers_shape'],
                            hps['inputs/coordinates_shape']]
            max_z = hps['inputs/max_z']
            unstandardization = Lambda(
                lambda x: x * float(hps['inputs/y_sigma']) + float(hps['inputs/y_mu']),
                name='unstandardization'
            )
            atom_ref = None

        # Build input tensors
        input_types = ['uint8', 'float32']
        input_names = ['atomic_numbers_input', 'coordinates_input']
        input_info = zip(input_shapes, input_names, input_types)
        self.x_in = [Input(shape=item,
                           name=name,
                           dtype=type_)
                     for item, name, type_ in input_info]

        one_hot_numbers = OneHot(
            max_atomic_number=max_z
        )(self.x_in[0])
        distance_matrix = DistanceMatrix()(self.x_in[1])
        gaussian_basis = GaussianBasis(
            **hps['gaussian_basis']
        )(distance_matrix)
        interaction_images = DummyAtomMasking(atom_axes=[1, 2])(
            [one_hot_numbers, gaussian_basis]
        )
        interaction_images = AtomicNumberBasis()(
            [one_hot_numbers, interaction_images]
        )

        #
        # Predictor
        #
        predictor_params = hps['predictor/layers']
        self.property_predictor = Predictor(
            layer_params=predictor_params,
            name='predictor_layer'
        )
        pred_out = self.property_predictor(interaction_images)

        if hps['predictor/standardize_outputs']:
            pred_out = unstandardization(pred_out)
        if hps['predictor/offset_by_ref'] and atom_ref is not None:
            pred_out = AtomRefOffset(atom_ref)([one_hot_numbers, pred_out])
        pred_out = DummyAtomMasking()([one_hot_numbers, pred_out])
        self.y_out = Lambda(lambda x: K.sum(x, axis=1),
                            name='predictor_output')(pred_out)
        self.model = Model(self.x_in, self.y_out,
                           name='predictor_model')

    def compile(self, **kwargs):
        hps = self.hyperparameters

        if 'config' in hps['optimizer']:
            optimizer = optimizers.get(
                {
                    'class_name': hps['optimizer/optimizer'],
                    'config': hps['optimizer/config']
                }
            )
        else:
            optimizer = hps['optimizer/optimizer']

        # Compile
        self.model.compile(
            loss=hps['predictor/loss'],
            metrics=hps['predictor/metrics'],
            optimizer=optimizer
        )

    def get_callbacks(self, **kwargs):
        from keras.callbacks import TensorBoard
        cbks = []
        if kwargs.get('use_tensorboard'):
            cbks.append(
                TensorBoard(
                    kwargs.get('tensorboard_dir', './tensorboard_logs'),
                    histogram_freq=1
                )
            )
        return cbks
