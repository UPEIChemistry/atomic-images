import tensorflow as tf
from keras import backend as K
from tensorflow.python import debug as tf_debug

from keras_model_loader import kml

data_loader_template = kml.import_module('data_loader.py',
                                         name='data_loader')
neural_network_template = kml.import_module('neural_network.py',
                                            name='neural_network')


class Builder(kml.ModelBuilder):
    def get_model(self, **kwargs):
        return neural_network_template.AtomicImagesModel()

    def get_model_hyperparameters(self, **kwargs):
        return {
            'batch_size': 32,
            'optimizer': {
                'optimizer': 'adam',
                'config': {
                    'lr': 1e-3
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
                        'activation': 'relu'
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

    def get_data_loader(self, **kwargs):
        return data_loader_template.DataLoader()

    def get_optimization_runner(self, **kwargs):
        from keras_model_loader.optimization.runner import CudaRunner
        return CudaRunner(self)

    def get_cross_validation_runner(self, **kwargs):
        from keras_model_loader.cross_validation.runner import CudaRunner
        return CudaRunner(self)

    def get_callbacks(self, **kwargs):
        from keras.callbacks import TensorBoard
        cbks = []
        if kwargs.get('use_tensorboard'):
            cbks.append(
                TensorBoard(
                    kwargs.get('tensorboard_dir', './tensorboard_logs'),
                    histogram_freq=1,
                    write_grads=True
                )
            )
        return cbks


def init(cmd):
    # sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    return Builder()
