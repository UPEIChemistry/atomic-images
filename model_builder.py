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

    def get_data_loader(self, **kwargs):
        return data_loader_template.DataLoader()

    def get_optimization_runner(self, **kwargs):
        from keras_model_loader.optimization.runner import CudaRunner
        return CudaRunner(self)

    def get_cross_validation_runner(self, **kwargs):
        from keras_model_loader.cross_validation.runner import CudaRunner
        return CudaRunner(self)


def init(cmd):
    # sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    return Builder()
