from tensorflow.python.keras.models import Model
from atomic_images import layers
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--eager', action='store_true', default=False
    )
    parser.addoption(
        '--trainable', action='store_true', default=False
    )


@pytest.fixture(scope='session')
def trainable(request):
    return request.config.getoption('--trainable')


@pytest.fixture(scope='session')
def random_cartesians_and_z():
    z = np.random.randint(5, size=(2, 10, 1))
    r = np.random.rand(2, 10, 3).astype('float32')
    return r, z


# =========== Model Fixtures =========== #
class SingleLayerModel(Model):
    def __init__(self,
                 layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, training=None, mask=None):
        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)


@pytest.fixture(scope='session')
def single_layer_model_class():
    return SingleLayerModel


@pytest.fixture(scope='session')
def distance_matrix_model(request):
    dynamic = request.config.getoption('--eager')
    return SingleLayerModel(layers.DistanceMatrix(dynamic=dynamic))


@pytest.fixture(scope='session')
def gaussian_basis_model(request):
    dynamic = request.config.getoption('--eager')
    return SingleLayerModel(layers.GaussianBasis(dynamic=dynamic))


@pytest.fixture(scope='session')
def cosine_basis_model(request):
    dynamic = request.config.getoption('--eager')
    return SingleLayerModel(layers.GaussianBasis(dynamic=dynamic))


@pytest.fixture(scope='session')
def atomic_num_basis_model(request):
    dynamic = request.config.getoption('--eager')
    return SingleLayerModel(layers.AtomicNumberBasis(dynamic=dynamic))


@pytest.fixture(scope='session')
def dummy_masking_model(request):
    dynamic = request.config.getoption('--eager')
    return SingleLayerModel(layers.DummyAtomMasking(dynamic=dynamic))
