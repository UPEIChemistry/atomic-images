from keras import backend as K
from keras.layers import Input, Lambda

from atomic_images.layers import (AtomicNumberBasis, DistanceMatrix,
                                  GaussianBasis, OneHot)


def main():
    r = Input(shape=(29, 3))
    z = Input(shape=(29,), dtype='uint8')

    dist = DistanceMatrix()(r)
    one_hot_z = OneHot(max_atomic_number=9)(z)

    gaussians = GaussianBasis(0.2, 0.2, 0.0, 8.0)(dist)
    interaction_images = AtomicNumberBasis()([one_hot_z, gaussians])
    atomic_images = Lambda(lambda x: K.sum(x, axis=2))(interaction_images)
    molecular_images = Lambda(lambda x: K.sum(x, axis=1))(atomic_images)


if __name__ == '__main__':
    main()
