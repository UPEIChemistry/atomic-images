import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda, Reshape
from keras.models import Model

from atomic_images.layers import (AtomicNumberBasis, DistanceMatrix,
                                  DummyAtomMasking, GaussianBasis, OneHot,
                                  Unstandardization)


np.random.seed(1234)


def main():
    n_atoms = 10
    batch_size = 32
    n_batches = 100
    max_z = 6
    n_props = 4

    dummy_r = np.random.normal(loc=0, scale=3.0, size=(n_batches * batch_size, n_atoms, 3))
    dummy_z = np.random.randint(0, max_z, size=(n_batches * batch_size, n_atoms))
    dummy_x = np.random.normal(loc=-500, scale=50.0, size=(n_batches * batch_size, n_props))

    mu = np.mean(dummy_x, axis=0)
    sigma = np.std(dummy_x, axis=0)

    r = Input(shape=(n_atoms, 3))
    z = Input(shape=(n_atoms,), dtype='uint8')

    dist = DistanceMatrix()(r)
    gaussians = GaussianBasis(
        min_value=0.0,
        max_value=8.0,
        width=0.2,
        spacing=0.2
    )(dist)
    one_hot_z = OneHot(max_atomic_number=max_z)(z)

    interaction_images = DummyAtomMasking(atom_axes=[1, 2])(
        [one_hot_z, gaussians]
    )

    interaction_images = AtomicNumberBasis()([one_hot_z, gaussians])
    atomic_images = Lambda(lambda x: K.sum(x, axis=2))(interaction_images)
    at_shape = K.int_shape(atomic_images)

    x = Reshape((at_shape[1], at_shape[2] * at_shape[3]))(atomic_images)

    x = Dense(100, activation='relu')(x)
    x = Dense(dummy_x.shape[-1], activation='linear')(x)

    unstd = Unstandardization(mu=mu, sigma=sigma, trainable=True)
    x = unstd([one_hot_z, x])

    x = DummyAtomMasking()([one_hot_z, x])
    x = Lambda(lambda x: K.sum(x, axis=1))(x)

    def mu_metric(y_true, y_pred):
        return unstd.mu

    def sigma_metric(y_true, y_pred):
        return unstd.sigma

    model = Model([z, r], x)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', mu_metric, sigma_metric])

    model.fit([dummy_z, dummy_r], dummy_x, epochs=25, verbose=1,
              batch_size=batch_size)


if __name__ == '__main__':
    main()
