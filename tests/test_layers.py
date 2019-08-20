from atomic_images.layers import DistanceMatrix, GaussianBasis, Unstandardization
import numpy as np
import tensorflow as tf


def run_model_test(inputs, targets, model):
    model.compile(loss='mae', optimizer='adam')
    return model.fit(inputs, targets, epochs=3)


def test_distance_matrix(random_cartesians_and_z, distance_matrix_model):
    run_model_test(
        random_cartesians_and_z[0], np.random.rand(2, 10, 10), distance_matrix_model
    )


def test_gaussian_basis(random_cartesians_and_z, gaussian_basis_model):
    r = random_cartesians_and_z[0]
    run_model_test(
        DistanceMatrix()(r).numpy(),
        np.random.rand(2, 10, 10, 50),
        gaussian_basis_model
    )


def test_atomic_basis(random_cartesians_and_z, atomic_num_basis_model):
    r, z = random_cartesians_and_z
    rbf = GaussianBasis()(DistanceMatrix()(r)).numpy()
    one_hot = tf.squeeze(tf.one_hot(z, np.max(z) + 1)).numpy()
    run_model_test(
        [one_hot, rbf],
        np.random.rand(2, 10, 10, 50, 5),
        atomic_num_basis_model
    )


def test_molecular_unstandardize(single_layer_model_class, trainable):
    atomic_energies, target = np.random.rand(2, 10, 1), np.random.rand(2, 10, 1)
    history = run_model_test(
        atomic_energies,
        target,
        single_layer_model_class(
            Unstandardization(
                np.mean(atomic_energies),
                np.std(atomic_energies),
                trainable=trainable
            )
        )
    )
    if trainable:
        assert history.history['loss'][0] >= history.history['loss'][1]
    else:
        assert history.history['loss'][0] == history.history['loss'][1]


def test_atomic_unstandardize(random_cartesians_and_z, single_layer_model_class, trainable):
    z = random_cartesians_and_z[1]
    one_hot = tf.squeeze(tf.one_hot(z, np.max(z) + 1)).numpy()
    atomic_energies, target = np.random.rand(2, 10, 1), np.random.rand(2, 10, 1)
    atomic_mu = np.random.rand(5, 1)
    atomic_sigma = np.random.rand(5, 1)
    history = run_model_test(
        [one_hot, atomic_energies],
        target,
        single_layer_model_class(
            Unstandardization(
                atomic_mu,
                atomic_sigma,
                trainable=trainable
            )
        )
    )
    if trainable:
        assert history.history['loss'][0] >= history.history['loss'][1]
    else:
        assert history.history['loss'][0] == history.history['loss'][1]


def test_dummy_atom_masking(random_cartesians_and_z, dummy_masking_model):
    r, z = random_cartesians_and_z
    one_hot = tf.squeeze(tf.one_hot(z, np.max(z) + 1)).numpy()
    run_model_test(
        [one_hot, r],
        np.random.rand(2, 10, 3),
        dummy_masking_model
    )
