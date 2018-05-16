from keras.layers import Layer
from keras import backend as K
from atomic_images import keras_utils

import numpy as np


class OneHot(Layer):
    """One-hot atomic number layer

    Converts a list of atomic numbers to one-hot vectors

    Input: atomic numbers (batch, atoms)
    Output: one-hot atomic number (batch, atoms, atomic_number)
    """
    def __init__(self,
                 max_atomic_number,
                 **kwargs):
        # Parameters
        self.max_atomic_number = max_atomic_number

        super(OneHot, self).__init__(**kwargs)

    def call(self, inputs):
        atomic_numbers = inputs
        return K.one_hot(atomic_numbers,
                         self.max_atomic_number + 1)

    def compute_output_shape(self, input_shapes):
        atomic_numbers = input_shapes
        return atomic_numbers + (self.max_atomic_number + 1,)

    def get_config(self):
        config = {
            'max_atomic_number': self.max_atomic_number
        }
        base_config = super(OneHot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistanceMatrix(Layer):
    """
    Distance matrix layer

    Expands Cartesian coordinates into a distance matrix.

    Input: coordinates (batch, atoms, 3)
    Output: distance matrix (batch, atoms, atoms)
    """
    def call(self, positions):
        # `positions` should be Cartesian coordinates of shape
        #    (batch, atoms, 3)
        v1 = K.expand_dims(positions, axis=2)
        v2 = K.expand_dims(positions, axis=1)

        diff = v2 - v1
        sq_diff = K.square(diff)
        summed = K.sum(sq_diff, axis=-1)
        return K.sqrt(summed)

    def compute_output_shape(self, positions_shape):
        return (positions_shape[0], positions_shape[1], positions_shape[1])


class GaussianBasis(Layer):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)     where: u is linspace(min_value, max_value, ceil((max_value - min_value) / width))
          2 * w^2              w is width

    Input: distance_matrix (batch, atoms, atoms)
    Output: distance_matrix expanded into Gaussian basis set

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of Gaussians
        spacing (float, optional): spacing between Gaussians
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
                (batch, atoms, atoms, n_gaussians)
    """
    def __init__(self, min_value=-1, max_value=9, width=0.2, spacing=0.1,
                 self_thresh=1e-5, include_self_interactions=True,
                 endpoint=False, **kwargs):
        super(GaussianBasis, self).__init__(**kwargs)
        self._n_centers = int(np.ceil((max_value - min_value) / spacing))
        self.min_value = min_value
        self.max_value = max_value
        self.spacing = spacing
        self.width = width
        self.self_thresh = self_thresh
        self.include_self_interactions = include_self_interactions
        self.endpoint = endpoint

        self._gamma = -0.5 / (self.width  ** 2)

    def call(self, distance_matrix):
        distances = K.expand_dims(distance_matrix, -1)
        mu = keras_utils.linspace(self.min_value, self.max_value, self._n_centers,
                                  endpoint=self.endpoint)
        mu = K.reshape(mu, (1, 1, 1, -1))

        gaussians = K.exp(self._gamma * K.square(distances - mu))

        if not self.include_self_interactions:
            mask = K.cast(distances >= self.self_thresh, K.floatx())
            gaussians *= mask

        return gaussians

    def compute_output_shape(self, distance_matrix_shape):
        return (distance_matrix_shape[0],
                distance_matrix_shape[1],
                distance_matrix_shape[2],
                self._n_centers,)

    def get_config(self):
        config = {
            'width': self.width,
            'spacing': self.spacing,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'self_thresh': self.self_thresh,
            'include_self_interactions': self.include_self_interactions,
            'endpoint': self.endpoint
        }
        base_config = super(GaussianBasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AtomicNumberBasis(Layer):
    """Expands Gaussian matrix into the one-hot atomic numbers basis

    Inputs:
        one_hot_numbers  (batch, atoms, max_atomic_number + 1)
        gaussians_matrix  (batch, atoms, atoms, n_gaussians)
    Output:
        gaussians_atom_matrix  (batch, atoms, atoms, n_gaussians, max_atomic_number + 1)
    """
    def __init__(self, max_atomic_number=None, zero_dummy_atoms=False, **kwargs):
        super(AtomicNumberBasis, self).__init__(**kwargs)
        self.max_atomic_number = max_atomic_number
        self.zero_dummy_atoms = zero_dummy_atoms

    def call(self, inputs):
        if len(inputs) == 3:
            atomic_numbers, one_hot_numbers, gaussian_mat = inputs
        else:
            atomic_numbers, gaussian_mat = inputs
            atomic_numbers_shape = K.int_shape(atomic_numbers)
            # If shape is 3-long, one-hot
            if len(atomic_numbers_shape) == 3:
                one_hot_numbers = atomic_numbers
                atomic_numbers = K.argmax(one_hot_numbers,
                                          axis=-1)
            else:
                one_hot_numbers = K.one_hot(
                    atomic_numbers,
                    self.max_atomic_number + 1
                )

        gaussian_mat = K.expand_dims(gaussian_mat, axis=-1)
        if self.zero_dummy_atoms:
            mask = K.eye(one_hot_numbers.shape[-1], dtype=K.floatx())
            mask[0] = 0
            one_hot_numbers = K.dot(one_hot_numbers, mask)
        one_hot_numbers = K.expand_dims(one_hot_numbers, axis=1)
        one_hot_numbers = K.expand_dims(one_hot_numbers, axis=3)
        return gaussian_mat * one_hot_numbers

    def compute_output_shape(self, input_shapes):
        if len(input_shapes) == 3:
            atomic_numbers_shape, one_hot_numbers_shape, gaussian_mat_shape = input_shapes
        else:
            atomic_numbers_shape, gaussian_mat_shape = input_shapes
            # If shape is 3-long, one-hot
            if len(atomic_numbers_shape) == 3:
                one_hot_numbers_shape = atomic_numbers_shape
            else:
                one_hot_numbers_shape = atomic_numbers_shape + (self.max_atomic_number + 1,)
        return gaussian_mat_shape + one_hot_numbers_shape[-1:]

    def get_config(self):
        config = {
            'max_atomic_number': self.max_atomic_number,
            'zero_dummy_atoms': self.zero_dummy_atoms
        }
        base_config = super(AtomicNumberBasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AtomRefOffset(Layer):
    """
    Offsets energies by per-atom reference energies.

    Inputs: atomic_numbers
                Either or both in this order:
                    atomic_numbers  (batch, atoms)
                or
                    one_hot_atomic_numbers (batch, atoms, atomic_number)
            atomic_energies  (batch, atoms, energies)
    Output: atomic energies  (batch, atoms, energies)

    Attributes:
        atom_ref (list or np.ndarray): a list or numpy array where
            `atom_ref[atomic_number]` is the reference energy for an
            atom with atomic number `atomic number`.

            This object gets serialized to JSON with the Keras model, so
            be careful not to pass a huge object.
    """
    def __init__(self, atom_ref=None, add_offset=True, **kwargs):
        super(AtomRefOffset, self).__init__(**kwargs)
        self.add_offset = add_offset
        self.atom_ref = atom_ref
        if self.atom_ref is not None:
            self.atom_ref = np.asanyarray(self.atom_ref)
            if len(self.atom_ref.shape) == 1:
                self.atom_ref = np.expand_dims(self.atom_ref, axis=1)

    def call(self, inputs):
        # `atomic_energies` should be of shape (batch, atoms, energies)
        if len(inputs) == 3:
            atomic_numbers, one_hot_atomic_numbers, atomic_energies = inputs
        else:
            atomic_numbers, atomic_energies = inputs
            atomic_numbers_shape = K.int_shape(atomic_numbers)
            # If shape is 3-long, one-hot
            if len(atomic_numbers_shape) == 3:
                one_hot_atomic_numbers = atomic_numbers
                atomic_numbers = K.argmax(one_hot_atomic_numbers,
                                          axis=-1)
            else:
                one_hot_atomic_numbers = K.one_hot(
                    atomic_numbers,
                    self.atom_ref.shape[0]
                )

        if self.atom_ref is not None:
            atom_ref = K.constant(self.atom_ref)
            ref_energies = K.dot(one_hot_atomic_numbers, atom_ref)
            if self.add_offset:
                atomic_energies += ref_energies
            else:
                atomic_energies -= ref_energies

        return atomic_energies

    def compute_output_shape(self, input_shapes):
        atomic_energies = input_shapes[-1]
        return atomic_energies

    def get_config(self):
        atom_ref = self.atom_ref
        if isinstance(atom_ref, (np.ndarray, np.generic)):
            if len(atom_ref.shape) > 0:
                atom_ref = atom_ref.tolist()
            else:
                atom_ref = float(atom_ref)

        config = {
            'atom_ref': atom_ref,
            'add_offset': self.add_offset
        }
        base_config = super(AtomRefOffset, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DummyAtomMasking(Layer):
    """
    Masks dummy atoms (atomic number = 0) with zeros

    Inputs: atomic_numbers
                Either or both in this order:
                    atomic_numbers  (batch, atoms)
                or
                    one_hot_atomic_numbers  (batch, atoms, atomic_number)
            value  (batch, atoms, ...)
    Output: value with zeroes for dummy atoms  (batch, atoms, ...)
    """
    def __init__(self, atom_axes=1, **kwargs):
        super(DummyAtomMasking, self).__init__(**kwargs)
        if isinstance(atom_axes, int):
            atom_axes = [atom_axes]
        elif isinstance(atom_axes, tuple):
            atom_axes = list(atom_axes)
        self.atom_axes = atom_axes

    def call(self, inputs):
        # `value` should be of shape (batch, atoms, ...)
        if len(inputs) == 3:
            atomic_numbers, _, value = inputs
        else:
            atomic_numbers, value = inputs
            atomic_numbers_shape = K.int_shape(atomic_numbers)
            # If shape is 3-long, one-hot
            if len(atomic_numbers_shape) == 3:
                atomic_numbers = K.argmax(atomic_numbers,
                                          axis=-1)

        # Form the mask that removes dummy atoms (atomic number = 0)
        dummy_mask = K.not_equal(atomic_numbers, 0)
        dummy_mask = K.cast(dummy_mask, K.floatx())

        for axis in self.atom_axes:
            mask = dummy_mask
            for _ in range(axis - 1):
                mask = K.expand_dims(mask, axis=1)
            # Add one since K.int_shape does not return batch dim
            while len(K.int_shape(value)) != len(K.int_shape(mask)):
                mask = K.expand_dims(mask, axis=-1)

            # Zeros the energies of dummy atoms
            value *= mask
        return value

    def compute_output_shape(self, input_shapes):
        value = input_shapes[-1]
        return value

    def get_config(self):
        config = {
            'atom_axes': self.atom_axes
        }
        base_config = super(DummyAtomMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
