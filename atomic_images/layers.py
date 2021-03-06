import numpy as np
from keras import backend as K
from keras.layers import Layer, Lambda

from atomic_images import keras_utils


#
# Basic math functions
#
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

    Input: coordinates (..., atoms, 3)
    Output: distance matrix (..., atoms, atoms)
    """
    def call(self, positions):
        # `positions` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        v1 = K.expand_dims(positions, axis=-2)
        v2 = K.expand_dims(positions, axis=-3)

        sum_squares = K.sum(K.square(v2 - v1), axis=-1)
        sqrt = K.sqrt(sum_squares + K.epsilon())
        K.switch(sqrt >= K.epsilon(), sqrt, K.zeros_like(sqrt))
        return sqrt

    def compute_output_shape(self, positions_shape):
        return positions_shape[0:-2] + (positions_shape[-2], positions_shape[-2])


class AngleTensor(Layer):
    """
    Angle tensor layer

    Calculate the tensor containing all possible angles between
    cartesian positions (between [0, 180] degrees)

    The returned tensor is indexed as follows:
        the first n_points axis is the central point forming the angle
        the latter two n_points axes are the two other points

    The tensor should be symmetric about a permutation of the last two indices.

    Input: coordinates (..., atoms, 3)
    Output: distance matrix (..., atoms, atoms)
    """
    def __init__(self, deg=False, eps=1e-10):
        super(AngleTensor, self).__init__()
        self.eps = eps
        self.deg = deg

        if K.backend() == 'tensorflow':
            import tensorflow as tf
            self.arccos = Lambda(tf.acos, output_shape=lambda x: x)
        else:
            self.arccos = None

    def call(self, inputs):
        # `positions` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        positions, dist_matrix = inputs

        n_positions = K.int_shape(positions)[-2]

        v1 = K.expand_dims(positions, axis=-2)
        v2 = K.expand_dims(positions, axis=-3)

        diff = v2 - v1
        magnitudes = dist_matrix
        magnitude_products = (
            K.expand_dims(magnitudes, axis=-1)
            * K.expand_dims(magnitudes, axis=-2)
        )
        dot_prod = K.sum(
            K.expand_dims(diff, axis=-2) * K.expand_dims(diff, axis=-3),
            axis=-1
        )
        # Avoids division by zero
        magnitude_products = K.switch(
            magnitude_products < self.eps,
            K.ones_like(magnitude_products),
            magnitude_products
        )

        # Calculate the angles
        if self.arccos is None:
            raise NotImplementedError('AngleTensor not implemented for '
                                      'backends other than TensorFlow '
                                      'at this time')
        angles = self.arccos(dot_prod / magnitude_products)

        # Zero invalid values
        mask1 = K.reshape(1 - K.eye(n_positions), (1, n_positions, n_positions))
        mask2 = K.reshape(1 - K.eye(n_positions), (n_positions, 1, n_positions))
        mask3 = K.reshape(1 - K.eye(n_positions), (n_positions, n_positions, 1))
        mask = mask1 * mask2 * mask3

        angles = mask * angles

        # Convert to degrees if asked
        if self.deg:
            angles *= 180 / np.pi

        return angles

    def compute_output_shape(self, input_shapes):
        positions_shape, _ = input_shapes
        return positions_shape[0:-2] + (positions_shape[-2], positions_shape[-2], positions_shape[-2])

    def get_config(self):
        config = {
            'deg': self.deg,
            'eps': self.eps
        }
        base_config = super(AngleTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#
# Kernel functions
#
class KernelBasis(Layer):
    """Expand tensor using kernel of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

    Input: tensor (batch, atoms, [atoms, [atoms...])
    Output: tensor expanded into kernel basis set (batch, atoms, [atoms, [atoms...]], n_gaussians)

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of kernel functions
        spacing (float, optional): spacing between kernel functions
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
    """
    def __init__(self, min_value=-1, max_value=9, width=0.2,
                 spacing=0.2, self_thresh=1e-5, include_self_interactions=True,
                 endpoint=False, **kwargs):
        super(KernelBasis, self).__init__(**kwargs)
        self._n_centers = int(np.ceil((max_value - min_value) / spacing))
        self.min_value = min_value
        self.max_value = max_value
        self.spacing = spacing
        self.width = width
        self.self_thresh = self_thresh
        self.include_self_interactions = include_self_interactions
        self.endpoint = endpoint

    def call(self, in_tensor):
        in_tensor = K.expand_dims(in_tensor, -1)
        mu = keras_utils.linspace(self.min_value, self.max_value, self._n_centers,
                                  endpoint=self.endpoint)

        mu_prefix_shape = tuple([1 for _ in range(len(K.int_shape(in_tensor)) - 1)])
        mu = K.reshape(mu, mu_prefix_shape + (-1,))
        values = self.kernel_func(in_tensor, mu)

        if not self.include_self_interactions:
            mask = K.cast(in_tensor >= self.self_thresh, K.floatx())
            values *= mask

        return values

    def kernel_func(self, inputs, centres):
        raise NotImplementedError

    def compute_output_shape(self, in_tensor_shape):
        return in_tensor_shape + (self._n_centers,)

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
        base_config = super(KernelBasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianBasis(KernelBasis):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)
        2 * w^2

    where: u is linspace(min_value, max_value, ceil((max_value - min_value) / width))
           w is width

    Input: distance_matrix (batch, atoms, atoms)
    Output: distance_matrix expanded into Gaussian basis set (batch, atoms, atoms, n_centres)

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
    def kernel_func(self, inputs, centres):
        gamma = -0.5 / (self.width  ** 2)
        return K.exp(gamma * K.square(inputs - centres))


class TriangularBasis(KernelBasis):
    """Expand distance matrix into triangles of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

          1
    1 - -----|x - u|
         2w

    where: u is linspace(min_value, max_value, ceil((max_value - min_value) / width))
           w is width

    Input: distance_matrix (batch, atoms, atoms)
    Output: distance_matrix expanded into Triangular basis set (batch, atoms, atoms, n_centres)

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of triangles
        spacing (float, optional): spacing between triangles
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
                (batch, atoms, atoms, n_triangles)
    """
    def kernel_func(self, inputs, centres):
        gamma = -0.5 / self.width
        return 1 + gamma * K.abs(inputs - centres)


#
# Cutoff functions
#
class CutoffLayer(Layer):
    """Base layer for cutoff functions.

    Applies a cutoff function to the expanded distance matrix

    Inputs:
        distance_matrix (batch, atoms, atoms)
        basis_functions (batch, atoms, atoms, n_centres)
    Output: basis_functions with cutoff function multiplied (batch, atoms, atoms, n_centres)
    """
    def __init__(self, cutoff, **kwargs):
        super(CutoffLayer, self).__init__(**kwargs)
        self.cutoff = cutoff

    def call(self, inputs):
        distance_matrix, basis_functions = inputs

        cutoffs = self.cutoff_function(distance_matrix)
        cutoffs = K.expand_dims(cutoffs, axis=-1)

        return basis_functions * cutoffs

    def cutoff_function(self, distance_matrix):
        """Function responsible for the cutoff. It should also return zeros
        for anything greater than the cutoff.

        Args:
            distance_matrix (Tensor): the distance matrix tensor
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shapes):
        _, basis_functions_shape = input_shapes
        return basis_functions_shape

    def get_config(self):
        config = {
            'cutoff': self.cutoff
        }
        base_config = super(CutoffLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineCutoff(CutoffLayer):
    """The cosine cutoff originally proposed by Behler et al. for ACSFs.
    """
    def cutoff_function(self, distance_matrix):
        cos_component = 0.5 * (1 + K.cos(np.pi * distance_matrix / self.cutoff))
        return K.switch(
            distance_matrix <= self.cutoff,
            cos_component,
            K.zeros_like(distance_matrix)
        )


class TanhCutoff(CutoffLayer):
    """Alternate tanh^3 cutoff function mentioned in some of the ACSF papers.
    """
    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (K.tanh(1.0) ** 3)
        tanh_component = (K.tanh(1.0 - (distance_matrix / self.cutoff))) ** 3
        return K.switch(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            K.zeros_like(distance_matrix)
        )


class LongTanhCutoff(CutoffLayer):
    """Custom tanh cutoff function that keeps symmetry functions relatively unscaled
    longer than the previously proposed tanh function
    """
    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (K.tanh(float(self.cutoff)) ** 3)
        tanh_component = (K.tanh(self.cutoff - distance_matrix)) ** 3
        return K.switch(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            K.zeros_like(distance_matrix)
        )


#
# Atom-related functions
#
class AtomicNumberBasis(Layer):
    """Expands Gaussian matrix into the one-hot atomic numbers basis

    Inputs:
        one_hot_numbers  (batch, atoms, max_atomic_number + 1)
        gaussians_matrix  (batch, atoms, atoms, n_gaussians)
    Output:
        gaussians_atom_matrix  (batch, atoms, atoms, n_gaussians, max_atomic_number + 1)
    """
    def __init__(self, zero_dummy_atoms=False, **kwargs):
        kwargs.pop('max_atomic_number', None)  # Backward compatibility
        super(AtomicNumberBasis, self).__init__(**kwargs)
        self.zero_dummy_atoms = zero_dummy_atoms

    def call(self, inputs):
        one_hot_numbers, gaussian_mat = inputs

        gaussian_mat = K.expand_dims(gaussian_mat, axis=-1)
        if self.zero_dummy_atoms:
            mask = K.eye(one_hot_numbers.shape[-1], dtype=K.floatx())
            mask[0] = 0
            one_hot_numbers = K.dot(one_hot_numbers, mask)
        one_hot_numbers = K.expand_dims(one_hot_numbers, axis=1)
        one_hot_numbers = K.expand_dims(one_hot_numbers, axis=3)
        return gaussian_mat * one_hot_numbers

    def compute_output_shape(self, input_shapes):
        one_hot_numbers_shape, gaussian_mat_shape = input_shapes
        return gaussian_mat_shape + one_hot_numbers_shape[-1:]

    def get_config(self):
        config = {
            'zero_dummy_atoms': self.zero_dummy_atoms
        }
        base_config = super(AtomicNumberBasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#
# Normalization-related layers
#
class Unstandardization(Layer):
    """
    Offsets energies by mean and standard deviation (optionally, per-atom)

    `mu` and `sigma` both follow the following:
        If the value is a scalar, apply it equally to all properties
        and all types of atoms

        If the value is a vector, each component corresponds to an
        output property. It is expanded into a matrix where the
        first axis shape is 1. It then follows the matrix rules.

        If the value is a matrix, rows correspond to types of atoms and
        columns correspond to properties.

            If there is only one row, then the row vector applies to every
            type of atom equally.

            If there is one column, then the scalars are applied to every
            property equally.

            If there is a single scalar, then it is treated as a scalar.

    Inputs: the inputs to this layer depend on whether or not mu and sigma
            are given as a single scalar or per atom type.

        If scalar:
            atomic_props  (batch, atoms, energies)
        If per type:
            one_hot_atomic_numbers (batch, atoms, atomic_number)
            atomic_props  (batch, atoms, energies)
    Output: atomic_props  (batch, atoms, energies)

    Attributes:
        mu (float, list, or np.ndarray): the mean values by which
            to offset the inputs to this layer
        sigma (float, list, or np.ndarray): the standard deviation
            values by which to scale the inputs to this layer
    """
    def __init__(self, mu, sigma, trainable=False, per_type=None, use_float64=False, **kwargs):
        super(Unstandardization, self).__init__(trainable=trainable, **kwargs)
        self.init_mu = mu
        self.init_sigma = sigma
        self.use_float64 = use_float64
        self.dtype = 'float64' if use_float64 else 'float32'

        self.mu = np.asanyarray(self.init_mu)
        self.sigma = np.asanyarray(self.init_sigma)

        self.per_type = len(self.mu.shape) > 0 or per_type

    @staticmethod
    def expand_ones_to_shape(arr, shape):
        if len(arr.shape) == 0:
            arr = arr.reshape((1 ,1))
        if 1 in arr.shape:
            tile_shape = tuple(shape[i] if arr.shape[i] == 1 else 1
                               for i in range(len(shape)))
            arr = np.tile(arr, tile_shape)
        if arr.shape != shape:
            raise ValueError('the arrays were not of the right shape: '
                             'expected %s but was %s' % (shape, arr.shape))
        return arr

    def build(self, input_shapes):
        # If mu is given as a vector, assume it applies to all atoms
        if len(self.mu.shape) == 1:
            self.mu = np.expand_dims(self.mu, axis=0)
        if len(self.sigma.shape) == 1:
            self.sigma = np.expand_dims(self.sigma, axis=0)

        if self.per_type:
            one_hot_atomic_numbers, atomic_props = input_shapes
            w_shape = (one_hot_atomic_numbers[-1], atomic_props[-1])

            self.mu = self.expand_ones_to_shape(self.mu, w_shape)
            self.sigma = self.expand_ones_to_shape(self.sigma, w_shape)
        else:
            w_shape = self.mu.shape

        self.mu = self.add_weight(
            name='mu',
            shape=w_shape,
            initializer=lambda x: self.mu,
            dtype=self.dtype
        )
        self.sigma = self.add_weight(
            name='sigma',
            shape=w_shape,
            initializer=lambda x: self.sigma,
            dtype=self.dtype
        )
        super(Unstandardization, self).build(input_shapes)

    def call(self, inputs):
        # `atomic_props` should be of shape (batch, atoms, energies)

        # If mu and sigma are given per atom type, need atomic numbers
        # to know how to apply them. Otherwise, just energies is enough.
        if self.per_type or isinstance(inputs, (list, tuple)):
            one_hot_atomic_numbers, atomic_props = inputs
        else:
            atomic_props = inputs
        atomic_props = K.cast(atomic_props, self.dtype)
        one_hot_atomic_numbers = K.cast(one_hot_atomic_numbers, self.dtype)

        if self.per_type:
            atomic_props *= K.dot(one_hot_atomic_numbers, self.sigma)
            atomic_props += K.dot(one_hot_atomic_numbers, self.mu)
        else:
            atomic_props *= self.sigma
            atomic_props += self.mu

        return atomic_props

    def compute_output_shape(self, input_shapes):
        if self.per_type or isinstance(input_shapes, list):
            atomic_props = input_shapes[-1]
        else:
            atomic_props = input_shapes
        return atomic_props

    def get_config(self):
        mu = self.init_mu
        if isinstance(mu, (np.ndarray, np.generic)):
            if len(mu.shape) > 0:
                mu = mu.tolist()
            else:
                mu = float(mu)

        sigma = self.init_sigma
        if isinstance(sigma, (np.ndarray, np.generic)):
            if len(sigma.shape) > 0:
                sigma = sigma.tolist()
            else:
                sigma = float(sigma)

        config = {
            'mu': mu,
            'sigma': sigma,
            'per_type': self.per_type,
            'use_float64': self.use_float64
        }
        base_config = super(Unstandardization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AtomRefOffset(Unstandardization):
    """
    Offsets energies by per-atom reference properties.
    Simpler case of Unstandardization.

    Inputs: one_hot_atomic_numbers (batch, atoms, atomic_number)
            atomic_props  (batch, atoms, energies)
    Output: atomic_props  (batch, atoms, energies)

    Attributes:
        atom_ref (list or np.ndarray): atom references of
            shape (atomic_number, n_props)
    """
    def __init__(self, atom_ref=None, add_offset=True, **kwargs):
        self.add_offset = add_offset
        self.atom_ref = atom_ref

        if not self.add_offset:
            self.atom_ref *= -1

        kwargs.pop('per_type', None)
        super(AtomRefOffset, self).__init__(
            mu=atom_ref,
            sigma=kwargs.pop('sigma', 1.0),
            per_type=True,
            **kwargs
        )

    def get_config(self):
        base_config = super(AtomRefOffset, self).get_config()
        config = {
            'atom_ref': base_config.pop('mu'),
            'add_offset': self.add_offset
        }
        return dict(list(base_config.items()) + list(config.items()))


#
# Dummy atom-related layers
#
class DummyAtomMasking(Layer):
    """
    Masks dummy atoms (atomic number = 0 by default) with zeros

    Inputs: atomic_numbers
                Either or both in this order:
                    atomic_numbers  (batch, atoms)
                or
                    one_hot_atomic_numbers  (batch, atoms, atomic_number)
            value  (batch, atoms, ...)
    Output: value with zeroes for dummy atoms  (batch, atoms, ...)

    Args:
        atom_axes (int or iterable of int): axes to which to apply
            the masking

    Keyword Args:
        dummy_index (int): the index to mask (default: 0)
        invert_mask (bool): if True, zeroes all but the desired index rather
            than zeroeing the desired index
    """
    def __init__(self, atom_axes=1, **kwargs):
        self.invert_mask = kwargs.pop('invert_mask', False)
        self.dummy_index = kwargs.pop('dummy_index', 0)
        super(DummyAtomMasking, self).__init__(**kwargs)
        if isinstance(atom_axes, int):
            atom_axes = [atom_axes]
        elif isinstance(atom_axes, tuple):
            atom_axes = list(atom_axes)
        self.atom_axes = atom_axes

    def call(self, inputs):
        # `value` should be of shape (batch, atoms, ...)
        one_hot_atomic_numbers, value = inputs
        atomic_numbers = K.argmax(one_hot_atomic_numbers,
                                  axis=-1)

        # Form the mask that removes dummy atoms (atomic number = dummy_index)
        if self.invert_mask:
            selection_mask = K.equal(atomic_numbers, self.dummy_index)
        else:
            selection_mask = K.not_equal(atomic_numbers, self.dummy_index)
        selection_mask = K.cast(selection_mask, value.dtype)

        for axis in self.atom_axes:
            mask = selection_mask
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
            'atom_axes': self.atom_axes,
            'invert_mask': self.invert_mask,
            'dummy_index': self.dummy_index
        }
        base_config = super(DummyAtomMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelectAtoms(DummyAtomMasking):
    """Selects atoms of a certain type.

    Convenience function around DummyAtomMasking to have it
    zero all other atoms rather than just the dummy index.
    """
    def __init__(self, *args, **kwargs):
        atom_index = kwargs.pop('atom_index', kwargs.pop('dummy_index', 0))
        kwargs['invert_mask'] = True
        super(SelectAtoms, self).__init__(
            dummy_index=atom_index,
            *args, **kwargs
        )
