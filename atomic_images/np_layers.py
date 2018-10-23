"""A module containing numpy implementations of the Keras layers
"""
import numpy as np


class Layer(object):
    """Dummy layer base class for numpy layers
    """
    def __init__(self, *args, **kwargs):
        self._is_built = False

    def build(self, inputs):
        pass

    def call(self, inputs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if not self._is_built:
            self.build(*args, **kwargs)
            self._is_built = True
        return self.call(*args, **kwargs)

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
        eye_mat = np.eye(self.max_atomic_number + 1)
        return eye_mat[atomic_numbers]


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
        v1 = np.expand_dims(positions, axis=-2)
        v2 = np.expand_dims(positions, axis=-3)

        sum_squares = np.sum(np.square(v2 - v1), axis=-1)
        return np.sqrt(sum_squares)


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

    def call(self, inputs):
        # `positions` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        positions, magnitudes = inputs

        v1 = np.expand_dims(positions, axis=-2)
        v2 = np.expand_dims(positions, axis=-3)

        diff = v2 - v1
        magnitude_products = (
            np.expand_dims(magnitudes, axis=-1)
            * np.expand_dims(magnitudes, axis=-2)
        )
        dot_prod = np.sum(
            np.expand_dims(diff, axis=-2) * np.expand_dims(diff, axis=-3),
            axis=-1
        )
        # Avoids division by zero
        magnitude_products[magnitude_products < self.eps] = 1.0

        # Calculate the angles
        angles = np.arccos(dot_prod / magnitude_products)

        # Zero invalid values
        n_positions = positions.shape[-2]
        mask1 = np.reshape(1 - np.eye(n_positions, dtype=int), (1, n_positions, n_positions))
        mask2 = np.reshape(1 - np.eye(n_positions, dtype=int), (n_positions, 1, n_positions))
        mask3 = np.reshape(1 - np.eye(n_positions, dtype=int), (n_positions, n_positions, 1))
        mask = mask1 * mask2 * mask3

        angles *= mask

        # Convert to degrees if asked
        if self.deg:
            angles *= 180 / np.pi

        return angles


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
        self.mu = None

    def call(self, in_tensor):
        in_tensor = np.expand_dims(in_tensor, -1)
        self.mu = np.linspace(self.min_value, self.max_value, self._n_centers,
                              endpoint=self.endpoint)

        mu_prefix_shape = tuple([1 for _ in range(len(in_tensor.shape) - 1)])
        mu = np.reshape(self.mu, mu_prefix_shape + (-1,))
        values = self.kernel_func(in_tensor, mu)

        if not self.include_self_interactions:
            mask = (in_tensor >= self.self_thresh).astype(in_tensor.dtype)
            values *= mask

        return values

    def kernel_func(self, inputs, centres):
        raise NotImplementedError


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
        return np.exp(gamma * np.square(inputs - centres))


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
        return 1 + gamma * np.abs(inputs - centres)


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
        cutoffs = np.expand_dims(cutoffs, axis=-1)

        return basis_functions * cutoffs

    def cutoff_function(self, distance_matrix):
        """Function responsible for the cutoff. It should also return zeros
        for anything greater than the cutoff.

        Args:
            distance_matrix (Tensor): the distance matrix tensor
        """
        raise NotImplementedError


class CosineCutoff(CutoffLayer):
    """The cosine cutoff originally proposed by Behler et al. for ACSFs.
    """
    def cutoff_function(self, distance_matrix):
        cos_component = 0.5 * (1 + np.cos(np.pi * distance_matrix / self.cutoff))
        return np.where(
            distance_matrix <= self.cutoff,
            cos_component,
            np.zeros_like(distance_matrix)
        )


class TanhCutoff(CutoffLayer):
    """Alternate tanh^3 cutoff function mentioned in some of the ACSF papers.
    """
    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (np.tanh(1.0) ** 3)
        tanh_component = (np.tanh(1.0 - (distance_matrix / self.cutoff))) ** 3
        return np.where(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            np.zeros_like(distance_matrix)
        )


class LongTanhCutoff(CutoffLayer):
    """Custom tanh cutoff function that keeps symmetry functions relatively unscaled
    longer than the previously proposed tanh function
    """
    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (np.tanh(float(self.cutoff)) ** 3)
        tanh_component = (np.tanh(self.cutoff - distance_matrix)) ** 3
        return np.where(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            np.zeros_like(distance_matrix)
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
        super(AtomicNumberBasis, self).__init__(**kwargs)
        self.zero_dummy_atoms = zero_dummy_atoms

    def call(self, inputs):
        one_hot_numbers, gaussian_mat = inputs

        gaussian_mat = np.expand_dims(gaussian_mat, axis=-1)
        if self.zero_dummy_atoms:
            mask = np.eye(one_hot_numbers.shape[-1], dtype=gaussian_mat.dtype)
            mask[0] = 0
            one_hot_numbers = np.tensordot(one_hot_numbers, mask, axes=1)
        one_hot_numbers = np.expand_dims(one_hot_numbers, axis=1)
        one_hot_numbers = np.expand_dims(one_hot_numbers, axis=3)
        return gaussian_mat * one_hot_numbers


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
    def __init__(self, mu, sigma, trainable=False, per_type=None, **kwargs):
        super(Unstandardization, self).__init__(trainable=trainable, **kwargs)
        self.init_mu = mu
        self.init_sigma = sigma

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

    def call(self, inputs):
        # `atomic_props` should be of shape (batch, atoms, energies)

        # If mu and sigma are given per atom type, need atomic numbers
        # to know how to apply them. Otherwise, just energies is enough.
        if self.per_type or isinstance(inputs, (list, tuple)):
            one_hot_atomic_numbers, atomic_props = inputs
        else:
            atomic_props = inputs

        if self.per_type:
            atomic_props *= np.tensordot(one_hot_atomic_numbers, self.sigma, axes=1)
            atomic_props += np.tensordot(one_hot_atomic_numbers, self.mu, axes=1)
        else:
            atomic_props *= self.sigma
            atomic_props += self.mu

        return atomic_props



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
        atomic_numbers = np.argmax(one_hot_atomic_numbers,
                                   axis=-1)

        # Form the mask that removes dummy atoms (atomic number = dummy_index)
        if self.invert_mask:
            selection_mask = atomic_numbers == self.dummy_index
        else:
            selection_mask = atomic_numbers != self.dummy_index
        selection_mask = selection_mask.astype(value.dtype)

        for axis in self.atom_axes:
            mask = selection_mask
            for _ in range(axis - 1):
                mask = np.expand_dims(mask, axis=1)
            # Add one since K.int_shape does not return batch dim
            while len(value.shape) != len(mask.shape):
                mask = np.expand_dims(mask, axis=-1)

            # Zeros the energies of dummy atoms
            value *= mask
        return value


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
