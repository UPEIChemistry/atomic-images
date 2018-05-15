"""Implement the same functions as the Keras layers, but for numpy.
"""
import numpy as np


def distance_matrix(coordinates):
    """Expand Cartesian coordinates into a distance matrix

    Args:
        coordinates (numpy.ndarray): array of coordinates (shape: (batch, atoms, 3))

    Returns:
        numpy.ndarray: the distance matrix (shape: (batch, atoms, atoms))
    """

    dist = np.expand_dims(coordinates, axis=1) - np.expand_dims(coordinates, axis=2)
    dist = np.square(dist)
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    return dist


def one_hot(indices_values, max_index=None):
    """Convert indices to one-hot vectors

    Args:
        indices_values (numpy.ndarray): indices_values to convert (shape: (batch, atoms))

    Returns:
        numpy.ndarray: the one-hot vectors (shape: (batch, atoms, max_atomic_number + 1))
    """

    if max_index is None:
        eye_mat = np.eye(np.max(indices_values) + 1)
    else:
        eye_mat = np.eye(max_index)
    return eye_mat[indices_values]


def indices(one_hot_values, axis=-1):
    """Convert one-hot vectors to indices

    Args:
        one_hot_values (numpy.ndarray): the one-hot vectors
            (shape: (batch, atoms, max_atomic_number + 1))

    Returns:
        numpy.ndarray: the indices (shape: (batch, atoms))
    """
    return np.argmax(one_hot_values, axis=axis)


def expand_gaussians(dist, min_value=0, max_value=8, width=0.2, spacing=0.1, return_mu=False):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at but not including max_value.

        -(x - u)^2
    exp(----------)  where u is np.arange(min_value, max_value, spacing)
          2 * w^2

    Args:
        dist (numpy.ndarray): distance matrix (shape: (batch, atoms ,atoms))
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of Gaussians
        spacing (float, optional): spacing between Gaussians

    Returns:
        np.ndarray: expanded distance matrix (shape: (batch, atoms, atoms, n_gaussians))
    """

    mu = np.arange(min_value, max_value, spacing)
    exponent = -((np.reshape(mu, tuple(1 for _ in range(len(dist.shape))) + (-1,))
                  - np.expand_dims(dist, axis=-1)) ** 2) / width
    if return_mu:
        return np.exp(exponent), mu
    else:
        return np.exp(exponent)


def expand_atomic_numbers(gaussians, one_hot_z, zero_dummy_atoms=False):
    """Expands Gaussians into one more dimension representing atomic number.

    Args:
        gaussians (numpy.ndarray): gaussian-expanded distance matrix (shape: (batch, atoms, atoms, n_gaussians))
        one_hot_z (numpy.ndarray): one-hot vectors of atomic numbers (shape: (batch, atoms, max_atomic_number + 1))
        zero_dummy_atoms (bool): whether or not to zero out dummy atoms

    Returns:
        numpy.ndarray: gaussian-expanded distance matrix with atomic number information
            (shape: (batch, atoms, atoms, n_gaussians, max_atomic_number + 1))
    """

    gaussians = np.expand_dims(gaussians, axis=-1)
    if zero_dummy_atoms:
        mask = np.eye(one_hot_z.shape[-1])
        mask[0] = 0
        one_hot_z = np.dot(one_hot_z, mask)
    one_hot_z = np.expand_dims(one_hot_z, axis=1)
    one_hot_z = np.expand_dims(one_hot_z, axis=3)
    return gaussians * one_hot_z


def zero_dummy_atoms(values, z, atom_axes=1):
    if isinstance(atom_axes, int):
        atom_axes = [atom_axes]
    elif isinstance(atom_axes, tuple):
        atom_axes = list(atom_axes)

    if len(z.shape) == 3:
        atomic_numbers = np.argmax(z, axis=-1)
    else:
        atomic_numbers = z

    dummy_mask = (atomic_numbers != 0).astype(float)

    # Expand dims as many times as necessary to get 1s in the last
    # dimensions
    for axe in atom_axes:
        mask = dummy_mask
        for _ in range(axe - 1):
            mask = np.expand_dims(mask, axis=1)
        while len(values.shape) != len(mask.shape):
            mask = np.expand_dims(mask, axis=-1)

        # Zeros the energies of dummy atoms
        values *= mask
        del mask

    return values
