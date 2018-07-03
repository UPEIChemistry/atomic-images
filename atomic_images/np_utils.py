"""Implement the same functions as the Keras layers, but for numpy.
"""
import numpy as np


def one_hot(indices_values, max_index=None):
    """Convert indices to one-hot vectors

    Args:
        indices_values (numpy.ndarray): indices_values to convert (shape: (batch, atoms))
        max_index (int): the max index (if not given, uses the maximum of indices)

    Returns:
        numpy.ndarray: the one-hot vectors (shape: (batch, atoms, max_index + 1))
    """

    if max_index is None:
        eye_mat = np.eye(np.max(indices_values) + 1)
    else:
        eye_mat = np.eye(max_index + 1)
    return eye_mat[indices_values]


def distance_matrix(positions):
    """Expand Cartesian coordinates into a distance matrix

    Args:
        positions (numpy.ndarray): array of positions (shape: (batch, atoms, 3))

    Returns:
        numpy.ndarray: the distance matrix (shape: (batch, atoms, atoms))
    """
    v1 = np.expand_dims(positions, axis=2)
    v2 = np.expand_dims(positions, axis=1)

    diff = v2 - v1
    sq_diff = np.square(diff)
    summed = np.sum(sq_diff, axis=-1)
    return np.sqrt(summed)


def indices(one_hot_values, axis=-1):
    """Convert one-hot vectors to indices

    Args:
        one_hot_values (numpy.ndarray): the one-hot vectors
            (shape: (batch, atoms, max_atomic_number + 1))
        axis (int): the axis to do argmax along (by default,
            the last)

    Returns:
        numpy.ndarray: the indices (shape: (batch, atoms))
    """
    return np.argmax(one_hot_values, axis=axis)


def expand_gaussians(dist, min_value=-1, max_value=9, width=2.0, spacing=0.2,
                     self_thresh=1e-5, include_self_interactions=True,
                     endpoint=False, return_mu=False):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)  where: u is np.linspace(min_value, max_value, ceil((max_value - min_value) / spacing))
        2 * (ws)^2          w is width
                            s is spacing

    Args:
        dist (numpy.ndarray): distance matrix (shape: (batch, atoms ,atoms))
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of Gaussians
        spacing (float, optional): spacing between Gaussians
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
        return_mu (bool, optional): whether or not to return the grid of means

    Returns:
        np.ndarray: expanded distance matrix (shape: (batch, atoms, atoms, n_gaussians))
        np.ndarray: if return_mu is True, also returns the grid of means
    """
    n_centers = int(np.ceil((max_value - min_value) / spacing))
    dist = np.expand_dims(dist, axis=-1)
    mu = np.linspace(min_value, max_value, n_centers, endpoint=endpoint)

    # Reshape mu
    mu_eff = np.reshape(mu, (1, 1, 1, -1))

    gamma = -0.5 / ((width * spacing) ** 2)
    gaussians = np.exp(gamma * (np.square(mu_eff - dist)))

    if not include_self_interactions:
        mask = (dist >= self_thresh).astype(float)
        gaussians *= mask

    if return_mu:
        return gaussians, mu
    else:
        return gaussians


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
    """Zeros values where atomic numbers are zero along one or more
    axes

    Args:
        values (numpy.ndarray): values to mask
        z (numpy.ndarray): atomic numbers (as indices or one-hot)
        atom_axes (int or list, optional): one or more axes to which to
            apply masking

    Returns:
        numpy.ndarray: the values, with all values for dummy atoms set to zero
    """

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
    for axis in atom_axes:
        mask = dummy_mask
        for _ in range(axis - 1):
            mask = np.expand_dims(mask, axis=1)
        while len(values.shape) != len(mask.shape):
            mask = np.expand_dims(mask, axis=-1)

        # Zeros the values of dummy atoms
        values *= mask
        del mask

    return values
