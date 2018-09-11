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
        positions (numpy.ndarray): array of positions (shape: (..., atoms, 3))

    Returns:
        numpy.ndarray: the distance matrix (shape: (..., atoms, atoms))
    """
    v1 = np.expand_dims(positions, axis=-2)
    v2 = np.expand_dims(positions, axis=-3)
    summed = np.sum(np.square(v2 - v1), axis=-1)
    return np.sqrt(summed)


def angle_tensor(positions, deg=False, eps=1e-10):
    """Calculate the tensor containing all possible angles between
    cartesian positions.

    The returned tensor is indexed as follows:
        the first n_points axis is the central point forming the angle
        the latter two n_points axes are the two other points

    The tensor should be symmetric about a permutation of the last two indices.

    Args:
        positions (numpy.ndarray): array of positions (shape: (..., n_points, 3))
        deg (bool): output in degrees if True
        eps (float, optional): a fuzz factor for the divison by magnitudes

    Returns:
        numpy.ndarray: the angle tensor (shape: (..., n_points, n_points, n_points))
    """
    v1 = np.expand_dims(positions, axis=-2)
    v2 = np.expand_dims(positions, axis=-3)

    diff = v2 - v1
    magnitudes = distance_matrix(positions)
    magnitude_products = (
        np.expand_dims(magnitudes, axis=-1)
        * np.expand_dims(magnitudes, axis=-2)
    )
    dot_prod = np.sum(
        np.expand_dims(diff, axis=-2) * np.expand_dims(diff, axis=-3),
        axis=-1
    )
    # Avoids division by zero
    magnitude_products[magnitude_products < eps] = 1.0

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
    if deg:
        angles *= 180 / np.pi

    return angles


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


def expand_gaussians(in_tensor, min_value=-1, max_value=9, width=0.2, spacing=0.2,
                     self_thresh=1e-5, include_self_interactions=True,
                     endpoint=False, return_mu=False):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)  where: u is np.linspace(min_value, max_value, ceil((max_value - min_value) / spacing))
          2 * w^2           w is width

    Args:
        in_tensor (numpy.ndarray): distance matrix (shape: (batch, atoms ,atoms))
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
        np.ndarray: expanded in_tensor (shape: (batch, atoms, atoms, n_gaussians))
        np.ndarray: if return_mu is True, also returns the grid of means
    """
    n_centers = int(np.ceil((max_value - min_value) / spacing))
    in_tensor = np.expand_dims(in_tensor, axis=-1)
    mu = np.linspace(min_value, max_value, n_centers, endpoint=endpoint)

    # Reshape mu
    mu_shape_prefix = tuple([1 for _ in range(len(in_tensor.shape) - 1)])
    mu_eff = np.reshape(mu, mu_shape_prefix + (-1,))

    gamma = -0.5 / (width ** 2)
    gaussians = np.exp(gamma * (np.square(mu_eff - in_tensor)))

    if not include_self_interactions:
        mask = (in_tensor >= self_thresh).astype(float)
        gaussians *= mask

    if return_mu:
        return gaussians, mu
    else:
        return gaussians


def expand_triangles(in_tensor, min_value=-1, max_value=9, width=0.2, spacing=0.2,
                     self_thresh=1e-5, include_self_interactions=True,
                     endpoint=False, return_mu=False):
    """Expand distance matrix into triangles of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

                1.
    max(0, 1 - ---- | x - u |)
                2w

    where: u is np.linspace(min_value, max_value, ceil((max_value - min_value) / spacing))
           w is width

    Args:
        in_tensor (numpy.ndarray): distance matrix (shape: (batch, atoms ,atoms))
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
        np.ndarray: expanded distance matrix (shape: (batch, atoms, atoms, n_triangles))
        np.ndarray: if return_mu is True, also returns the grid of means
    """
    n_centers = int(np.ceil((max_value - min_value) / spacing))
    in_tensor = np.expand_dims(in_tensor, axis=-1)
    mu = np.linspace(min_value, max_value, n_centers, endpoint=endpoint)

    # Reshape mu
    mu_shape_prefix = tuple([1 for _ in range(len(in_tensor.shape) - 1)])
    mu_eff = np.reshape(mu, mu_shape_prefix + (-1,))

    scaling = (0.5 / width)
    triangles = np.maximum(0, 1 - scaling * np.abs(mu_eff - in_tensor))

    if not include_self_interactions:
        mask = (in_tensor >= self_thresh).astype(float)
        triangles *= mask

    if return_mu:
        return triangles, mu
    else:
        return triangles


def cosine_cutoff(d_mat, gaussians, cutoff):
    cos_component = 0.5 * (1 + np.cos(np.pi * d_mat / cutoff))
    cutoffs = np.where(
        d_mat <= cutoff,
        cos_component,
        np.zeros_like(d_mat)
    )
    cutoffs = np.expand_dims(cutoffs, axis=-1)
    return gaussians * cutoffs


def tanh_cutoff(d_mat, gaussians, cutoff):
    norm_factor = 1 / (np.tanh(1) ** 3)
    tanh_component = (np.tanh(1 - d_mat / cutoff)) ** 3
    cutoffs = np.where(
        d_mat <= cutoff,
        norm_factor * tanh_component,
        np.zeros_like(d_mat)
    )
    cutoffs = np.expand_dims(cutoffs, axis=-1)
    return gaussians * cutoffs


def long_tanh_cutoff(d_mat, gaussians, cutoff):
    norm_factor = 1 / (np.tanh(cutoff) ** 3)
    tanh_component = (np.tanh(cutoff - d_mat)) ** 3
    cutoffs = np.where(
        d_mat <= cutoff,
        norm_factor * tanh_component,
        np.zeros_like(d_mat)
    )
    cutoffs = np.expand_dims(cutoffs, axis=-1)
    return gaussians * cutoffs


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
