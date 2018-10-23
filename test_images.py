import argparse

import matplotlib.pyplot as plt
import numpy as np

from atomic_images.np_utils import (cosine_cutoff, distance_matrix,
                                    expand_atomic_numbers, expand_gaussians,
                                    long_tanh_cutoff, one_hot, tanh_cutoff,
                                    zero_dummy_atoms, expand_triangles)


def main(args):
    # Positions and atomic numbers for para-chlorobenzoic acid and methane
    r = np.array([[
        [-0.7320000, 1.1400000, 0.1000000],
        [0.6630000, 1.2080000, -0.0480000],
        [1.3710000, 0.1170000, -0.0510000],
        [0.7450000, -1.1270000, 0.1040000],
        [-0.6590000, -1.2280000, -0.0510000],
        [-1.3900000, -0.1120000, -0.0540000],
        [-2.8241321, -0.1737057, -0.2108237],
        [-3.3809771, -1.2597305, -0.3429256],
        [-3.5519957, 0.9620820, -0.2121630],
        [3.0997221, 0.2023762, -0.2479639],
        [-3.1269647, 1.7910275, -0.1113317],
        [-1.2746546, 1.9902055, 0.3184048],
        [1.1264011, 2.1239486, -0.1544032],
        [1.3038395, -1.9646996, 0.3297916],
        [-1.1094586, -2.1500925, -0.1598872]
    ],
    [
        [0.0000, 0.0000, 0.0000],
        [0.5288, 0.1610, 0.9359],
        [0.2051, 0.8240, -0.6786],
        [0.3345, -0.9314, -0.4496],
        [-1.0685, -0.0537, 0.1921],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]])
    z = np.array([[6, 6, 6, 6, 6, 6, 6, 8, 8, 17, 1, 1, 1, 1, 1],
                  [6, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    cutoff = args.cutoff

    # Converts index based atomic numbers to one-hot
    one_hot_z = one_hot(z)

    # Calculates the distance matrix for the given Cartesian coordinates
    dist_matrix = distance_matrix(r)

    # Expands distances into a Gaussian basis set, returning mu for plotting
    # purposes.
    if args.kernel == 'gaussian':
        gaussians, mu = expand_gaussians(dist_matrix, return_mu=True)
    elif args.kernel == 'triangle':
        gaussians, mu = expand_triangles(dist_matrix, return_mu=True)

    if args.cutoff_type == 'tanh':
        gaussians = tanh_cutoff(dist_matrix, gaussians, cutoff)
    elif args.cutoff_type == 'long_tanh':
        gaussians = long_tanh_cutoff(dist_matrix, gaussians, cutoff)
    elif args.cutoff_type == 'cos':
        gaussians = cosine_cutoff(dist_matrix, gaussians, cutoff)

    # Sets elements of dummy atoms (z = 0) to zero
    interaction_images = zero_dummy_atoms(gaussians, one_hot_z, atom_axes=[1, 2])

    # Expands the Gaussians into a one-hot atomic number dimension
    interaction_images = expand_atomic_numbers(interaction_images, one_hot_z, zero_dummy_atoms=False)
    interaction_images = np.moveaxis(interaction_images, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])

    # Which molecule to visualize
    mol_i = 0

    # Show image before summing
    # Corresponds to an image of an "interaction" between two atoms
    plt.matshow(interaction_images[mol_i, 0, 4])
    plt.xticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('Interaction Image')
    plt.xlabel('Distance (Angstroms)')
    plt.ylabel('Atomic Number')
    # plt.show()
    plt.savefig('single_interaction.svg')

    # Squash one atoms dimension to get atomic images
    # Corresponds to all interactions for single atoms
    atomic_images = np.sum(interaction_images, axis=2)
    plt.matshow(atomic_images[mol_i, 0])
    plt.xticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('Atomic Image')
    plt.xlabel('Distance (Angstroms)')
    plt.ylabel('Atomic Number')
    # plt.show()
    plt.savefig('atomic_image.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff',
                        type=float,
                        default=15.0,
                        help='the cutoff to use')
    parser.add_argument('--cutoff-type',
                        help='the type of cutoff function',
                        choices=['none', 'tanh', 'cos', 'long_tanh'],
                        default='none')
    parser.add_argument('--kernel',
                        default='gaussian',
                        choices=['gaussian', 'triangle'],
                        help='kernel function to use')
    main(parser.parse_args())
