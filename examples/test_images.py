import argparse

import matplotlib.pyplot as plt
import numpy as np

from atomic_images.np_layers import (CosineCutoff, DistanceMatrix,
                                    AtomicNumberBasis, GaussianBasis,
                                    LongTanhCutoff, OneHot, TanhCutoff,
                                    DummyAtomMasking, TriangularBasis)


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
    one_hot_z = OneHot(np.max(z))(z)

    # Calculates the distance matrix for the given Cartesian coordinates
    dist_matrix = DistanceMatrix()(r)

    # Expands distances into a Gaussian basis set, returning mu for plotting
    # purposes.
    if args.kernel == 'gaussian':
        layer = GaussianBasis()
        gaussians = layer(dist_matrix)
    elif args.kernel == 'triangle':
        layer = TriangularBasis()
        gaussians = layer(dist_matrix)
    mu = layer.mu
    del layer

    if args.cutoff_type == 'tanh':
        gaussians = TanhCutoff(cutoff)([dist_matrix, gaussians])
    elif args.cutoff_type == 'long_tanh':
        gaussians = LongTanhCutoff(cutoff)([dist_matrix, gaussians])
    elif args.cutoff_type == 'cos':
        gaussians = CosineCutoff(cutoff)([dist_matrix, gaussians])

    # Sets elements of dummy atoms (z = 0) to zero
    interaction_images = DummyAtomMasking(atom_axes=[1, 2])([one_hot_z, gaussians])

    # Expands the Gaussians into a one-hot atomic number dimension
    interaction_images = AtomicNumberBasis()([one_hot_z, interaction_images])

    # Which molecule to visualize
    mol_i = 0

    # Show image before summing
    # Corresponds to an image of an "interaction" between two atoms
    plt.imshow(interaction_images[mol_i, 0, 0])
    plt.yticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('Interaction Image')
    plt.ylabel('Distance (Angstroms)')
    plt.xlabel('Atomic Number')
    plt.show()

    flattened_interaction_images = np.concatenate(tuple(interaction_images[mol_i, 0, i] for i in range(interaction_images.shape[2])), axis=-1)
    plt.imshow(flattened_interaction_images)
    plt.xticks(np.arange(0, flattened_interaction_images.shape[-1], interaction_images.shape[-1]))
    plt.yticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('All Interaction Images')
    plt.ylabel('Distance (Angstroms)')
    plt.xlabel('Atomic Number')
    plt.grid(True, which='major', axis='x')
    plt.show()

    # Squash one atoms dimension to get atomic images
    # Corresponds to all interactions for single atoms
    atomic_images = np.sum(interaction_images, axis=2)
    plt.imshow(atomic_images[mol_i, 0])
    plt.yticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('Atomic Image')
    plt.ylabel('Distance (Angstroms)')
    plt.xlabel('Atomic Number')
    plt.show()

    # Show all atoms for para-chlorobenzoic acid
    flattened_atomic_images = np.concatenate(tuple(atomic_images[mol_i, i] for i in range(atomic_images.shape[1])), axis=-1)
    plt.imshow(flattened_atomic_images)
    plt.xticks(np.arange(0, flattened_atomic_images.shape[-1], atomic_images.shape[-1]))
    plt.yticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('All Atomic Images')
    plt.ylabel('Distance (Angstroms)')
    plt.xlabel('Atomic Number')
    plt.grid(True, which='major', axis='x')
    plt.show()

    # Squash the last atoms dimension to get molecular images
    # Corresponds to all interactions for every atom in the molecule
    molecular_images = np.sum(atomic_images, axis=1)
    plt.imshow(molecular_images[mol_i])
    plt.yticks(np.arange(len(mu))[::10], mu[::10])
    plt.title('Molecular Image')
    plt.ylabel('Distance (Angstroms)')
    plt.xlabel('Atomic Number')
    plt.show()


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
