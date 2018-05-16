import numpy as np

import matplotlib.pyplot as plt
from atomic_images.np_utils import (distance_matrix, expand_atomic_numbers,
                                    expand_gaussians, one_hot, zero_dummy_atoms)


def main():
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
    one_hot_z = one_hot(z)

    dist_matrix = distance_matrix(r)
    gaussians, mu = expand_gaussians(dist_matrix, return_mu=True)
    interaction_images = zero_dummy_atoms(gaussians, one_hot_z, atom_axes=[1, 2])
    interaction_images = expand_atomic_numbers(interaction_images, one_hot_z, zero_dummy_atoms=False)

    # Which molecule to visualize
    mol_i = 1

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
    main()
