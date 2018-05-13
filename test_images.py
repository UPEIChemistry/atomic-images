import numpy as np
import matplotlib.pyplot as plt


def distance_matrix(coordinates):
    dist = np.expand_dims(coordinates, axis=1) - np.expand_dims(coordinates, axis=2)
    dist = np.square(dist)
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    return dist


def one_hot(indices):
    return np.eye(np.max(indices) + 1)[indices]


def indices(one_hot):
    return np.argmax(one_hot, axis=-1)


def expand_gaussians(dist, width=0.2, spacing=0.1, max_value=8, min_value=0):
    mu = np.arange(min_value, max_value, spacing)
    exponent = -((np.reshape(mu, tuple(1 for _ in range(len(dist.shape))) + (-1,))
                  - np.expand_dims(dist, axis=-1)) ** 2) / width
    return np.exp(exponent)


def expand_atomic_numbers(gaussians, one_hot_z):
    gaussians = np.expand_dims(gaussians, axis=-1)
    z = one_hot_z.reshape((gaussians.shape[0], 1,
                           gaussians.shape[1], 1, one_hot_z.shape[-1]))
    return gaussians * z


def main():
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
    ]])
    z = np.array([[6, 6, 6, 6, 6, 6, 6, 8, 8, 17, 1, 1, 1, 1, 1]])
    one_hot_z = one_hot(z)

    dist = distance_matrix(r)
    print(dist)

    gaussians = expand_gaussians(dist)

    atomic_images = expand_atomic_numbers(gaussians, one_hot_z)
    atomic_images = np.sum(atomic_images, axis=2)
    flattened_atomic_images = np.concatenate(tuple(atomic_images[0, i] for i in range(atomic_images.shape[1])), axis=-1)

    # Display the carbon atom
    print(flattened_atomic_images.shape)
    plt.imshow(flattened_atomic_images)
    plt.xticks(np.arange(0, flattened_atomic_images.shape[-1], atomic_images.shape[-1]))
    plt.grid(True, which='major', axis='x')
    plt.show()


if __name__ == '__main__':
    main()
