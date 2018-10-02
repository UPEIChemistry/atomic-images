from atomic_images import np_utils
import numpy as np


def test_one_hot():
    indices = np.array([
        [1, 1],
        [2, 0]
    ])

    assert np.allclose(
        np_utils.one_hot(indices),
        np.array(
            [
                [
                    [0, 1, 0],
                    [0, 1, 0]
                ],
                [
                    [0, 0, 1],
                    [1, 0, 0]
                ]
            ]
        )
    )


def test_distance_matrix():
    r = np.array([
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    ])
    r_ij = np_utils.distance_matrix(r)

    assert np.allclose(
        r_ij,
        [
            [
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            [
                [0.0, 2.0],
                [2.0, 0.0]
            ]
        ]
    )


def test_indices():
    one_hots = np.array(
        [
            [
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 1],
                [1, 0, 0]
            ]
        ]
    )
    assert np.allclose(
        np.array([[1, 1], [2, 0]]),
        np_utils.indices(one_hots)
    )


def test_select_atoms():
    values = np.array([
        [
            [1.0, 2.0, 3.1],
            [1.0, 2.0, 3.1]
        ]
    ])
    atoms = np.array([[1, 2]])

    selected = np_utils.select_atoms(values, atoms, dummy_index=2)
    assert np.allclose(selected,
        np.array([
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.1]
            ]
        ])
    )
