from atomic_images import np_layers
import numpy as np


def test_one_hot():
    indices = np.array([
        [1, 1],
        [2, 0]
    ])

    assert np.allclose(
        np_layers.OneHot(np.max(indices))(indices),
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
    r_ij = np_layers.DistanceMatrix()(r)

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


def test_select_atoms():
    values = np.array([
        [
            [1.0, 2.0, 3.1],
            [1.0, 2.0, 3.1]
        ]
    ])
    atoms = np.array([[1, 2]])
    one_hot = np_layers.OneHot(np.max(atoms))(atoms)

    selected = np_layers.SelectAtoms(atom_index=2)([one_hot, values])
    assert np.allclose(selected,
        np.array([
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.1]
            ]
        ])
    )
