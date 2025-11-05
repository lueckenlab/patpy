import numpy as np

from patpy.tl.sample_representation import _remove_negative_distances


def test_remove_negative_distances():
    non_negative_distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    np.testing.assert_array_equal(_remove_negative_distances(non_negative_distances), non_negative_distances)

    negative_distances = np.array([[0, -1, 2], [1, 0, 3], [2, 3, 0]])
    correct_distances = np.array([[0, 0, 2], [1, 0, 3], [2, 3, 0]])
    np.testing.assert_array_equal(_remove_negative_distances(negative_distances), correct_distances)

    # Test with float type
    np.testing.assert_array_equal(
        _remove_negative_distances(negative_distances.astype(float)), correct_distances.astype(float)
    )
