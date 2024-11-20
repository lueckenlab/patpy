import numpy as np
import pandas as pd
import scipy.sparse as sp

from patient_representation.tl.basic import (
    calculate_average_without_nans,
    create_colormap,
    describe_metadata,
    make_matrix_symmetric,
)


def test_make_matrix_symmetric():
    toy_matrix = np.array(
        [
            [0, 1, np.nan],
            [1, 0, 2],
            [np.nan, 2, 0],
        ]
    )
    # dense matrix
    sym_matrix = make_matrix_symmetric(toy_matrix)
    assert np.allclose(sym_matrix, sym_matrix.T, equal_nan=True), "The output matrix should be symmetric."

    # sparse matrix
    sparse_matrix = sp.csr_matrix(toy_matrix)
    sym_sparse_matrix = make_matrix_symmetric(sparse_matrix)
    assert sp.issparse(sym_sparse_matrix), "The output should be a sparse matrix."
    assert np.allclose(
        sym_sparse_matrix.toarray(), sym_sparse_matrix.T.toarray(), equal_nan=True
    ), "The output sparse matrix should be symmetric."


def test_create_colormap(toy_dataframe):
    colormap = create_colormap(toy_dataframe, "cell_group_key")
    assert isinstance(colormap, pd.Series), "The output should be a pandas Series."
    assert len(colormap) == len(toy_dataframe), "The colormap length should match the number of rows in the dataframe."
    assert all(colormap.notna()), "The colormap should not contain NaN values."
    unique_colors = colormap.unique()
    assert len(unique_colors) == len(
        toy_dataframe["cell_group_key"].unique()
    ), "There should be a unique color for each unique value in the column."


def test_describe_metadata(toy_dataframe, capsys):
    describe_metadata(toy_dataframe)
    captured = capsys.readouterr()
    assert "Column" in captured.out, "The function should print information about each column."
    assert "numeric_col" in captured.out, "The function should describe all columns in the dataframe."
    assert "Possibly, numerical columns:" in captured.out, "The function should suggest possible numerical columns."
    assert "Possibly, categorical columns:" in captured.out, "The function should suggest possible categorical columns."
    assert "numeric_col" in captured.out, "The function should correctly identify 'numeric_col' as a numerical column."
    assert (
        "cell_group_key" in captured.out
    ), "The function should correctly identify 'cell_group_key' as a categorical column."


def test_calculate_average_without_nans():
    # no nan
    array = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    averages, sample_sizes = calculate_average_without_nans(array, axis=0)
    expected_averages = np.array([4, 5, 6])
    expected_sample_sizes = np.array([3, 3, 3])
    assert np.allclose(averages, expected_averages), "Averages are not calculated correctly for array without NaNs."
    assert np.array_equal(
        sample_sizes, expected_sample_sizes
    ), "Sample sizes are not calculated correctly for array without NaNs."

    # wuth nan
    array = np.array(
        [
            [1, 2, np.nan],
            [4, np.nan, 6],
            [np.nan, 8, 9],
        ]
    )
    averages, sample_sizes = calculate_average_without_nans(array, axis=0)
    expected_averages = np.array([2.5, 5, 7.5])
    expected_sample_sizes = np.array([2, 2, 2])
    assert np.allclose(averages, expected_averages), "Averages are not calculated correctly for array with NaNs."
    assert np.array_equal(
        sample_sizes, expected_sample_sizes
    ), "Sample sizes are not calculated correctly for array with NaNs."

    # all nan in same axis
    array = np.array(
        [
            [np.nan, 2, np.nan],
            [np.nan, np.nan, 6],
            [np.nan, 8, np.nan],
        ]
    )
    averages, sample_sizes = calculate_average_without_nans(array, axis=0, default_value=0)
    expected_averages = np.array([0, 5, 6])
    expected_sample_sizes = np.array([0, 2, 1])
    assert np.allclose(
        averages, expected_averages
    ), "Averages are not calculated correctly when all values are NaN along an axis."
    assert np.array_equal(
        sample_sizes, expected_sample_sizes
    ), "Sample sizes are not calculated correctly when all values are NaN along an axis."

    averages = calculate_average_without_nans(array, axis=0, return_sample_sizes=False)
    assert np.allclose(
        averages, expected_averages
    ), "Averages are not calculated correctly when return_sample_sizes is False."
