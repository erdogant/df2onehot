
import numpy as np
import pandas as pd
import df2onehot


def test_deep_extract_all_empty_lists():

    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'features': [[], [], []],
    })

    out = df2onehot.df2onehot(
        df,
        deep_extract=True,
        y_min=1,
        verbose=0,
    )

    assert out['df'].columns.tolist() == ['ID']
    assert len(out['labels']) == out['df'].shape[1]
    assert len(out['dtypes']) == out['df'].shape[1]
    assert out['df'].shape[0] == 3


def test_deep_extract_mixed_empty_lists():

    df = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'features': [
            ['red', 'blue'],
            [],
            ['green'],
            np.nan,
        ],
    })

    out = df2onehot.df2onehot(
        df,
        deep_extract=True,
        y_min=1,
        verbose=0,
    )

    assert set(out['df'].columns) == {
        'ID', 'blue', 'green', 'red'
    }

    assert out['df']['blue'].tolist() == [True, False, False, False]
    assert out['df']['green'].tolist() == [False, False, True, False]
    assert out['df']['red'].tolist() == [True, False, False, False]

    assert len(out['labels']) == out['df'].shape[1]
    assert len(out['dtypes']) == out['df'].shape[1]
    assert out['df'].shape[0] == 4


def test_deep_extract_multiple_list_columns_with_empty_values():

    df = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'colors': [['red'], [], ['blue'], np.nan],
        'numbers': [[1, 2], [2], [], np.nan],
    })

    out = df2onehot.df2onehot(
        df,
        deep_extract=True,
        y_min=1,
        verbose=0,
    )

    assert out['df'].shape[0] == 4
    assert len(out['labels']) == out['df'].shape[1]
    assert len(out['dtypes']) == out['df'].shape[1]

    assert {'1', '2', 'ID', 'blue', 'red'}.issubset(set(out['df'].columns))
