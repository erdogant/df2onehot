# pytest tests\test_df2onehot.py

import numpy as np
from df2onehot import df2onehot, import_example
df = import_example()


def test_df2onehot():
    # TEST 1: check output is unchanged
    out = df2onehot(df)
    assert [*out.keys()]==['numeric', 'onehot', 'labx', 'dtypes']
    # TEST 2: Check model output is unchanged
    [uiy, ycounts] = np.unique(out['labx'], return_counts=True)
    assert np.all(ycounts==np.array([148,   4, 891,   7, 891,   3,   2,   7,   2, 681]))
    # TEST 3:
    out = df2onehot(df, perc_min_num=0.8)
    [uiy, ycounts] = np.unique(out['labx'], return_counts=True)
    assert np.all(ycounts==np.array([148,   4, 891,   7,   3,   2,   7,   2, 681]))
    # TEST 4:
    out = df2onehot(df, y_min=2)
    [uiy, ycounts] = np.unique(out['labx'], return_counts=True)
    assert np.all(ycounts==np.array([ 47,   4,   6,   3,   2,   7,   2, 134]))
    # TEST 4:
    out = df2onehot(df, y_min=2, excl_background=['male'])
    assert out['onehot'].shape[1]==204
