# pytest tests\test_df2onehot.py
import pandas as pd
import numpy as np
from df2onehot import df2onehot, import_example


def test_df2onehot():
    df = import_example()
    # TEST 1: check output is unchanged
    out = df2onehot(df)
    assert [*out.keys()]==['numeric','dtypes','onehot','labx']
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
    out = df2onehot(df, y_min=0)
    assert np.all(out['onehot'].sum(axis=0)>=0)
    # TEST 4:
    out = df2onehot(df, y_min=1)
    assert np.all(out['onehot'].sum(axis=0)>=1)
    # TEST 4:
    out = df2onehot(df, y_min=10)
    assert np.all(out['onehot'].sum(axis=0)>=10)
    out = df2onehot(df, y_min=100)
    assert np.all(out['onehot'].sum(axis=0)>=100)
    # TEST 4:
    out = df2onehot(df, y_min=2, excl_background=['male'])
    assert out['onehot'].shape[1]==204
    
    ymins = [0,1,10,100]
    k=[5,10,100]
    for n in k:
        for y_min in ymins:
            print(y_min,n)
            tmpdata = np.random.random_integers(0,1,size=(n,n))
            df=pd.DataFrame(data=tmpdata)
            df.columns=df.columns.astype(str)
            df['allfalse']=0
            df['alltrue']=1
            df['alltrue_butone']=1
            df['alltrue_butone'][0]=0
            out = df2onehot(df, y_min=y_min, verbose=0)
            assert np.all(out['onehot'].sum(axis=0)>=y_min)

