# %%
# import numpy as np
# from df2onehot import df2onehot, import_example
# print(df2onehot.__version__)
# Import libraries

# %%
import pandas as pd
import numpy as np
import df2onehot

df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'colors': [
        ['red', 'blue'],
        [],
        ['green'],
        ['red'],
        [],
        np.nan
    ],
    'numbers': [
        [1, 2],
        [2, 3],
        [],
        [1],
        [3, 4],
        np.nan
    ],
    'category': ['A', 'B', 'A', 'C', 'B', 'A']
})

print(df)

out = df2onehot.df2onehot(df, deep_extract=True, y_min=1, verbose='debug')

print(out['df'])
print(out['onehot'])
print(out['numeric'])

# %%
df_empty = pd.DataFrame({
    'ID': [1, 2, 3],
    'features': [[], [], []]
})

out = df2onehot.df2onehot(
    df_empty,
    deep_extract=True,
    y_min=1
)

print(out['df'])
print(out['onehot'])

# %% Load example
from df2onehot import df2onehot, import_example
df = import_example(data="titanic")
dfhot = df2onehot(df, remove_multicollinearity=True, y_min=2)['onehot']

# %%
df = import_example('student')

# Run df2onehot
results = df2onehot(df, deep_extract=True)

results['onehot']
results['numeric']



# %% Convert
results = df2onehot(df)
results['numeric']

# %% Force feature (int or float) to be numeric if unique non-zero values are above percentage.
out = df2onehot(df, perc_min_num=0.8)


# %% Remove categorical features for which less then 2 values exists
out = df2onehot(df, y_min=2)


# %% Combine two rules above
results = df2onehot(df, y_min=10, perc_min_num=0.8, excl_background=['0.0'])

# %%
import numpy as np
[uiy, ycounts] = np.unique(out['labx'], return_counts=True)
idx = np.argsort(ycounts)[::-1]
print(np.c_[uiy[idx], ycounts[idx]])
print('Total onehot features: %.0d' %(np.sum(ycounts)))
