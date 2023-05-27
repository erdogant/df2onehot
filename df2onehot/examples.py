# %%
# import numpy as np
# from df2onehot import df2onehot, import_example
# print(df2onehot.__version__)

# %% Load example
from df2onehot import df2onehot, import_example
df = import_example(data="titanic")
dfhot = df2onehot(df, remove_multicollinearity=True, y_min=2)['onehot']


# %%
df = import_example('complex')

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
[uiy, ycounts] = np.unique(out['labx'], return_counts=True)
idx = np.argsort(ycounts)[::-1]
print(np.c_[uiy[idx], ycounts[idx]])
print('Total onehot features: %.0d' %(np.sum(ycounts)))
