# %%
import numpy as np
import df2onehot
print(df2onehot.__version__)

# %% Load example
df = df2onehot.import_example()


# %% Convert
out = df2onehot.df2onehot(df)


# %% Force feature (int or float) to be numeric if unique non-zero values are above percentage.
out = df2onehot.df2onehot(df, perc_min_num=0.8)


# %% Remove categorical features for which less then 2 values exists
out = df2onehot.df2onehot(df, y_min=2)


# %% Combine two rules above
out = df2onehot.df2onehot(df, y_min=2, perc_min_num=0.8)

# %%
[uiy, ycounts] = np.unique(out['labx'], return_counts=True)
idx = np.argsort(ycounts)[::-1]
print(np.c_[uiy[idx], ycounts[idx]])
print('Total onehot features: %.0d' %(np.sum(ycounts)))
