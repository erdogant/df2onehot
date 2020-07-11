# %%
# import numpy as np
# import df2onehot
# df = df2onehot.import_example()
# colnames = np.array(['3','4','5','6','7','8','9','11'])
# df['lists'] = np.nan
# df['lists'].iloc[0] = ['3','4']
# df['lists'].iloc[2] = ['5','6','7','8']
# df['lists'].iloc[888] = ['9','11','4']
# df['lists'].iloc[889] = 10
# df['lists'].iloc[890] = 1

# # Run df2onehot
# out = df2onehot.df2onehot(df)

# # TEST 1: check output is unchanged
# out = df2onehot(df)
# out['onehot']

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
