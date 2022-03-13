from df2onehot import df2onehot
import pandas as pd
import pypickle
filepath = 'D://GITLAB/PROJECTS/shodanrws/results/shodandata.pkl'
X = pypickle.load(filepath)
df = pd.json_normalize(X['matches'])

Xout = df2onehot(df, deep_extract=True, verbose=3)
# Xext = df2onehot(df, deep_extract=deep_extract, y_min=1, perc_min_num=0.8, verbose=3)

Xout['onehot'].shape
Xout['onehot'].sum(axis=0)
len(Xout['labx'])

Xout['numeric'].shape
len(Xout['labels'])

Xout['df'].shape

import sweetviz as sv
my_report = sv.analyze(Xout['df'].iloc[:,0:100], pairwise_analysis='off')
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"

# Doe iets met timestamp
# Doe iets met locatie


# %%
import numpy as np
import df2onehot
# print(df2onehot.__version__)

df = df2onehot.import_example()
colnames = np.array(['3','4','5','6','7','8','9','11','10','1'])
df['lists'] = np.nan
df['lists'].iloc[0] = ['3',4]
df['lists'].iloc[2] = ['5','6','7','8']
df['lists'].iloc[888] = ['9','11','4']
df['lists'].iloc[889] = 10
df['lists'].iloc[890] = 1

df['list2'] = np.nan
df['list2'].iloc[0] = ['4','45']
df['list2'].iloc[890] = 1
df['list2'].iloc[888] = 10

# Run df2onehot
out = df2onehot.df2onehot(df, deep_extract=True)

# # TEST 1: check output is unchanged
# out = df2onehot(df)
# out['onehot']
out['numeric'][colnames]


# %% Load example
import df2onehot
df = df2onehot.import_example()


# %% Convert
out = df2onehot.df2onehot(df)
out['numeric']

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
