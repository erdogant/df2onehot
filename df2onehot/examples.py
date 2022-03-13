from df2onehot import df2onehot, import_example
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
from df2onehot import df2onehot, import_example
# print(df2onehot.__version__)

df = import_example('complex')

# Run df2onehot
results = df2onehot(df, deep_extract=True)

results['onehot']
results['numeric']


# %% Load example
from df2onehot import df2onehot, import_example
df = import_example(data="titanic")


# %% Convert
results = df2onehot(df)
out['numeric']

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
