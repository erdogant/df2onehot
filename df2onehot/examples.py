# %%
# import numpy as np
# from df2onehot import df2onehot, import_example
# print(df2onehot.__version__)
# Import libraries
from clusteval import clusteval
from df2onehot import df2onehot

ce = clusteval()

# Import data from url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
df = ce.import_example(url=url)

# Preprocessing
cols_as_float = ['ProductRelated', 'Administrative']
df[cols_as_float]=df[cols_as_float].astype(float)
dfhot = df2onehot(df, excl_background=['0.0', 'None', '?', 'False'], y_min=50, perc_min_num=0.8, remove_mutual_exclusive=True, verbose=4)['onehot']
# Initialize library
from sklearn.manifold import TSNE
xycoord = TSNE(n_components=2, init='random', perplexity=30).fit_transform(dfhot.values)

# Initialize clusteval
ce = clusteval(cluster='agglomerative', metric='euclidean', linkage='complete', min_clust=9, max_clust=30)

# Clustering and evaluation
results = ce.fit(xycoord)

enrichment_results = ce.enrichment(df)

# Make plots
ce.plot()
ce.plot_silhouette()

ce.scatter(n_feat=2)

# %%




# %%
# Import libraries
# Import libraries
from clusteval import clusteval
from df2onehot import df2onehot

# Load data from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'

# Initialize clusteval
ce = clusteval()

# Import data from url
df = ce.import_example(url=url)

# Preprocessing
cols_as_float = ['ProductRelated', 'Administrative']
df[cols_as_float]=df[cols_as_float].astype(float)
dfhot = df2onehot(df, excl_background=['0.0', 'None', '?', 'False'], y_min=50, perc_min_num=0.8, remove_mutual_exclusive=True, verbose=4)['onehot']

# Initialize using the specific parameters
ce = clusteval(evaluate='silhouette',
               cluster='agglomerative',
               metric='hamming',
               linkage='complete',
               min_clust=2,
               verbose='info')

# Clustering and evaluation
results = ce.fit(dfhot)

# [clusteval] >INFO> Saving data in memory.
# [clusteval] >INFO> Fit with method=[agglomerative], metric=[hamming], linkage=[complete]
# [clusteval] >INFO> Evaluate using silhouette.
# [clusteval] >INFO: 100%|██████████| 23/23 [00:28<00:00,  1.23s/it]
# [clusteval] >INFO> Compute dendrogram threshold.
# [clusteval] >INFO> Optimal number clusters detected: [9].
# [clusteval] >INFO> Fin.

# Make plots
ce.plot()
# ce.plot_silhouette()
ce.plot_silhouette(embedding='tsne')

# %%

# Initialize library
from sklearn.manifold import TSNE
xycoord = TSNE(n_components=2, init='random', perplexity=30).fit_transform(dfhot.values)

# Initialize clusteval
ce = clusteval(cluster='agglomerative', metric='euclidean', linkage='complete', min_clust=5, max_clust=30)

# Clustering and evaluation
results = ce.fit(xycoord)

# Make plots
ce.plot()
ce.plot_silhouette()

enrichment_results = ce.enrichment(df)

# %% Load example
from df2onehot import df2onehot, import_example
df = import_example(data="titanic")
dfhot = df2onehot(df, remove_multicollinearity=True, y_min=2)['onehot']

# %%

enrichment_results = ce.enrichment(df)

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
