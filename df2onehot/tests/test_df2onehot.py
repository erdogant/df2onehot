# pytest tests\test_df2onehot.py
import pandas as pd
import numpy as np
from df2onehot import df2onehot, import_example
import unittest


class Testdf2onehot(unittest.TestCase):

	def test_df2onehot(self):
		df = import_example()
		df['all_true']=1
		# TEST 1: check output is unchanged
		out = df2onehot(df)
		assert [*out.keys()]==['numeric','dtypes','onehot','labx']
		# TEST 2: Check model output is unchanged
		[uiy, ycounts] = np.unique(out['labx'], return_counts=True)
		assert np.all(ycounts==np.array([148,   4, 891,   7, 891,   3,   2,   7,   2, 681, 1]))
		# TEST WHETHER SIMILAR VALUES ARE SET TO TRUE
		assert out['onehot']['all_true'].sum()==df.shape[0]
		# TEST WHETHER SIZE MATCHES
		assert out['numeric'].shape[1]==len(out['dtypes'])

		df = import_example()

		# TEST 3:
		out = df2onehot(df, deep_extract=False, perc_min_num=0.8)
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

		# TEST ARRAYS:
		df = import_example()
		colnames = np.array(['3','4','5','6','7','8','9','11'])
		df['lists'] = np.nan
		df['lists'].iloc[0] = colnames[[0,1]]
		df['lists'].iloc[2] = colnames[[2,3,4,5]]
		df['lists'].iloc[888] = colnames[[6,7,1]]
		df['lists'].iloc[889] = 10
		df['lists'].iloc[890] = 1

		# Run df2onehot
		out = df2onehot(df, deep_extract=True)
		counts = out['numeric'][list(colnames)].sum(axis=1)
		# Make some checks
		for i in range(0,df.shape[0]):
			if isinstance(list(), type(df['lists'].iloc[i])):
				assert counts[i] == len(df['lists'].iloc[i])
				idx=np.where(out['numeric'].iloc[i][colnames]==1)[0]
				assert np.all(np.sort(df['lists'].iloc[i])==np.sort(colnames[idx]))
		
		# TEST lists:
		df = import_example()
		colnames = np.array(['3','4','5','6','7','8','9','11'])
		df['lists'] = np.nan
		df['lists'].iloc[0] = ['3','4']
		df['lists'].iloc[2] = ['5','6','7','8']
		df['lists'].iloc[888] = ['9','11','4']
		df['lists'].iloc[889] = 10
		df['lists'].iloc[890] = 1

		# Run df2onehot
		del out
		out = df2onehot(df, deep_extract=True)
		counts = out['numeric'][list(colnames)].sum(axis=1)
		# Make some checks
		for i in range(0,df.shape[0]):
			if isinstance(list(), type(df['lists'].iloc[i])):
				assert counts[i] == len(df['lists'].iloc[i])
				idx=np.where(out['numeric'].iloc[i][colnames]==1)[0]
				assert np.all(np.sort(df['lists'].iloc[i])==np.sort(colnames[idx]))
				
				
		# TEST 1: check output is unchanged
		out = df2onehot(df)
		assert [*out.keys()]==['numeric','dtypes','onehot','labx']


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

