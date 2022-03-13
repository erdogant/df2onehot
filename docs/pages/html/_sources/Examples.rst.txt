Process Mixed dataset
####################################

In the following example we load the Titanic dataset, and use ``df2onehot`` to convert it towards a structured dataset.

.. code:: python
	
	# Load library
	from df2onehot import df2onehot, import_example
	
	# Import Titanic dataset
	df = import_example(data="titanic")

	print(df)
	#      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
	# 0              1         0       3  ...   7.2500   NaN         S
	# 1              2         1       1  ...  71.2833   C85         C
	# 2              3         1       3  ...   7.9250   NaN         S
	# 3              4         1       1  ...  53.1000  C123         S
	# 4              5         0       3  ...   8.0500   NaN         S
	# ..           ...       ...     ...  ...      ...   ...       ...
	# 886          887         0       2  ...  13.0000   NaN         S
	# 887          888         1       1  ...  30.0000   B42         S
	# 888          889         0       3  ...  23.4500   NaN         S
	# 889          890         1       1  ...  30.0000  C148         C
	# 890          891         0       3  ...   7.7500   NaN         Q
	# 
	# [891 rows x 12 columns]

	# Convert the matrix into a structured datasset
	results = df2onehot(df)

	print(results.keys())
	# dict_keys(['numeric', 'dtypes', 'onehot', 'labx', 'df', 'labels'])
	
	# The onehot array exploded into 2637 features!
	print(results['onehot'].shape)
	# (891, 2637)
	
	# The reason for the large onehot dataset is, among others incorrect typing of variables. 
	# Columns such as PassengerId should be removed because now these are typed as categorical.
	# Prevent that integer variables are typed as categorical.
	
	print(np.c_[results['labels'],results['dtypes']])
	# array([['PassengerId', 'cat'],
	#        ['Survived', 'cat'],
	#        ['Pclass', 'cat'],
	#        ['Name', 'cat'],
	#        ['Sex', 'cat'],
	#        ['Age', 'num'],
	#        ['SibSp', 'cat'],
	#        ['Parch', 'cat'],
	#        ['Ticket', 'cat'],
	#        ['Fare', 'num'],
	#        ['Cabin', 'cat'],
	#        ['Embarked', 'cat'],
	#        ['all_true', 'cat']], dtype=object)


Force categorical values into numeric
**************************************

We can force variables to be numeric if the number of unique values are above the given percentage: 80%.
Or in other words, if a variable contains more then 80% unique values, it is set as numerical.

.. code:: python
	
	# Set the parameter to force columns into numerical dtypes
	results = df2onehot(df, perc_min_num=0.8)

	# Also remove categorical features for which less then 2 values exists.
	results = df2onehot(df, perc_min_num=0.8, y_min=2)
	
	# Check whether the dtypes are correct.
	# PassengerId, Age and Fare are set as numerical, and the rest categorical.
	
	print(np.c_[results['labels'],results['dtypes']])
	
	# [['PassengerId' 'num']
	#  ['Survived' 'cat']
	#  ['Pclass' 'cat']
	#  ['Name' 'cat']
	#  ['Sex' 'cat']
	#  ['Age' 'num']
	#  ['SibSp' 'cat']
	#  ['Parch' 'cat']
	#  ['Ticket' 'cat']
	#  ['Fare' 'num']
	#  ['Cabin' 'cat']
	#  ['Embarked' 'cat']
	#  ['all_true' 'cat']]

	# If we look at our one hot dense array, we notice that behind each column the sub-category is added.
	print(results['onehot'])
	
	#      Survived_0.0  Survived_1.0  Pclass_1.0  ...  Embarked_Q  Embarked_S  all_true
	# 0            True         False       False  ...       False        True      True
	# 1           False          True        True  ...       False       False      True
	# 2           False          True       False  ...       False        True      True
	# 3           False          True        True  ...       False        True      True
	# 4            True         False       False  ...       False        True      True
	# ..            ...           ...         ...  ...         ...         ...       ...
	# 886          True         False       False  ...       False        True      True
	# 887         False          True        True  ...       False        True      True
	# 888          True         False       False  ...       False        True      True
	# 889         False          True        True  ...       False       False      True
	# 890          True         False       False  ...        True       False      True
	# 
	# [891 rows x 206 columns]


Exclude redundant variables 
**************************************

We can make further clean the data by removing mutually exclusive columns.
As an example, the column **Survived** is split into *Survived_0.0* and *Survived_1.0* but the column *Survived_0.0* may not be so relevant. With the parameter ``excl_background`` we can ignore the labels that are put begin the columns.

.. code:: python

	# Ignore specific subcategories
	results = df2onehot(df, perc_min_num=0.8, y_min=2, excl_background=['0.0'])

	# The final shape of our structured dataset is:
	results['onehot'].shape
	(891, 203)

	# The original variable names can be found here:
	results['labx']

Exclude sparse variables
**************************************

By converting categorical values into one-hot dense arrays, it can easily occur that certain variables will only contain a single or few ``True`` values. We can use the ``y_min`` functionality to remove such columns.


.. code:: python

	# We can tune the ``y_min`` parameter further remove even more columns.
	results = df2onehot(df, perc_min_num=0.8, y_min=5, excl_background=['0.0'])

	# The final shape of our structured dataset is:
	results['onehot'].shape
	(891, 29)

We still need to manually remove the identifier column and then we are ready to go for analysis!


Custom dtypes
####################################

In the following example we load the **fifa** dataset and structure the dataset. 


.. code:: python

	# Load library
	from df2onehot import df2onehot, import_example

	# Import Fifa dataset
	df = import_example('sprinkler')

	# Custom typing of the columns
	results = df2onehot(df, dtypes=['cat','cat','cat','cat'], excl_background=['0.0'])



Extracting deep lists
####################################

.. code:: python

	# Load library
	from df2onehot import df2onehot, import_example

	# Import complex dataframe containing lists in lists
	df = import_example('complex')
	
	# 
	#           feat_1   feat_2
	# 0         [3, 4]  [4, 45]
	# 1            NaN      NaN
	# 2   [5, 6, 7, 8]      NaN
	# 3            NaN      NaN
	# 4            NaN      NaN
	# 5             10      NaN
	# 6            NaN      NaN
	# 7            NaN      NaN
	# 8            NaN      NaN
	# 9            NaN      NaN
	# 10           NaN      NaN
	# 11           NaN      NaN
	# 12           NaN      NaN
	# 13           NaN      NaN
	# 14           NaN      NaN
	# 15             1        1
	# 16           NaN      NaN
	# 17           NaN      NaN
	# 18           NaN      NaN
	# 19           NaN      NaN
	# 20    [9, 11, 4]       10
	# 21           NaN      NaN
	# 22           NaN      NaN
	# 23           NaN      NaN
	# 24           NaN      NaN


Convert to onehot dense-array without using the ``deep_extract`` function.

.. code:: python

	results = df2onehot(df, deep_extract=False)
	
	# With ``deep_extract=False`` we the full element value is used as a new column name.
	print(results['onehot'])

	#     feat_1_1  feat_1_10  ...  feat_2_None  feat_2_['4', '45']
	# 0      False      False  ...        False                True
	# 1      False      False  ...         True               False
	# 2      False      False  ...         True               False
	# 3      False      False  ...         True               False
	# 4      False      False  ...         True               False
	# 5      False       True  ...         True               False
	# 6      False      False  ...         True               False
	# 7      False      False  ...         True               False
	# 8      False      False  ...         True               False
	# ...
	# [25 rows x 10 columns]


With ``deep_extract=False``, each element is analyzed whether it contains lists or dictionaries and each element value has become a new column. If a column name already exists, the value is added to that row.

.. code:: python

	# Convert to onehot dense-array with the ``deep_extract=True`` function
	results = df2onehot(df, deep_extract=True)
	
	# print
	print(results['onehot'])

	#	1     10     11      3      4      5      6      7      8      9     45
	# 0   False  False  False   True   True  False  False  False  False  False   True
	# 1   False  False  False  False  False  False  False  False  False  False  False
	# 2   False  False  False  False  False   True   True   True   True  False  False
	# 3   False  False  False  False  False  False  False  False  False  False  False
	# 4   False  False  False  False  False  False  False  False  False  False  False
	# 5   False   True  False  False  False  False  False  False  False  False  False
	# 6   False  False  False  False  False  False  False  False  False  False  False
	# 7   False  False  False  False  False  False  False  False  False  False  False
	# 8   False  False  False  False  False  False  False  False  False  False  False
	# ...
	# [25 rows x 11 columns]


	# Lets print only the relevant rows.
	idx = results['onehot'].sum(axis=1)>0
	print(results['onehot'].loc[idx,:])
	# 	1     10     11      3      4      5      6      7      8      9     45
	# 0   False  False  False   True   True  False  False  False  False  False   True
	# 2   False  False  False  False  False   True   True   True   True  False  False
	# 5   False   True  False  False  False  False  False  False  False  False  False
	# 15   True  False  False  False  False  False  False  False  False  False  False
	# 20  False   True   True  False   True  False  False  False  False   True  False


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
