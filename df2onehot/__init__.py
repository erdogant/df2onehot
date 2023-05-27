from df2onehot.df2onehot import (
    df2onehot,
    import_example,
    )

from df2onehot.utils import (
 	set_dtypes,
 	is_DataFrame,
    set_y,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.4'

# module level doc-string
__doc__ = """
df2onehot is an Python package to convert a pandas dataframe into a stuctured dataframe.
=============================================================================================

Example
-------
>>> import df2onehot
>>> df = df2onehot.import_example()
>>> out = df2onehot.df2onehot(df)

"""
