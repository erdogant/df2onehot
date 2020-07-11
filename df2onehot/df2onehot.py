"""Convert dataframe to one-hot matrix."""
# ----------------------------------------------------
# Name        : df2onehot.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/df2onehot
# Licence     : MIT
# ----------------------------------------------------

# %% Libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from df2onehot.set_dtypes import set_dtypes
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')


# %% Dataframe to one-hot
def df2onehot(df, dtypes='pandas', y_min=None, perc_min_num=None, hot_only=True, list_expand=True, excl_background=None, verbose=3):
    """Convert dataframe to one-hot matrix.

    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe for which the rows are the features, and colums are the samples.
    dtypes : list of str or 'pandas', optional
        Representation of the columns in the form of ['cat','num']. By default the dtype is determiend based on the pandas dataframe.
    y_min : int [0..len(y)], optional
        Minimal number of sampels that must be present in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the enriching model. The default is None.
    perc_min_num : float [None, 0..1], optional
        Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8
    hot_only : bool [True, False], optional
        When True; the output of the onehot matrix exclusively contains categorical values that are transformed to one-hot. The default is True.
    list_expand : bool [True, False], optional
        Expanding of columns that contain lists of strings.. The default is True.
    excl_background : list or None, [0], [0, '0.0', 'male', ...], optional
        Remove values/strings that labeled as background. As an example, in a two-class approach with [0,1], the 0 is usually the background and not of interest. The default is None.
    verbose : int, optional
        Print message to screen. The default is 3.
        0: (default), 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    dict:
    numeric : DataFrame
        Input-dataframe with converted numerical values
    onehot : DataFrame
        Input-dataframe with converted one-hot values. Note that continues values are removed.
    labx : list of str
        Input feature-labels or names
    dtypes : list of str
        The set dtypes for the feature-labels. These can be of type 'num' (numerical) or 'cat' (categorical).

    Examples
    --------
    >>> import df2onehot
    >>> df = df2onehot.import_example()
    >>> out = df2onehot.df2onehot(df)

    """
    args = {}
    args['dtypes'] = dtypes
    args['verbose'] = verbose
    args['perc_min_num'] = perc_min_num
    args['list_expand'] = list_expand
    args['excl_background'] = excl_background
    labx = []
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Determine Dtypes
    df, dtypes = set_dtypes(df, args['dtypes'], is_list=args['list_expand'], perc_min_num=args['perc_min_num'], verbose=args['verbose'])
    # If any column is a list, also expand the list!
    df, dtypes = _expand_column_with_list(df, dtypes, args['verbose'])

    # Make empty frames
    out_numeric = pd.DataFrame()
    out_onehot = pd.DataFrame()
    max_str_len = np.max(list(map(len, df.columns.values.astype(str).tolist())))+2

    # Run over all columns
    for i in np.arange(0, df.shape[1]):
        makespaces = ''.join(['.'] * (max_str_len - len(df.columns[i])))
        # Do not touch a float
        if 'float' in str(df.dtypes[i]):
            # if verbose>=3: print('[df2onehot] >Working on %s' %(df.columns[i]))
            if verbose>=3: print('[df2onehot] >Working on %s%s[float]' %(df.columns[i], makespaces))
            out_numeric[df.columns[i]] = df.iloc[:,i]
            if hot_only is False:
                out_onehot[df.columns[i]] = df.iloc[:,i]
                labx.append(df.columns[i])
        else:
            integer_encoded = label_encoder.fit_transform(df.iloc[:,i])
            # If all values are the same, the encoder will return 0 (=False). We set values at 1 (by +1) and make them True. Otherwise it can be mis interpreted the the value was not present in the datset.
            if np.all(np.unique(integer_encoded)==0): integer_encoded=integer_encoded+1
            # integer_encoded = set_y(integer_encoded, y_min=y_min, numeric=True, verbose=0)
            out_numeric[df.columns[i]] = integer_encoded
            out_numeric[df.columns[i]] = out_numeric[df.columns[i]].astype('category')
            if verbose>=3: print('[df2onehot] >Working on %s%s[%.0f]' %(df.columns[i], makespaces, len(np.unique(integer_encoded)) ))

            # Contains a single value
            if len(np.unique(integer_encoded))<=1:
                out_onehot[df.columns[i]] = integer_encoded.astype('Bool')
                labx.append(df.columns[i])
            else:
                # binary encode
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
                # Remove columns if it does not fullfill minimum nr. of samples (>=y_min)
                if y_min is not None:
                    onehot_encoded = onehot_encoded[:,onehot_encoded.sum(axis=0)>=y_min]
                # Make new one-hot columns
                for k in range(0,onehot_encoded.shape[1]):
                    # Get the colname based on the value in the orignal dataframe
                    label=df.iloc[onehot_encoded[:,k]==1,i].unique().astype(str)[0]

                    # Check whether this is a label that should be excluded.
                    if (isinstance(args['excl_background'], type(None))) or (not np.isin(label, args['excl_background'])):
                        colname = df.columns[i] + '_' + label
                        out_onehot[colname] = onehot_encoded[:,k].astype('Bool')
                        labx.append(df.columns[i])

                # Make numerical vector
                if onehot_encoded.shape[1]>2:
                    out_numeric[df.columns[i]] = (onehot_encoded * np.arange(1,onehot_encoded.shape[1] + 1)).sum(axis=1)

    uiy, ycounts = np.unique(labx, return_counts=True)
    if verbose >=3: print('[df2onehot] >\n[df2onehot] >Total onehot features: %.0d' %(np.sum(ycounts)))
    # idx = np.argsort(ycounts)[::-1]
    # print(np.c_[uiy[idx], ycounts[idx]])

    # Make sure its limited to the number of y_min
    labx = np.array(labx, dtype=str)
    dtypes = np.array(dtypes)
    if y_min is not None:
        Iloc = (out_onehot.sum(axis=0)>=y_min).values
        out_onehot = out_onehot.loc[:,Iloc]
        labx=labx[Iloc]

    out = {}
    out['numeric'] = out_numeric
    out['dtypes'] = dtypes
    out['onehot'] = out_onehot
    out['labx'] = labx
    return(out)


# %%
def _expand_column_with_list(df, dtypes, verbose=3):
    # Check for any lists in dtypes
    Icol = np.isin(dtypes,'list')

    # If any
    if np.any(Icol):
        # Empty df
        df2 = pd.DataFrame()
        idxCol = np.where(Icol)[0]

        # Expand columns with lists
        for i in range(0,len(idxCol)):
            if verbose>=3: print('[df2onehot] >Array-like column is detected: [%s]' %(df.columns[idxCol[i]]))
            # Gather only the not NaN rows
            Inan = df.iloc[:,idxCol[i]].isna()
            # Grap only unique elements from the combined set of lists
            dftmp = df.iloc[~Inan.values, idxCol[i]].copy()

            # Make str, float and int elements of type list
            typing = list(map(type, dftmp.values))
            Iloc = np.logical_or(list(map(lambda x: isinstance(str(), x), typing)), np.logical_or(list(map(lambda x: isinstance(float(), x), typing)), list(map(lambda x: isinstance(int(), x), typing))))
            if np.any(Iloc):
                # Convert to string! This is required because the colnames also converts all to string.
                dftmp.loc[Iloc] = list(map(lambda x: [str(x)], dftmp.loc[Iloc]))
                df.iloc[dftmp.loc[Iloc].index,idxCol[i]] = dftmp.loc[Iloc]

            # Get all unique elements
            listvector = list(map(lambda x: list(x) , dftmp))
            listvector = sum(listvector, [])
            # All elements are converted to string!
            colnames = np.unique(listvector)
            # Lookup colname in the vector and make array
            dftmp = df.iloc[:,idxCol[i]].apply(_findcol, cols=colnames)
            arr = np.concatenate(dftmp).reshape((dftmp.shape[0], dftmp[0].shape[0]))
            df1 = pd.DataFrame(index=np.arange(0,df.shape[0]), columns=colnames, data=arr, dtype='bool')
            # Combine in one big matrix
            if df2.empty:
                df2 = df1.copy()
                # df2 = pd.concat([df2.astype(bool), df1], axis=1)
            else:
                for colname in df1.columns:
                    if np.any(np.isin(colname, df2.columns.values)):
                        df2[colname] = np.logical_or(df2[colname], df1[colname])
                    else:
                        df2[colname] = df1[colname].copy()

            # pd.merge(df_list_to_onehot, df1, on=df_list_to_onehot.columns.values).sum(axis=1)
            # pd.merge(df_list_to_onehot, df1, how='inner')
            # totlist = list(df_list_to_onehot.columns.values)+list(df1.columns.values)
            # pd.concat([df_list_to_onehot, df1]).groupby(totlist).sum().reset_index()
            # pd.merge(df_list_to_onehot, df1, on=totlist).sum(axis=1)
            
        # Drop columns that are expanded
        df.drop(labels = df.columns[Icol].values, axis=1, inplace=True)
        # Combine new one-hot-colums
        df = pd.concat([df, df2], axis=1)

    # Redo the typing
    df, dtypes = set_dtypes(df, verbose=0)
    # Return
    return(df, dtypes)


# %% Find columns
def _findcol(x, cols):
    return(np.isin(cols, x))


# %% Example data
def import_example(getfile='titanic'):
    """Import example.

    Description
    -----------

    Parameters
    ----------
    getfile : String, optional
        'titanic'

    Returns
    -------
    df : DataFrame

    """

    if getfile=='titanic':
        getfile='titanic_train.zip'

    print('[df2onehot] >Loading %s..' %getfile)
    curpath = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_DATA=os.path.join(curpath, 'data', getfile)
    if os.path.isfile(PATH_TO_DATA):
        df=pd.read_csv(PATH_TO_DATA, sep=',')
        return df
    else:
        print('[df2onehot] >Oops! Example data not found!')
        return None

