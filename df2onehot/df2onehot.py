"""Convert dataframe to one-hot matrix."""
# ----------------------------------------------------
# Name        : df2onehot.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/df2onehot
# Licence     : MIT
# ----------------------------------------------------

# %% Libraries
import wget
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from df2onehot.utils import set_dtypes
# from set_dtypes import set_dtypes
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')


# %% Dataframe to one-hot
def df2onehot(df, dtypes='pandas', y_min=None, perc_min_num=None, hot_only=True, deep_extract=False, excl_background=None, verbose=3):
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
    deep_extract : bool [False, True] (default : False)
        True: Extract information from a vector that contains a list/array/dict.
        False: converted to a string and treated as catagorical ['cat'].
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
    args['deep_extract'] = deep_extract
    args['excl_background'] = excl_background
    labx = []
    labels = None
    disable = (True if (verbose==0 or verbose>3) else False)

    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Determine Dtypes
    df, dtypes = set_dtypes(df, args['dtypes'], deep_extract=args['deep_extract'], perc_min_num=args['perc_min_num'], verbose=args['verbose'])
    # If any column is a list, also expand the list!
    if args['deep_extract']:
        df, dtypes, labels = _deep_extract(df, dtypes, perc_min_num=args['perc_min_num'], verbose=args['verbose'])

    # Make empty frames
    maxstring=50
    out_numeric = pd.DataFrame()
    out_onehot = pd.DataFrame()
    max_str_len = np.minimum(np.max(list(map(len, df.columns.values.astype(str).tolist()))) + 2, maxstring)

    # Run over all columns
    for i in tqdm(np.arange(0, df.shape[1]), disable=disable):
        makespaces = ''.join(['.'] * np.minimum( (max_str_len - len(df.columns[i])), maxstring) )
        # Do not touch a float
        if 'float' in str(df.dtypes[i]):
            # if verbose>=3: print('[df2onehot] >Working on %s' %(df.columns[i]))
            if verbose>=4: print('[df2onehot] >Processing: %s%s [float]' %(df.columns[i][0:maxstring], makespaces))
            out_numeric[df.columns[i]] = df.iloc[:, i]
            if hot_only is False:
                out_onehot[df.columns[i]] = df.iloc[:, i]
                labx.append(df.columns[i])
        else:
            integer_encoded = label_encoder.fit_transform(df.iloc[:, i])
            # If all values are the same, the encoder will return 0 (=False). We set values at 1 (by +1) and make them True. Otherwise it can be mis interpreted the the value was not present in the datset.
            if np.all(np.unique(integer_encoded)==0): integer_encoded=integer_encoded + 1
            # integer_encoded = set_y(integer_encoded, y_min=y_min, numeric=True, verbose=0)
            out_numeric[df.columns[i]] = integer_encoded
            out_numeric[df.columns[i]] = out_numeric[df.columns[i]].astype('category')
            if verbose>=4: print('[df2onehot] >Processing: %s%s [%.0f]' %(df.columns[i][0:maxstring], makespaces, len(np.unique(integer_encoded)) ))

            # Contains a single value or is bool
            if (len(np.unique(integer_encoded))<=1) or (str(df.dtypes[i])=='bool'):
                out_onehot[df.columns[i]] = integer_encoded.astype('bool')
                labx.append(df.columns[i])
            else:
                # binary encode
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
                # Remove columns if it does not fullfill minimum nr. of samples (>=y_min)
                if y_min is not None:
                    onehot_encoded = onehot_encoded[:, onehot_encoded.sum(axis=0) >= y_min]
                # Make new one-hot columns
                for k in range(0, onehot_encoded.shape[1]):
                    # Get the colname based on the value in the orignal dataframe
                    label = df.iloc[onehot_encoded[:,k]==1, i].unique().astype(str)[0]

                    # Check whether this is a label that should be excluded.
                    if (isinstance(args['excl_background'], type(None))) or (not np.isin(label, args['excl_background'])):
                        colname = df.columns[i] + '_' + label
                        out_onehot[colname] = onehot_encoded[:, k].astype('bool')
                        labx.append(df.columns[i])

                # Make numerical vector
                if onehot_encoded.shape[1]>2:
                    out_numeric[df.columns[i]] = (onehot_encoded * np.arange(1, onehot_encoded.shape[1] + 1)).sum(axis=1)

    uiy, ycounts = np.unique(labx, return_counts=True)
    if verbose >=3: print('[df2onehot] >Total onehot features: %.0d' %(np.sum(ycounts)))
    # idx = np.argsort(ycounts)[::-1]
    # print(np.c_[uiy[idx], ycounts[idx]])

    # Make sure its limited to the number of y_min
    labx = np.array(labx, dtype=str)
    dtypes = np.array(dtypes)
    if y_min is not None:
        Iloc = (out_onehot.sum(axis=0)>=y_min).values
        out_onehot = out_onehot.loc[:, Iloc]
        labx = labx[Iloc]

    if labels is None:
        labels = df.columns.values

    out = {}
    out['numeric'] = out_numeric
    out['dtypes'] = dtypes
    out['onehot'] = out_onehot
    out['labx'] = labx
    out['df'] = df
    out['labels'] = labels
    return(out)


# %% Convert str/float/int to type
def _col2type(dfc, verbose=3):
    # dtype = dtypes[idx]
    # Gather only the not NaN rows
    Inan = dfc.isna()
    # Grap only unique elements from the combined set of lists
    dfcol = dfc.iloc[~Inan.values].copy()
    # Remvoe empty lists
    # Inan = dfcol.apply(len)==0
    # dfcol = dfcol[~Inan.values]

    # If any str, float or int elements is fount, convert to list
    # if dtype=='list':
    typing = list(map(type, dfcol.values))
    Iloc = np.logical_or(list(map(lambda x: isinstance(str(), x), typing)), np.logical_or(list(map(lambda x: isinstance(float(), x), typing)), list(map(lambda x: isinstance(int(), x), typing))))
    if np.any(Iloc):
        # Convert to string! This is required because the colnames also converts all to string.
        dfcol.loc[Iloc] = list(map(lambda x: [str(x)], dfcol.loc[Iloc]))
        dfc.iloc[dfcol.loc[Iloc].index] = dfcol.loc[Iloc]

    # Get all unique elements
    uifeat = _get_unique_elements(dfcol, verbose=verbose)
    # Return
    return(dfc, uifeat)

def _get_unique_elements(dfcol, verbose=3):
    try:
        # Get all unique elements
        listvector = list(map(lambda x: list(x) , dfcol))
        listvector = sum(listvector, [])
        # All elements are converted to string!
        uifeat = np.unique(listvector)
    except:
        # if verbose>=1: print('[df2onehot] >Error catched:' %(str(sys.exc_info()[0])))
        if verbose>=5: print('[df2onehot] >Error catched.')
        uifeat = None

    return uifeat


def _array2df(df, uifeat, idx):
    # Lookup colname in the vector and make array
    dfcol = df.iloc[:, idx].apply(_findcol, cols=uifeat)
    arr = np.concatenate(dfcol).reshape((dfcol.shape[0], dfcol[0].shape[0]))
    dfhot = pd.DataFrame(index=np.arange(0, df.shape[0]), columns=uifeat, data=arr, dtype='bool')
    return dfhot


def _concat(dftot, dfc):
    # Combine in one big matrix
    if dftot.empty:
        dftot = dfc.copy()
    else:
        for colname in dfc.columns:
            if np.any(np.isin(colname, dftot.columns.values)):
                dftot[colname] = np.logical_or(dftot[colname], dfc[colname])
            else:
                dftot[colname] = dfc[colname].copy()
    return dftot


def dict2df(dfc):
    dftot = pd.DataFrame()
    Iloc = ~dfc.isnull()
    idx = np.where(Iloc)[0]
    idxempty = list(np.where(Iloc==False)[0])

    # result_list = [int(v) for k,v in dfc.iloc[i].items()]
    # pd.DataFrame(data=dfc.iloc[i].items(), index=dfc.iloc[i].keys())

    # Convert dict to dataframe
    for i in idx:
        # try:
            # if isinstance(dict(), type(dfc.iloc[i])):
            #     for key in dfc.iloc[i].keys():
            #         print(key)
            #         isinstance(dict, dfc.iloc[i][key])
            #         pd.DataFrame.from_dict(dfc.iloc[i][key], orient='index')
            # else:
        dftmp = pd.DataFrame.from_dict(dfc.iloc[i], orient='index')
        dftmp.rename(columns={0:i}, inplace=True)
        # Combine into larger dataframe
        dftot = pd.concat([dftot, dftmp], axis=1)
        # except:
        #     idxempty.append(i)

    # Fill the empty ones with None
    if not dftot.empty:
        for i in idxempty:
            dftmp = pd.DataFrame(index=dftot.index.values, data=[None] * len(dftot.index.values), columns=[i])
            dftot = pd.concat([dftot, dftmp], axis=1)

        # Transpose data and sort on index again
        dftot = dftot.T
        dftot.sort_index(inplace=True)
        # Check
        assert np.all(dftot.index.values==dfc.index.values)

    return(dftot, idxempty)


# %%
def _deep_extract(df, dtypes, perc_min_num=None, verbose=3):
    if verbose>=4: print('\n[df2onehot] >Deep extract..')
    # Extract dict
    dftot1, label1, idxrem1 = _extract_dict(df, dtypes, verbose=verbose)
    # Extract lists
    dftot2, label2, idxrem2 = _extract_list(df, dtypes, verbose=verbose)
    # Combine the extracts
    df, dtypes, labels = _extract_combine(df, dtypes, dftot1, dftot2, idxrem1, idxrem2, label1, label2, perc_min_num, verbose=verbose)

    # Return
    if df.shape[1]!=len(dtypes): raise Exception('[df2onehot] >Error: size of dtypes and dataframe does not match.')
    if df.shape[1]!=len(labels): raise Exception('[df2onehot] >Error: size of dtypes and dataframe does not match.')
    return(df, dtypes, labels)


# %%
def _extract_dict(df, dtypes, verbose=3):
    disable = (True if (verbose==0 or verbose>3) else False)
    dfout = pd.DataFrame()
    idxrem = []
    Idict = np.isin(dtypes, 'dict')
    label = []

    # Expand dict
    if np.any(Idict):
        if verbose >=3: print('[df2onehot] >Deep extraction of dictionaries..')
        idxCol = np.where(Idict)[0]
        max_str_len = np.max(list(map(len, df.columns[idxCol].values.astype(str).tolist())))
        # Expand every columns that contains dict
        for idx in tqdm(idxCol, disable=disable):
            makespaces = ''.join([' '] * (max_str_len - len(df.columns[idx])))
            try:
                dfc, idxempty = dict2df(df.iloc[:, idx])
                # dfc = pd.DataFrame.from_records(df.iloc[:,idx])
                # Store the original label
                label = label + [df.columns[idx]] * dfc.shape[1]
                if verbose>=4: print('[df2onehot] >[%s]%s >deep extract > [%s]  [%d]' %(df.columns[idx], makespaces, dtypes[idx], dfc.shape[1]))
            except:
                if verbose>=4: print('[df2onehot] >[%s]%s >deep extract > [failed]' %(df.columns[idx], makespaces))
                # dfc = df.iloc[:,idx].astype(str)
                # dfc = dfc.apply(_remove_non_ascii)

            # Combine extracted columns into big dataframe
            dfout = pd.concat([dfout, dfc], axis=1)
            # Add idx to remove
            idxrem.append(idx)

    return dfout, label, idxrem


# %%
def _extract_list(df, dtypes, verbose=3):
    if verbose >=3: print('[df2onehot] >Deep extraction of lists..')
    disable = (True if (verbose==0 or verbose>3) else False)
    Ilist = np.isin(dtypes, 'list')
    dfout = pd.DataFrame()
    idxrem = []
    label = []

    # Expand list
    if np.any(Ilist):
        idxCol = np.where(Ilist)[0]
        max_str_len = np.max(list(map(len, df.columns[idxCol].values.astype(str).tolist())))
        # Expand every columns that contains either list
        for idx in tqdm(idxCol, disable=disable):
            makespaces = ''.join([' '] * (max_str_len - len(df.columns[idx])))
            # Convert str/float/int to list
            # df, uifeat = _col2type(df, dtypes, idx)
            df.iloc[:, idx], uifeat = _col2type(df.iloc[:, idx], verbose=verbose)

            # Convert column into onehot
            if uifeat is not None:
                dfc = _array2df(df, uifeat, idx)
                # Combine hot-dataframes into one big dataframe
                dfout = _concat(dfout, dfc)
                # Store the original label
                label = label + [df.columns[idx]] * dfc.shape[1]

            # Add idx to remove
            idxrem.append(idx)
            if verbose>=4: print('[df2onehot] >[%s]%s >deep extract > [%s]  [%d]' %(df.columns[idx], makespaces, dtypes[idx], dfc.shape[1]))

    return dfout, label, idxrem


# %%
def _extract_combine(df, dtypes, dftot1, dftot2, idxrem1, idxrem2, label1, label2, perc_min_num, verbose=3):
    if verbose>=5: print('[df2onehot] >Deep extract merging..')
    # Drop columns that are expanded
    idxrem = idxrem1 + idxrem2
    if len(idxrem)>0:
        # Remove the extracted column names from list and dict
        idxkeep = np.setdiff1d(np.arange(0, df.shape[1]), idxrem)
        df.drop(labels = df.columns[idxrem].values, axis=1, inplace=True)
        dtypes = np.array(dtypes)
        dtypes = list(dtypes[idxkeep])
        # Combine the extracted list and dict data
        dftot = pd.concat([dftot1, dftot2], axis=1)
        labeltot = label1 + label2
        # Remove repetative column names
        dftot, labeltot = _make_columns_unique(dftot, labeltot, verbose=verbose)
        # Set dtypes
        dftot, dtypest = set_dtypes(dftot, perc_min_num=perc_min_num, deep_extract=False, verbose=0)
        # Combine into dataframe
        dflabels = df.columns.values
        df = pd.concat([df, dftot], axis=1)
        dtypes = dtypes + dtypest
        labels = list(dflabels) + labeltot
        if verbose>=3: print('\n[df2onehot] >Deep extract extracted: [%d] features.' %(dftot1.shape[1] + dftot2.shape[1]))
    return df, dtypes, labels


# %% Remove repetative column
def _make_columns_unique(dftot, labeltot, verbose=3):
    columns = dftot.columns.value_counts()
    columns = columns[columns.values>1].index.values
    if verbose>=4: print('[df2onehot] >[%d] repetative columns detected: %s' %(len(columns), columns))

    # for column in columns:
    #     dfc = dftot[column]
    #     dfmerged = dfc.stack().groupby(level=0).apply(lambda x: x.unique().tolist())

    _, uiidx = np.unique(dftot.columns, return_index=True)
    uiidx = np.sort(uiidx)
    dftot = dftot.iloc[:, uiidx]
    labeltot = list(np.array(labeltot)[uiidx])

    if len(labeltot)!=dftot.shape[1]: raise Exception('[df2onehot] Error: The total labels and combined dataframe has not same size.')
    return dftot, labeltot


# %% Find columns
def _findcol(x, cols):
    # SLICE COPY WARNING!
    return(np.isin(cols, x))


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
    verbose : int, (default: 3)
        Print message to screen.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('[hnet] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[hnet] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[hnet] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df
