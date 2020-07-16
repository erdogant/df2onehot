"""Various helper functions to set the dtypes."""
# ----------------------------------------------------
# Name        : df2onehot.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/df2onehot
# Licence     : MIT
# ----------------------------------------------------

# %% Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from tqdm import tqdm


# %% Set dtypes
def set_dtypes(df, dtypes='pandas', deep_extract=False, perc_min_num=None, num_if_decimal=True, verbose=3):
    """Set the dtypes of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe for which the rows are the features, and colums are the samples.
    dtypes : list of str or 'pandas', optional
        Representation of the columns in the form of ['cat','num']. By default the dtype is determiend based on the pandas dataframe.
    deep_extract : bool [False, True] (default : False)
        True: Extract information from a vector that contains a list/array/dict.
        False: converted to a string and treated as catagorical ['cat'].
    perc_min_num : float [None, 0..1], optional
        Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8
    num_if_decimal : bool [False, True], optional
        Force column to be numerical if column with original dtype (int or float) show values with one or more decimals. The default is True.
    verbose : int, optional
        Print message to screen. The default is 3.
        0: (default), 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE


    Returns
    -------
    tuple containing dataframe and dtypes.

    """
    config = {}
    config['dtypes'] = dtypes
    config['deep_extract'] = deep_extract
    config['perc_min_num'] = perc_min_num
    config['num_if_decimal'] = num_if_decimal
    config['verbose'] = verbose

    # Determine dtypes for columns
    config['dtypes'] = _auto_dtypes(df, config['dtypes'], deep_extract=config['deep_extract'], perc_min_num=config['perc_min_num'], num_if_decimal=config['num_if_decimal'], verbose=config['verbose'])
    # Setup dtypes in columns
    df = _set_types(df.copy(), config['dtypes'], verbose=config['verbose'])
    # return
    return(df, config['dtypes'])


# %% Setup columns in correct dtypes
def _auto_dtypes(df, dtypes, deep_extract=False, perc_min_num=None, num_if_decimal=True, verbose=3):
    if isinstance(dtypes, str):
        if verbose>=3: print('\n[df2onehot] >Auto detecting dtypes.')
        disable = (True if (verbose==0 or verbose>3) else False)
        max_str_len = np.max(list(map(len, df.columns.values.astype(str).tolist())))
        dtypes = [''] * df.shape[1]
        logstr = '   '

        for i in tqdm(range(0, df.shape[1]), disable=disable):
            if 'float' in str(df.dtypes[i]):
                dtypes[i]='num'
                logstr = ('[float]')
            elif 'int' in str(df.dtypes[i]):
                # logstr = (' > [integer]: Set to categorical. Uniqueness=%.2f' %(df.iloc[:,i].unique().shape[0]/df.shape[0]))
                dtypes[i]='cat'
                logstr = ('[int]  ')
            elif 'str' in str(df.dtypes[i]):
                dtypes[i]='cat'
                logstr = ('[str]  ')
            elif ('object' in str(df.dtypes[i])) and not deep_extract:
                dtypes[i]='cat'
                logstr = ('[obj]  ')
            elif 'object' in str(df.dtypes[i]) and deep_extract:
                # Check whether this is a list or array
                logstr = ('[obj]  ')
                tmpdf = df.iloc[:, i]
                Iloc = ~tmpdf.isna()
                if np.any(Iloc):
                    tmpdf = tmpdf.loc[Iloc].values[0]
                else:
                    tmpdf = None
                if isinstance(list(), type(tmpdf)):
                    dtypes[i]='list'
                elif 'numpy.ndarray' in str(type(tmpdf)):
                    dtypes[i]='list'
                elif isinstance(dict(), type(tmpdf)):
                    dtypes[i]='dict'
                else:
                    dtypes[i]='cat'
            elif 'bool' in str(df.dtypes[i]):
                dtypes[i]='bool'
                logstr = ('[bool]  ')
            else:
                dtypes[i]='cat'
                logstr = ('[???]  ')
            
            # Force numerical if unique elements are above percentage
            if (perc_min_num is not None) and (('float' in str(df.dtypes[i])) or ('int' in str(df.dtypes[i]))):
                tmpvalues = df.iloc[:,i].dropna().astype(float).copy()
                perc=0
                if len(tmpvalues)>0:
                    perc = (len(np.unique(tmpvalues)) / len(tmpvalues))
                if (perc>=perc_min_num):
                    dtypes[i]='num'
                    logstr = ('[force]')
                    # logstr=' > [numerical]: Uniqueness %.2f>=%.2f' %((df.iloc[:,i].unique().shape[0]/df.shape[0]), perc_min_num)

            # Force numerical if values are found with decimals
            if num_if_decimal and (('float' in str(df.dtypes[i])) or ('int' in str(df.dtypes[i]))):
                tmpvalues = df.iloc[:, i].dropna().copy()
                if np.any(tmpvalues.astype(int) - tmpvalues.astype(float) > 0):
                    dtypes[i] = 'num'
                    logstr = ('[force]')

            # Remove the non-ascii chars from categorical values
            if dtypes[i]=='cat':
                df.iloc[:,i] = _remove_non_ascii(df.iloc[:,i])

            try:
                makespaces = ''.join([' '] * (max_str_len - len(df.columns[i])))
                if verbose>=4: print('[df2onehot] >[%s]%s > %s > [%s] [%.0d]' %(df.columns[i], makespaces, logstr, dtypes[i], len(df.iloc[:,i].dropna().unique())))
            except:
                if verbose>=4: print('[df2onehot] >[%s]%s > %s > [%s] [%.0d]' %(df.columns[i], makespaces, logstr, dtypes[i], len(df.iloc[:,i].dropna())))

    # assert len(dtypes)==df.shape[1], 'Length of dtypes and dataframe columns does not match'
    return(dtypes)


# %% Setup columns in correct dtypes
def _set_types(df, dtypes, verbose=3):
    assert len(dtypes)==df.shape[1], 'Number of dtypes and columns in df does not match'
    if verbose>=3: print('[df2onehot] >Set dtypes in dataframe..')
    max_str_len = np.max(list(map(len, df.columns.values.astype(str).tolist()))) + 2

    # remcols=[]
    for col, dtype in zip(df.columns, dtypes):
        makespaces = ''.join([' '] * (max_str_len - len(col)))
        if verbose>=4: print('[df2onehot] >%s' %(col))

        if dtype=='num':
            df[col]=df[col].astype(float)
        elif dtype=='cat':
            Inull = df[col].isna().values
            df[col].loc[Inull] = None
            df[col] = df[col].astype(str)
            # df[col] = df[col].astype('category')
        elif dtype=='bool':
            Inull = df[col].isna().values
            df[col].loc[Inull] = None
            df[col] = df[col].astype(bool)
        else:
            if verbose>=5: print('[df2onehot] >[%s] %s > deep extract > [%s]' %(col, makespaces, dtype))

    return(df)


# %% Set y
def set_y(y, y_min=None, numeric=False, verbose=3):
    """Group labels if required.

    Parameters
    ----------
    y : list
        input labels.
    y_min : int, optional
        If unique y-labels are less then absolute y_min, labels are grouped into the _other_ group. The default is None.
    numeric : bool [True, False], optional
        Convert to numeric labels. The default is False.
    verbose : int, optional
        Print message to screen. The default is 3.
        0: (default), 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    list of labels.

    """
    y = y.astype(str)

    if not isinstance(y_min, type(None)):
        if verbose>=3: print('[df2onehot] >Group [y] labels that contains less then %d occurences are grouped under one single name [_other_]' %(y_min))
        [uiy, ycounts] = np.unique(y, return_counts=True)
        labx = uiy[ycounts<y_min]
        y = y.astype('O')
        y[np.isin(y, labx)] = '_other_'  # Note that this text is captured in compute_significance! Do not change or also change it over there!
        y = y.astype(str)

        if numeric:
            y = label_encoder.fit_transform(y).astype(int)

    return(y)

# %% function to remove non-ASCII
def _remove_non_ascii(dfc):
    # Get the current dtype
    dftype = dfc.dtype
    # Set as string
    dfc = dfc.astype('str')
    # Find the nans
    Iloc = ~( (dfc.str.lower()=='nan') | (dfc.str.lower()=='none') | dfc.isnull() )
    # Remove non-ascii chars
    dfc.loc[Iloc] = np.array(list(map(lambda x: str(x).encode('ascii','ignore').decode('ascii','ignore').strip(), dfc.loc[Iloc])))
    dfc.loc[Iloc] = np.array(list(map(lambda x: str(x).encode('unicode_escape').decode('ascii','ignore').strip(), dfc.loc[Iloc])))
    # dfc.loc[Iloc] = dfc.loc[Iloc].replace(r'\W+', ' ', regex=True)
    dfc.loc[Iloc] = dfc.loc[Iloc].replace('[^\x00-\x7F]', ' ')
    # Set the None back    
    dfc.loc[~Iloc] = None
    # Bring back to origial dtype
    dfc = dfc.astype(dftype)
    # Return
    return dfc

# %% Convert to pandas dataframe
def is_DataFrame(data, verbose=3):
    """Convert data into dataframe.

    Parameters
    ----------
    data : array-like
        Array-like data matrix.
    verbose : int, optional
        Print message to screen. The default is 3.
        0: (default), 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    pd.dataframe()

    """
    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        if verbose>=3: print('Typing should be pd.DataFrame()!')
        data=None

    return(data)
