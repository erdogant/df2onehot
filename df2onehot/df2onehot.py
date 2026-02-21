"""Convert dataframe to one-hot matrix."""
# ----------------------------------------------------
# Name        : df2onehot.py
# Author      : E.Taskesen
# github      : https://github.com/erdogant/df2onehot
# Licence     : MIT
# ----------------------------------------------------

# %% Libraries
from packaging import version
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import datazets as dz

try:
    from df2onehot.utils import set_dtypes
except:
    from utils import set_dtypes

logger = logging.getLogger(__name__)

# Make checks
import sklearn
if version.parse(sklearn.__version__) < version.parse('1.4.0'):
    logger.warning(f"For best results update scikit-learn >= 1.4.0. pip install -U scikit-learn")
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
else:
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')

# Set the label encoder
label_encoder = LabelEncoder()
warnings.filterwarnings('ignore')


# %% Dataframe to one-hot
def df2onehot(df,
              dtypes='pandas',
              y_min=2,
              perc_min_num=None,
              hot_only=True,
              deep_extract=False,
              excl_background=None,
              remove_mutual_exclusive=False,
              remove_multicollinearity=False,
              verbose: str | int = 'info'):
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
        This parameters can be used to force variables into numeric ones if unique non-zero values are above the percentage. The default is None. Alternative can be 0.8
    hot_only : bool [True, False], optional
        When True; the output of the onehot matrix exclusively contains categorical values that are transformed to one-hot. The default is True.
    deep_extract : bool [False, True] (default : False)
        True: Extract information from a vector that contains a list/array/dict.
        False: converted to a string and treated as catagorical ['cat'].
    remove_mutual_exclusive: bool [False, True] (default : False)
        True: Remove the mutual exclusive groups. In binairy features; False and 0 are excluded.
        False: Do nothing
    remove_multicollinearity: bool [False, True] (default : False)
        True: Remove multicollinear columns by removing one columns for each catagory that is converted into onehot.
        False: Do nothing
    excl_background : list or None, [0], [0, '0.0', 'unknown', 'nan', 'None' ...], optional
        Remove values/strings that labeled in the list. As an example, the following column: ['yes', 'no', 'yes', 'yes','no','unknown', ...], is split into 'column_yes', 'column_no' and 'column_unknown'. If unknown listed, then 'column_unknown' is not transformed into a new one-hot column.
        The default is None (every possible name is converted into a one-hot column)
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    dict:
    numeric : DataFrame
        Input-dataframe with converted numerical values
    onehot : DataFrame
        Input-dataframe with converted one-hot values. Note that continuous values are only removed if hot_only=True.
    labx : list of str
        Input feature-labels or names
    df : DataFrame
        Input-dataframe but with set dtypes. Note that df is extended if deep_extract=True
    labels : list of str
        Column names of df
    dtypes : list of str
        dtypes for the feature-labels for df in the form of 'num' (numerical) and/or 'cat' (categorical).

    Examples
    --------
    >>> import df2onehot
    >>> df = df2onehot.import_example()
    >>> out = df2onehot.df2onehot(df)

    """
    # Set the logger
    set_logger(verbose=verbose)

    args = {}
    args['dtypes'] = dtypes
    args['verbose'] = verbose
    args['perc_min_num'] = perc_min_num
    args['deep_extract'] = deep_extract
    args['excl_background'] = excl_background
    labx = []
    labels = None

    if len(np.unique(df.columns))!=len(df.columns):
        logger.warning(f"The column labels must be unique.")
        df.columns = make_elements_unique(df.columns.values)

    # Reset index
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    # Determine Dtypes
    df, dtypes = set_dtypes(df, args['dtypes'], deep_extract=args['deep_extract'], perc_min_num=args['perc_min_num'])
    # If any column is a list, also expand the list!
    if args['deep_extract']:
        df, dtypes, labels = _deep_extract(df, dtypes, perc_min_num=args['perc_min_num'], verbose=args['verbose'])

    # Make empty frames
    maxstring=50
    out_numeric = pd.DataFrame()
    # @jjaycez: Fix to avoid repeatedly using pd.concat in the main loop
    out_onehot_parts = []
    out_onehot = pd.DataFrame()
    max_str_len = np.minimum(np.max(list(map(len, df.columns.values.astype(str).tolist()))) + 2, maxstring)

    # Run over all columns
    for i in tqdm(np.arange(0, df.shape[1]), disable=disable_tqdm(), desc="[df2onehot]"):
        makespaces = ''.join(['.'] * np.minimum( (max_str_len - len(df.columns[i])), maxstring) )
        # Do not touch a float
        if 'float' in str(df.dtypes.iloc[i]):
            logger.debug(f"Processing: {df.columns[i][0:maxstring]}{makespaces} [float]")
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
            logger.debug(f"Processing: {df.columns[i][0:maxstring]}{makespaces} [{len(np.unique(integer_encoded))}]")

            # Remove mutual exclusive values
            status_bool=False
            if (remove_mutual_exclusive or remove_multicollinearity) and len(np.unique(integer_encoded))==2:
                if np.isin(np.unique(integer_encoded), [0, 1]).sum()>=2:
                    status_bool=True

            # Contains a single value or is bool
            if status_bool:
                logger.info(f"Remove mutual exclusive for [{df.columns[i]}]")
                label = df.columns[i] + '_' + str(df.iloc[integer_encoded==1, i].values[0])
                temp_df = pd.DataFrame({label: integer_encoded.astype('bool')}, index=df.index)
                out_onehot_parts.append(temp_df)
                labx.append(label)
            elif (len(np.unique(integer_encoded))<=1) or (str(df.dtypes.iloc[i])=='bool'):
                temp_df = pd.DataFrame({df.columns[i]: integer_encoded.astype('bool')}, index=df.index)
                out_onehot_parts.append(temp_df)
                labx.append(df.columns[i])
            else:
                # binary encode
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
                total_columns = onehot_encoded.shape[1]
                # Remove columns if it does not fullfill minimum nr. of samples (>=y_min)
                if y_min is not None:
                    onehot_encoded = onehot_encoded[:, onehot_encoded.sum(axis=0) >= y_min]
                # Remove a columns in case to prevent multicollinearity. If a column was already removed for some reason. Do not touch.
                if remove_multicollinearity and onehot_encoded.shape[1]==total_columns:
                    logger.info(f"Remove multicollinearity for [{df.columns[i]}]")
                    onehot_encoded = onehot_encoded[:, 1:]

                # Make new one-hot columns
                # @jjaycez Avoided warning about fragmented array by using a dictionary to store the columns and then
                #   make a dataframe from the dictionary.
                temp_cols = {}
                for k in range(0, onehot_encoded.shape[1]):
                    label = df.iloc[onehot_encoded[:, k] == 1, i].unique().astype(str)[0]
                    if (isinstance(args['excl_background'], type(None))) or (
                    not np.isin(label, args['excl_background'])):
                        # @jjaycez: It seems that one of the factors may return as a float, so ensuring string type...
                        colname = str(df.columns[i]) + '_' + str(label)
                        temp_cols[colname] = onehot_encoded[:, k].astype('bool')
                        labx.append(df.columns[i])

                if temp_cols:
                    temp_df = pd.DataFrame(temp_cols, index=df.index)
                    out_onehot_parts.append(temp_df)

                # Make numerical vector
                if onehot_encoded.shape[1]>2:
                    out_numeric[df.columns[i]] = (onehot_encoded * np.arange(1, onehot_encoded.shape[1] + 1)).sum(axis=1)
    # @jjaycez: Single pd.concat outside of main loop
    out_onehot = pd.concat(out_onehot_parts, axis=1) if out_onehot_parts else pd.DataFrame(index=df.index)

    uiy, ycounts = np.unique(labx, return_counts=True)
    logger.info(f"Total onehot features: {np.sum(ycounts)}")
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
    # If any str, float or int elements is fount, convert to list
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
        logger.error(f"Error catched.")
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

    # Convert dict to dataframe
    for i in idx:
        dftmp = pd.DataFrame.from_dict(dfc.iloc[i], orient='index')
        dftmp.rename(columns={0:i}, inplace=True)
        # Combine into larger dataframe
        dftot = pd.concat([dftot, dftmp], axis=1)

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
    logger.debug(f"\nDeep extract..")
    # Extract dict
    dftot1, label1, idxrem1 = _extract_dict(df, dtypes, verbose=verbose)
    # Extract lists
    dftot2, label2, idxrem2 = _extract_list(df, dtypes, verbose=verbose)
    # Combine the extracts
    df, dtypes, labels = _extract_combine(df, dtypes, dftot1, dftot2, idxrem1, idxrem2, label1, label2, perc_min_num, verbose=verbose)

    # Return
    if df.shape[1]!=len(dtypes): raise Exception('Error: size of dtypes and dataframe does not match.')
    if df.shape[1]!=len(labels): raise Exception('Error: size of dtypes and dataframe does not match.')
    return df, dtypes, labels


# %%
def _extract_dict(df, dtypes, verbose=3):
    dfout = pd.DataFrame()
    idxrem = []
    Idict = np.isin(dtypes, 'dict')
    label = []

    # Expand dict
    if np.any(Idict):
        logger.info(f"Deep extraction of dictionaries..")
        idxCol = np.where(Idict)[0]
        max_str_len = np.max(list(map(len, df.columns[idxCol].values.astype(str).tolist())))
        # Expand every columns that contains dict
        for idx in tqdm(idxCol, disable=disable_tqdm(), desc="[df2onehot]"):
            makespaces = ''.join([' '] * (max_str_len - len(df.columns[idx])))
            try:
                dfc, idxempty = dict2df(df.iloc[:, idx])
                # Store the original label
                label = label + [df.columns[idx]] * dfc.shape[1]
                logger.debug(f"[{df.columns[idx]}]{makespaces} >deep extract > [{dtypes[idx]}]  [{dfc.shape[1]}]")
            except:
                logger.debug(f"[{df.columns[idx]}]{makespaces} >deep extract > [failed]")

            # Combine extracted columns into big dataframe
            dfout = pd.concat([dfout, dfc], axis=1)
            # Add idx to remove
            idxrem.append(idx)

    return dfout, label, idxrem


# %%
def _extract_list(df, dtypes, verbose=3):
    logger.info(f"Deep extraction of lists..")
    Ilist = np.isin(dtypes, 'list')
    dfout = pd.DataFrame()
    idxrem = []
    label = []

    # Expand list
    if np.any(Ilist):
        idxCol = np.where(Ilist)[0]
        max_str_len = np.max(list(map(len, df.columns[idxCol].values.astype(str).tolist())))
        # Expand every columns that contains either list
        for idx in tqdm(idxCol, disable=disable_tqdm(), desc='[df2onehot]'):
            makespaces = ''.join([' '] * (max_str_len - len(df.columns[idx])))
            # Convert str/float/int to list
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
            logger.debug(f"[{df.columns[idx]}]{makespaces} >deep extract > [{dtypes[idx]}]  [{dfc.shape[1]}]")

    return dfout, label, idxrem


# %%
def _extract_combine(df, dtypes, dftot1, dftot2, idxrem1, idxrem2, label1, label2, perc_min_num, verbose=3):
    if logger.debug: print('Deep extract merging..')
    # Drop columns that are expanded
    labels = df.columns.values
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
        dftot, dtypest = set_dtypes(dftot, perc_min_num=perc_min_num, deep_extract=False)
        # Combine into dataframe
        dflabels = df.columns.values
        df = pd.concat([df, dftot], axis=1)
        dtypes = dtypes + dtypest
        labels = list(dflabels) + labeltot
        logger.info('\nDeep extract extracted: [%d] features.' %(dftot1.shape[1] + dftot2.shape[1]))
    return df, dtypes, labels


# %% Remove repetative column
def _make_columns_unique(dftot, labeltot, verbose=3):
    columns = dftot.columns.value_counts()
    columns = columns[columns.values>1].index.values
    logger.debug(f"[{len(columns)}] repetative columns detected: {columns}")

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
def import_example(data='titanic', url=None, sep=',', overwrite=False):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump'
    url : str
        url link to to dataset.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if data=='complex':
        df = pd.DataFrame(index=np.arange(0, 25))
        df['feat_1'] = np.nan
        df['feat_1'].iloc[0] = ['3', 4]
        df['feat_1'].iloc[2] = ['5', '6', '7', '8']
        df['feat_1'].iloc[20] = ['9', '11', '4']
        df['feat_1'].iloc[5] = 10
        df['feat_1'].iloc[15] = 1
        df['feat_2'] = np.nan
        df['feat_2'].iloc[0] = ['4', '45']
        df['feat_2'].iloc[15] = 1
        df['feat_2'].iloc[20] = 10
    else:
        df = dz.get(data=data, url=url, sep=sep, overwrite=overwrite)

    return df


# %%
def make_elements_unique(X):
    uix, counts = np.unique(X, return_counts=True)
    idx = np.where(counts>1)[0]
    for i in idx:
        Iloc = uix[i]==X
        lst = X[Iloc]
        X[Iloc] = [f"{element}_{index+1}" for index, element in enumerate(lst)]
    return X


# %%
def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[df2onehot] WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose

def get_logger():
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info', return_status: bool = False):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)

    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert verbose to numeric level
    if verbose in (0, None):
        log_level = logging.CRITICAL + 10  # silent
    elif isinstance(verbose, str):
        levels = {
            'silent': logging.CRITICAL + 10,
            'off': logging.CRITICAL + 10,
            'no': logging.CRITICAL + 10,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        log_level = levels.get(verbose.lower(), logging.INFO)
    elif isinstance(verbose, int):
        log_level = verbose
    else:
        log_level = logging.INFO

    # Set package logger
    logger = logging.getLogger('XXX')
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if return_status:
        return log_level



def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)

# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"No commandline arguments possible.")
