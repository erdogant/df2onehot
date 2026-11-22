"""Various helper functions to set the dtypes."""
# ----------------------------------------------------
# Name        : df2onehot.py
# Author      : E.Taskesen
# github      : https://github.com/erdogant/df2onehot
# Licence     : MIT
# ----------------------------------------------------

# %% Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from tqdm import tqdm
import logging

PACKAGE_LOGGER_NAME = 'df2onehot'
logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# Optional TRACE level
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')


def convert_verbose_to_new(verbose):
    """Convert legacy verbosity values to standardized log-level names."""
    if verbose is None:
        verbose = 0
    if not isinstance(verbose, str) and verbose < 10:
        status_map = {
            0: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'trace',
            6: 'silent',
            'None': 'silent',
        }
        if isinstance(verbose, int) and verbose >= 2:
            logger.warning("Use standardized verbose levels; numeric legacy [1-6] will be deprecated.")
        return status_map.get(verbose, 'silent')
    return verbose


def _coerce_log_level(verbose='info'):
    """Normalize verbosity input to a logging level."""
    verbose = convert_verbose_to_new(verbose)

    if verbose is None or verbose in (0, 'silent', 'off', 'no', 'None'):
        return logging.CRITICAL + 1

    if isinstance(verbose, str):
        name = verbose.strip().lower()
        mapping = {
            'silent': logging.CRITICAL + 1,
            'off': logging.CRITICAL + 1,
            'no': logging.CRITICAL + 1,
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'warn': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'trace': getattr(logging, 'TRACE', 5),
        }
        if name not in mapping:
            raise ValueError(f"[{PACKAGE_LOGGER_NAME}] Unknown log level: {verbose}")
        return mapping[name]

    if isinstance(verbose, int):
        if verbose >= 10:
            return verbose
        legacy = {
            1: logging.CRITICAL,
            2: logging.WARNING,
            3: logging.INFO,
            4: logging.DEBUG,
            5: getattr(logging, 'TRACE', 5),
            6: logging.CRITICAL + 1,
        }
        return legacy.get(verbose, logging.INFO)

    raise TypeError(f"[{PACKAGE_LOGGER_NAME}] verbose must be str, int, or None; got {type(verbose)}")


def _apply_level(name, level):
    """Set *level* on logger *name* and all its existing child loggers."""
    pkg = logging.getLogger(name)
    pkg.setLevel(level)
    for handler in pkg.handlers:
        # Keep handlers permissive; logger level should control filtering.
        handler.setLevel(logging.DEBUG)

    prefix = name + '.'
    for child_name in list(logging.Logger.manager.loggerDict):
        if child_name.startswith(prefix):
            logging.getLogger(child_name).setLevel(level)


def set_logger(verbose: [str, int] = 'info', return_status: bool = False):
    """Set logging level for df2onehot package and its submodules."""
    level = _coerce_log_level(verbose)
    _apply_level(PACKAGE_LOGGER_NAME, level)

    if return_status:
        return level


def get_logger():
    """Return the package logger."""
    return logging.getLogger(PACKAGE_LOGGER_NAME)

# %% Set dtypes
def set_dtypes(df, dtypes='pandas', deep_extract=False, perc_min_num=None, num_if_decimal=True, verbose=None):
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
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    tuple containing dataframe and dtypes.

    """
    config = {}
    config['dtypes'] = dtypes
    config['deep_extract'] = deep_extract
    config['perc_min_num'] = perc_min_num
    config['num_if_decimal'] = num_if_decimal
    if verbose is not None:
        logger.warning(f"Verbose is depricated as input parameter. This must be set using the logger.")

    # Determine dtypes for columns
    config['dtypes'] = _auto_dtypes(df, config['dtypes'], deep_extract=config['deep_extract'], perc_min_num=config['perc_min_num'], num_if_decimal=config['num_if_decimal'])
    # Setup dtypes in columns
    df = _set_types(df.copy(), config['dtypes'])
    # return
    return(df, config['dtypes'])


# %% Setup columns in correct dtypes
def _auto_dtypes(df, dtypes, deep_extract=False, perc_min_num=None, num_if_decimal=True, verbose=None):
    if verbose is not None:
        logger.warning(f"Verbose is depricated as input parameter. This must be set using the logger.")

    if isinstance(dtypes, str):
        logger.info(f"Auto detecting dtypes.")
        max_str_len = np.max(list(map(len, df.columns.values.astype(str).tolist())))
        dtypes = [''] * df.shape[1]
        logstr = '   '

        for i in tqdm(range(0, df.shape[1]), disable=disable_tqdm(), desc="[df2onehot]"):
            if 'float' in str(df.dtypes.iloc[i]):
                dtypes[i]='num'
                logstr = ('[float]')
            elif 'int' in str(df.dtypes.iloc[i]):
                # logstr = (' > [integer]: Set to categorical. Uniqueness=%.2f' %(df.iloc[:,i].unique().shape[0]/df.shape[0]))
                dtypes[i]='cat'
                logstr = ('[int]  ')
            elif 'str' in str(df.dtypes.iloc[i]):
                dtypes[i]='cat'
                logstr = ('[str]  ')
            elif ('object' in str(df.dtypes.iloc[i])) and not deep_extract:
                dtypes[i]='cat'
                logstr = ('[obj]  ')
            elif 'object' in str(df.dtypes.iloc[i]) and deep_extract:
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
            elif 'bool' in str(df.dtypes.iloc[i]):
                dtypes[i]='bool'
                logstr = ('[bool]  ')
            else:
                dtypes[i]='cat'
                logstr = ('[???]  ')

            # Force numerical if unique elements are above percentage
            if (perc_min_num is not None) and (('float' in str(df.dtypes.iloc[i])) or ('int' in str(df.dtypes.iloc[i]))):
                tmpvalues = df.iloc[:,i].dropna().astype(float).copy()
                perc=0
                if len(tmpvalues)>0:
                    perc = (len(np.unique(tmpvalues)) / len(tmpvalues))
                if (perc>=perc_min_num):
                    dtypes[i]='num'
                    logstr = ('[force]')
                    # logstr=' > [numerical]: Uniqueness %.2f>=%.2f' %((df.iloc[:,i].unique().shape[0]/df.shape[0]), perc_min_num)

            # Force numerical if values are found with decimals
            if num_if_decimal and (('float' in str(df.dtypes.iloc[i])) or ('int' in str(df.dtypes.iloc[i]))):
                tmpvalues = df.iloc[:, i].dropna().copy()
                if np.any(tmpvalues.astype(int) - tmpvalues.astype(float) > 0):
                    dtypes[i] = 'num'
                    logstr = ('[force]')

            # Remove the non-ascii chars from categorical values
            if dtypes[i]=='cat':
                df.iloc[:,i] = _remove_non_ascii(df.iloc[:,i])

            try:
                makespaces = ''.join([' '] * (max_str_len - len(df.columns[i])))
                logger.debug(f"[{df.columns[i]}]{makespaces} > {logstr} > [{dtypes[i]}] [{len(df.iloc[:,i].dropna().unique())}]")
            except:
                logger.debug(f"[{df.columns[i]}]{makespaces} > {logstr} > [{dtypes[i]}] [{len(df.iloc[:,i].dropna())}]")

    # assert len(dtypes)==df.shape[1], 'Length of dtypes and dataframe columns does not match'
    return(dtypes)


# %% Setup columns in correct dtypes
def _set_types(df, dtypes):
    assert len(dtypes)==df.shape[1], 'Number of dtypes and columns in df does not match'
    logger.debug(f"Set dtypes in dataframe..")
    max_str_len = np.max(list(map(len, df.columns.values.astype(str).tolist()))) + 2

    # remcols=[]
    for col, dtype in zip(df.columns, dtypes):
        makespaces = ''.join([' '] * (max_str_len - len(col)))
        logger.debug(f"{col}")

        if dtype=='num':
            df[col]=df[col].astype(float)
        elif dtype=='cat':
            Inull = df[col].isna().values
            df.loc[Inull, col] = None
            df[col] = df[col].astype(str)
            # df[col] = df[col].astype('category')
        elif dtype=='bool':
            Inull = df[col].isna().values
            if Inull.any():
                # Use pandas nullable boolean dtype to support missing values
                df[col] = df[col].astype("boolean")
                df.loc[Inull, col] = pd.NA
            else:
                # No missing values: regular bool dtype is fine
                df[col] = df[col].astype(bool)
        else:
            logger.debug(f"[{col}] {makespaces} > deep extract > [{dtype}]")

    return(df)


# %% Set y
def set_y(y, y_min=None, numeric=False, verbose=None):
    """Group labels if required.

    Parameters
    ----------
    y : list
        input labels.
    y_min : int, optional
        If unique y-labels are less then absolute y_min, labels are grouped into the _other_ group. The default is None.
    numeric : bool [True, False], optional
        Convert to numeric labels. The default is False.

    Returns
    -------
    list of labels.

    """
    y = y.astype(str)

    if not isinstance(y_min, type(None)):
        logger.debug(f"Group [y] labels that contains less than {y_min} occurrences are grouped under one single name [_other_]")
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
def is_DataFrame(data):
    """Convert data into dataframe.

    Parameters
    ----------
    data : array-like
        Array-like data matrix.

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
        logger.info("Typing should be pd.DataFrame()!")
        data=None

    return(data)


def disable_tqdm():
    """Return True when tqdm progress bars should be disabled (level >= WARNING)."""
    return get_logger().getEffectiveLevel() >= logging.WARNING
