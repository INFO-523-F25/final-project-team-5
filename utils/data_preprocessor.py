# utils/data_preprocessor.py
"""
Contains utility functions for data preprocessing.
"""
import pandas as pd

def read_data(path, **kwargs):
    """
    Read a CSV safely. Pass-through extra pandas.read_csv kwargs.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File not found: "{path}"') from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f'CSV is empty: "{path}"') from e
    except pd.errors.ParserError as e:
        raise ValueError(f'Malformed CSV: "{path}"') from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end,
                                    f'Encoding issue while reading "{path}"')
    except Exception as e:
        raise RuntimeError(f'Unexpected error reading "{path}": {e}') from e


def print_missing(df: pd.DataFrame):
    """
    Print count of missing values per column.
    """
    try:
        for col in df.columns:
            n_miss = df[col].isna().sum()
            print(f'{col} has {n_miss} missing values')
    except Exception as e:
        # except and move on since this function is not critical
        print(f'Encountered unexpected exception: {e}')



