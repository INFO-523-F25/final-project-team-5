#utils/data_wrangling.py
'''
Contains utility functions related to data cleaning and wrangling steps.
More specifically, these functions will adjust column types, unpivot data,
map descriptive column names, and merge the DataFrames together.
'''

import pandas as pd

def date_transform(df, col):
    '''
    Converts a column into a datetime data type and extracts year and month information
    from the newly converted column.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that the date transformation will be applied to.
    col (pd.Series):
        The date variable that will be transformed.

    Results
    ------------
    df (DataFrame):
        The DataFrame with the transformed date column.
    '''
    try:
        # Converting the column to a datetime type
        df[col] = pd.to_datetime(df[col])
        try:
            #Extracting the Year and Month information from the column
            df['Year'] = df[col].dt.year
            df['Month'] = df[col].dt.month
            return df
        except ValueError:
            print(f'Error converting {col} to type integer.')
    except KeyError:
        print(f'{col} is not a valid column within the DataFrame.')
    except ValueError:
        print(f'Error converting {col} to type datetime.')
    return df

def unpivot_df(df, id_vars = 'Year', var_name = 'Month', value_name = 'Unemployment Percent'):
    '''
    Takes a DataFrame in the long format and unpivots the columns to the rows.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that will be unpivoted.
    id_vars (pd.Series):
        The ID variable of the DataFrame for the purpose of unpivoting the data.
    var_name (str):
        The name that will be applied to the variable column.
    value_name (str):
        The name that will be applied to the value column.

    Results
    ------------
    df (DataFrame):
        The DataFrame that has been unpivoted.
    '''
    try:
        # Unpivoting the DataFrame
        df = df.melt(id_vars = id_vars, var_name = var_name, value_name = value_name)
        return df
    except KeyError as e:
        print(f'An unexpected error occured: {e}.')
    # return df

# Initializing a Dictionary which holds the month names and their respctive integer counterpart
month_replacement = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def mapping_month_names(df, month):
    '''
    Takes a column of months in the integer format and converts them to a short text format.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that will obtain the mapped month names.
    month (pd.Series):
        The month variable that will be given new names.

    Results
    ------------
    df (DataFrame):
        The DataFrame with the transformed month column.
    '''
    try:
        # Mapping the new text column headers to the existing integer column headers
        df[month] = df[month].map(month_replacement)
        return df
    except KeyError:
        print(f'{month} is not a valid column within the DataFrame.')

def merge_dfs(df1, df2, on = ['Year', 'Month'], how = 'left'):
    '''
    Joins two DataFrames together into one via a left join.

    Parameters
    ------------
    df1 (pd.DataFrame):
        The first DataFrame that will be joined.
    df2 (pd.DataFrame):
        The second DataFrame that will be joined.
    on (list):
        The list of value(s) that the DataFrames will be joined on.
    how (str):
        The type of join that the table will use (i.e. left, right, outer, etc.)

    Results
    ------------
    df (DataFrame):
        The DataFrame that has been created as a result of df1 and df2 joining together.
    '''
    try:
        # Merging the two DataFrames together
        df = pd.merge(df1, df2, on = on, how = how)
        return df
    except KeyError as e:
        print(f'An unexpected error occured: {e}.')
    # return df