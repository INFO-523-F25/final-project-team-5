#utils/data_transformation.py
'''
Contains utility functions related to applying a square root transformation
on our data to fix the positive skew present within our data.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

column_name_mapping = {
    'vix': 'Volatility Index (VIX)',
    'us3m': 'U.S. Treasury 3-Month Bond Yield',
    'epu': 'Economic Policy Uncertainty Index',
    'GPRD': 'Geopolitical Risk Index',
    'Unemployment Percent': 'Unemployment Percent'
}

def log_transform(df, columns):
    try:
        # Creating the figure for all subplots
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
        axes_flat = axes.flatten()

        # Performing the log transformation to correct the positiv skew
        for col in columns:
            # Checking for zeroes before peforming the transformation to prevent ValueError
            if (df[col] == 0).any():
                # Skipping over any zeros
                continue
            # Performing the log transformation
            df[col] = np.log(df[col] + 1)
            
        # Iterating through columns and corresponding axes
        for i, col_name in enumerate(columns):
            # Calculating the starting index for the current row's axes
            ax_qq = axes[i, 0]
            ax_kde = axes[i, 1]

            # Mapping the New Column Names
            new_name = column_name_mapping.get(col_name, col_name)
            
            # Creating the Q-Q Plot
            sm.qqplot(df[col_name], line='s', ax=ax_qq)
            ax_qq.set_title(f'Q-Q Plot: {new_name}')
            
            # Creating the KDE Plot
            sns.kdeplot(df[col_name], fill=True, ax=ax_kde)
            ax_kde.set_title(f'Density: {new_name}')
            ax_kde.set_xlabel('Log-Transformed Value')

        plt.tight_layout()
        plt.show()
        return df

    except Exception as e:
        print(f'An exception has occurred: {e}')
        return None # Return None or raise the exception depending on your needs
