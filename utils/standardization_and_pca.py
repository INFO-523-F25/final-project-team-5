import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def standardize_data(df, y):
    '''
    This function standardizes the numerical features of the dataset.
    '''
    # Initializing a List of Numerical Columns
    numeric_columns = df.select_dtypes(include='number').columns

    # Dropping the Response Variable from Standardization
    numeric_columns = numeric_columns.drop(y)

    # Initializing a Variable to Standardize the Numeric Features
    scaler = StandardScaler()

    # Standardizing the Numeric Features
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


def pca_transformation(df, y, n_components=0.95):
    '''
    This function performs PCA on the standardized numerical features of the dataset.
    '''
    # Initializing a List of Numerical Columns
    numeric_columns = df.select_dtypes(include='number').columns

    # Dropping the Response Variable from Standardization
    numeric_columns = numeric_columns.drop(y)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[numeric_columns])

    # Creating a DataFrame with Principal Components
    pca_columns = [f'Principal Component {i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

    # Concatenating with Non-Numeric Columns
    non_numeric_columns = df.select_dtypes(exclude='number').reset_index(drop=True)
    final_df = pd.concat([non_numeric_columns, pca_df, df[y]], axis=1)

    return final_df