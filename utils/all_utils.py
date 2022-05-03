import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml


def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content

def get_data(file_path):
    """
    Reads the data from the file and returns a pandas dataframe.
    """
    return pd.read_csv(file_path)


def remove_unnecessary_columns(df, unnecessary_columns):
    """
    Removes the unnecessary columns from the dataframe.
    """
    df = df.drop(unnecessary_columns , axis=1)
    return df

def missing_columns(df):
    return df.columns[df.isnull().any()].tolist()


def drop_missing_columns(df):
    return df.drop(missing_columns(df), axis=1)


def check_string_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            print(col)
            return True
        else:
            return False

def all_zero(df):
    return df.columns[df.apply(lambda x: x == 0).all()].tolist()

def drop_zero_columns(df):
    return df.drop(all_zero(df), axis=1)


def split_data(df):
    """
    Splits the data into train and test sets.
    """
    x = df.drop(columns=['pIC50'])
    y = pd.DataFrame(df['pIC50'])
    return x, y

def remove_correlated_columns(df, threshold):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)


def minmaxscaler_scaling(x):
    """
    Applies standard scaling on the feature columns.
    """
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    return x


def split_train_test_data(x, y, test_size):
    """
    Splits the data into train and test sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test