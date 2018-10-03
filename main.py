#%%
import os
import json

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler


def load_csv(path, nrows=None):
    columns = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(path,
                     converters={column: json.loads for column in columns},
                     dtype={'fullVisitorId': 'str'},
                     nrows=nrows)

    for column in columns:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")

    return df


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def df_value_types(df):
    return df.dtypes.value_counts()


def constant_columns(df):
    const_columns = [column for column in df.columns if df[column].nunique(dropna=False) == 1]
    return const_columns


def plot_transaction_revenue(df):
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].astype('float')
    revenuebyuser = df.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(range(revenuebyuser.shape[0]), np.sort(np.log1p(revenuebyuser['totals.transactionRevenue'].values)))
    plt.xlabel('index')
    plt.ylabel('transactionRevenue')
    plt.show()


#%%
if __name__ == "__main__":
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    # Columns with constant values
    const_train = constant_columns(train_df)
    const_test = constant_columns(test_df)
    train_df = train_df.drop(columns=const_train)
    test_df = test_df.drop(columns=const_test)
    #%%
    print(missing_values_table(train_df))
    print(missing_values_table(test_df))
    df_value_types(train_df)
    df_value_types(test_df)
    plot_transaction_revenue(train_df)


    #%%


