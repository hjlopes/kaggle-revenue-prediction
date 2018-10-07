#%%
import os
import json
import pickle
import gc
import logging
gc.enable()

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

import lightgbm as gbm


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


def df_show_value_types(df):
    return df.dtypes.value_counts()


def constant_columns(df):
    const_columns = [column for column in df.columns if df[column].nunique(dropna=False) == 1]
    return const_columns


def data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def factorize_variables(df, excluded_columns=[], cat_indexers=None):
    categorical_features = [
        _f for _f in df.columns
        if (_f not in excluded_columns) & (df[_f].dtype == 'object')
    ]
    logger.info("Categorical features: {}".format(categorical_features))
    if cat_indexers is None:
        cat_indexers = {}
        for f in categorical_features:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in categorical_features:
            df[f] = cat_indexers[f].get_indexer(df[f])

    return df, cat_indexers, categorical_features


def hotencode_variables(df, excluded_columns=[], nan_as_category=False):
    # Encode binary class with Label encoder
    categorical_features = [
        _f for _f in df.columns
        if (_f not in excluded_columns) & (df[_f].dtype == 'object')
    ]

    label_enc = LabelEncoder()
    for column in categorical_features:
        if df[column].dtype == 'object':
            if len(df[column].unique().tolist()) <= 2:
                print(column)
                # df[column] = df[column].fillna('0')
                label_enc.fit(df[column])
                df[column] = label_enc.transform(df[column])
                print("Enconded ", column)

    # Encode multi-class with one Hot-encoder
    df = pd.get_dummies(df, columns=categorical_features)
    return df


def align_train_test(train_df, test_df):
    return train_df.align(test_df, join='inner', axis=1)


def get_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_


### Particular dataset functions


def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


def plot_transaction_revenue(df):
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].astype('float')
    revenuebyuser = df.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(range(revenuebyuser.shape[0]), np.sort(np.log1p(revenuebyuser['totals.transactionRevenue'].values)))
    plt.xlabel('index')
    plt.ylabel('transactionRevenue')
    plt.show()


def preprocess_missings(df):
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)  # Variable only contains Falses
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)  # Variable only contains Trues
    df['totals.bounces'].fillna(0, inplace=True)
    df['totals.bounces'] = df['totals.bounces'].astype("int", copy=False)
    df['totals.newVisits'].fillna(0, inplace=True)
    df['totals.newVisits'] = df['totals.newVisits'].astype("int", copy=False)
    return df


def generate_features(df):
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
    df['sess_date_mon'] = df['date'].dt.month

logger = get_logger()

#%%
if __name__ == "__main__":
    # %%
    # Load initial data
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    #%%
    # Remove Columns with constant values
    const_train = constant_columns(train_df)
    const_test = constant_columns(test_df)
    train_df = train_df.drop(columns=const_train)
    train_df = train_df.drop(columns=['trafficSource.campaignCode'])
    test_df = test_df.drop(columns=const_test)

    #%%
    # Dump data to pickles
    data_to_pickle(train_df, 'data/reduced_train_ndf.pickle')
    data_to_pickle(test_df, 'data/reduced_test_df.pickle')

    #%%
    # Load reduced df
    train_path = 'data/reduced_train_ndf.pickle'
    test_path = 'data/reduced_test_df.pickle'
    train_df = load_pickle(train_path)
    test_df = load_pickle(test_path)

    #%%
    print(missing_values_table(train_df))
    print(missing_values_table(test_df))
    df_show_value_types(train_df)
    df_show_value_types(test_df)
    plot_transaction_revenue(train_df)

    #%%
    generate_features(train_df)
    generate_features(test_df)

    excluded_features = [
        'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue',
        'visitId', 'visitStartTime', 'non_zero_proba'
    ]
    train_df = preprocess_missings(train_df)
    test_df = preprocess_missings(test_df)
    train_df = hotencode_variables(train_df, excluded_columns=excluded_features)
    test_df = hotencode_variables(test_df, excluded_columns=excluded_features)



