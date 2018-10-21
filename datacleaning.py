import json
import re
import os
import numpy as np
from pandas.io.json import json_normalize

import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from main import hotencode_variables
from utils import *


def preprocess_features(df):
    # V2 Data has substituted nans as strings
    df[df == 'not available in demo dataset'] = np.nan
    df[df == '(not set)'] = np.nan
    df[df == '(not provided)'] = np.nan

    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)  # Variable only contains Falses
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)  # Variable only contains Trues
    df['totals.bounces'].fillna(0, inplace=True)
    df['totals.bounces'] = df['totals.bounces'].astype("int", copy=False)
    df['totals.newVisits'].fillna(0, inplace=True)
    df['totals.newVisits'] = df['totals.newVisits'].astype("int", copy=False)
    df['totals.pageviews'].fillna(0, inplace=True)
    df['totals.pageviews'] = df['totals.pageviews'].astype("int", copy=False)
    df['totals.hits'].fillna(0, inplace=True)
    df['totals.hits'].astype('int', copy=False)

    # Remove silly column?
    if 'trafficSource.campaignCode' in df.columns:
        df.drop(columns=['trafficSource.campaignCode'], inplace=True)


    if 'totals.transactionRevenue' in df.columns:
        df['totals.transactionRevenue'].fillna(0, inplace=True)
        df['totals.transactionRevenue'] = df['totals.transactionRevenue'].astype("float", copy=False)

    return df


def constant_columns(df):
    const_columns = [column for column in df.columns if df[column].nunique(dropna=False) == 1]
    return const_columns


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
    df = preprocess_features(df)
    print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")

    return df

#%%
if __name__== "__main__":
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
    test_df = test_df.drop(columns=const_test)

    #%%
    # Dump data to pickles
    data_to_pickle(train_df, 'data/reduced_train_df.pickle')
    data_to_pickle(test_df, 'data/reduced_test_df.pickle')

    #%%
    train_df = load_pickle('data/reduced_train_df.pickle')
    test_df = load_pickle('data/reduced_test_df.pickle')

    #%%
    # Geographic data cleaning
    # https://www.kaggle.com/mithrillion/fixing-conflicts-in-the-geonetwork-attributes
    geo_colnames = [c for c in train_df.columns if re.match(r'geoNetwork', c) is not None]
    geo_colnames.remove('geoNetwork.networkDomain')

    # Geo categories pre-processing
    for c in geo_colnames:
        train_df[c] = train_df[c].astype('category')

    for c in geo_colnames:
        train_df[c] = train_df[c].cat.add_categories('N/A').fillna('N/A')

    #%%
    train_geo_data = train_df[geo_colnames].copy()

    #%%
    key_attributes = ['geoNetwork.city', 'geoNetwork.region', 'geoNetwork.country']
    train_geo_data_enc = hotencode_variables(train_geo_data[key_attributes])

