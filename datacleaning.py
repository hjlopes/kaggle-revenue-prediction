import gc
import json
import re
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

from sklearn.preprocessing import LabelEncoder

gc.enable()

from utils import *

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import random

from sklearn.ensemble import RandomForestClassifier


def hotencode_variables(df, excluded_columns=[], nan_as_category=False):
    # Encode binary class with Label encoder
    categorical_features = [
        _f for _f in df.columns
        if (_f not in excluded_columns) & (df[_f].dtype in ['object', 'category'])
    ]

    label_enc = LabelEncoder()
    for column in categorical_features:
        if df[column].dtype in ['object', 'category']:
            if len(df[column].unique().tolist()) <= 2:
                print(column)
                # df[column] = df[column].fillna('0')
                label_enc.fit(df[column])
                df[column] = label_enc.transform(df[column])
                print("Enconded ", column)

    # Encode multi-class with one Hot-encoder
    df = pd.get_dummies(df, columns=categorical_features)
    return df


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
    df['totals.hits'] = df['totals.hits'].astype('int', copy=False)

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
    json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ignore_columns = ['hits']

    # Set the DF columns
    cols = pd.read_csv(path, nrows=0).columns
    df = pd.DataFrame(columns=cols)

    chunk = pd.read_csv(path,
                     converters={column: json.loads for column in json_columns},
                     dtype={'fullVisitorId': 'str'},
                     nrows=nrows, chunksize=200000)

    for idx, df_chunk in enumerate(chunk):
        logger.info("Processing chunk #{}".format(idx))

        # Unpack JSON  columns
        df_chunk.reset_index(drop=True, inplace=True)
        for column in json_columns:
            column_as_df = json_normalize(df_chunk[column])
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            df_chunk = df_chunk.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        df_chunk.drop(columns=ignore_columns, inplace=True, errors='ignore')  # Drop unused columns

        df = pd.concat([df, df_chunk], axis=0, ignore_index=True, sort=True)  # Merge the chunk with the master DF

        del df_chunk  # Memory save

    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")
    return df


def data_clean_and_reduce(df, dataset_name='train', geo_fix=False):
    #%%
    # Dump data to pickles
    reduced_data_path = 'data/reduced_{}_df.pickle'.format(dataset_name)
    df.to_pickle(reduced_data_path)
    logger.info("Finished saving reduced {} pkl data".format(dataset_name))

    if dataset_name == 'train' and geo_fix is True:
        # Geographic data cleaning
        # https://www.kaggle.com/mithrillion/fixing-conflicts-in-the-geonetwork-attributes
        geo_colnames = [c for c in df.columns if re.match(r'geoNetwork', c) is not None]
        geo_colnames.remove('geoNetwork.networkDomain')

        # Overwrite the original df to reduce memory footprint, we will load it later
        df = df[geo_colnames].copy()

        # Geo categories pre-processing
        for c in geo_colnames:
            df[c] = df[c].astype('category')

        for c in geo_colnames:
            df[c] = df[c].cat.add_categories('N/A').fillna('N/A')

        #%%
        key_attributes = ['geoNetwork.city', 'geoNetwork.region', 'geoNetwork.country']
        train_geo_data_enc = hotencode_variables(df[key_attributes])

        # The target is the country, and city and region are used as training
        # Train columns are those which do not contain 'country' in the column name
        country_columns = train_geo_data_enc.columns[~train_geo_data_enc.columns.str.contains("country")]
        target = df['geoNetwork.country'].cat.codes
        clf = RandomForestClassifier(n_jobs=3)
        clf.fit(train_geo_data_enc[country_columns], target)

        #%%
        # Predict
        pred = clf.predict(train_geo_data_enc[country_columns])
        # is_anomaly = pred != target

        #%%
        # anomaly_cases = train_geo_data[is_anomaly][(train_geo_data['geoNetwork.city'] != 'N/A') | (
        #         train_geo_data['geoNetwork.region'] != 'N/A')][key_attributes]

        # Peak at the prob of the anomaly cases
        # certainty = clf.predict_proba(train_geo_data_enc[country_columns])
        # uncertain_idx = np.max(certainty, axis=1) < 0.95
        # uncertain_samples = train_geo_data[key_attributes][uncertain_idx]
        # uncertain_samples[(uncertain_samples['geoNetwork.city'] != 'N/A') & (uncertain_samples['geoNetwork.region'] != 'N/A')].groupby(key_attributes).size()
        not_na_idx = (df['geoNetwork.city'] != 'N/A') | (df['geoNetwork.region'] != 'N/A')

        target_corrected = pd.Categorical.from_codes(pred, df['geoNetwork.country'].cat.categories)

        # Set the fixed country column back to the original data
        original_df = pd.read_pickle(reduced_data_path)
        original_df.loc[not_na_idx, 'geoNetwork.country'] = pd.Series(target_corrected[not_na_idx])
        original_df['geoNetwork.country'] = original_df['geoNetwork.country'].astype('object')

        # Score improvement:
        # 1.4328 to
        # 1.4326 :(
        original_df.to_pickle("data/redu_geo_fix_{}_df.pickle".format(dataset_name))
        logger.info("Saved geofixing pkl data for: {}".format(dataset_name))
        df = original_df

    return df


def main():
    # Load initial data
    train_path = "./data/train_v2.csv"
    test_path = "./data/test_v2.csv"
    train_df = load_csv(train_path)
    train_df = preprocess_features(train_df)

    # Remove Columns with constant values
    const_cols = constant_columns(train_df)
    train_df.drop(columns=const_cols, inplace=True)
    gc.collect()
    # Dump data to pickles
    reduced_data_path = 'data/reduced_{}_df.pickle'.format("train")
    train_df.to_pickle(reduced_data_path)
    logger.info("Finished saving reduced {} pkl data".format("train"))
    del train_df

    # data_clean_and_reduce(train_df, dataset_name='train',)

    test_df = load_csv(test_path)
    test_df = preprocess_features(test_df)
    test_df.drop(columns=const_cols, inplace=True)
    reduced_data_path = 'data/reduced_{}_df.pickle'.format("test")
    test_df.to_pickle(reduced_data_path)
    logger.info("Finished saving reduced {} pkl data".format("test"))

    logger.info('Finished datacleaning')


if __name__ == "__main__":
    # %%
    main()
