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
import seaborn as sns

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


import lightgbm as lgb


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


def factorize_variables(df, excluded=[], cat_indexers=None):
    categorical_features = [
        _f for _f in df.columns
        if (_f not in excluded) & (df[_f].dtype == 'object')
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

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


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


def preprocess_features(df):
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


def generate_features(df):
    # Add date features
    df['visit_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['visit_date'].dt.dayofweek
    df['sess_date_hours'] = df['visit_date'].dt.hour
    df['sess_date_dom'] = df['visit_date'].dt.day
    df['sess_date_mon'] = df['visit_date'].dt.month

    # Add next session features
    df.sort_values(['fullVisitorId', 'visit_date'], ascending=True, inplace=True)
    df['next_session_1'] = (
       df['visit_date'] - df[['fullVisitorId', 'visit_date']].groupby('fullVisitorId')[
                               'visit_date'].shift(1)
                           ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['visit_date'] - df[['fullVisitorId', 'visit_date']].groupby('fullVisitorId')[
                               'visit_date'].shift(-1)
                           ).astype(np.int64) // 1e9 // 60 // 60
    df['nb_pageviews'] = df['visit_date'].map(
        df[['visit_date', 'totals.pageviews']].groupby('visit_date')['totals.pageviews'].sum()
    )

    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']

    # Add cumulative count for user
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','visit_date', 'dummy']].groupby(['fullVisitorId','visit_date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','visit_date', 'dummy']].groupby(['fullVisitorId','visit_date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day']
    df.drop('dummy', axis=1, inplace=True)


def generate_user_aggregate_features(df):
    """
    Aggregate session data for each fullVisitorId
    :param df: DataFrame to aggregate on
    :param cat_feats: List of Categorical features
    :param sum_of_logs: if set to True, revenues are first log transformed and then summed up
    :return: aggregated fullVisitorId data over Sessions
    """
    aggs = {
        'totals.hits': ['sum', 'min', 'max', 'mean', 'median'],
        'totals.pageviews': ['sum', 'min', 'max', 'mean', 'median'],
        'totals.bounces': ['sum', 'mean', 'median'],
        'totals.newVisits': ['sum', 'mean', 'median']
    }
    if 'totals.transactionRevenue' in df.columns:
        aggs['totals.transactionRevenue'] = ['sum']

    users = df.groupby('fullVisitorId').agg(aggs)

    # Generate column names
    columns = [
        c + '_' + agg for c in aggs.keys() for agg in aggs[c]
    ]
    users.columns = columns
    logger.info("Finished aggregations. New columns: {}".format(columns))

    if 'totals.transactionRevenue' in df.columns:
        users['totals.transactionRevenue_sum'] = np.log1p(users['totals.transactionRevenue_sum'])
        y = users['totals.transactionRevenue_sum']
        users.drop(['totals.transactionRevenue_sum'], axis=1, inplace=True)
    else:
        y = None

    return users, y


def feature_importance(feat_importance, filename="distributions.png"):
    feat_importance['gain_log'] = np.log1p(feat_importance['gain'])
    mean_gain = feat_importance[['gain', 'feature']].groupby('feature').mean()
    feat_importance['mean_gain'] = feat_importance['feature'].map(mean_gain['gain'])

    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain_log', y='feature', data=feat_importance.sort_values('mean_gain', ascending=False))
    plt.savefig(filename)


def stack_features(df, train_feats=None):
    df_user_group = df.groupby('fullVisitorId').mean()
    pred_list = df[['fullVisitorId', 'predictions']].groupby('fullVisitorId') \
        .apply(lambda _df: list(_df.predictions)) \
        .apply(lambda x: {'pred_' + str(i): pred for i, pred in enumerate(x)})
    all_predictions = pd.DataFrame(list(pred_list.values), index=df_user_group.index)

    if train_feats is not None:
        for f in train_feats:
            if f not in all_predictions.columns:
                all_predictions[f] = np.nan

    feats = all_predictions.columns
    all_predictions['t_mean'] = np.log1p(all_predictions[feats].mean(axis=1))
    all_predictions['t_median'] = np.log1p(all_predictions[feats].median(axis=1))
    all_predictions['t_sum_log'] = np.log1p(all_predictions[feats]).sum(axis=1)
    all_predictions['t_sum_act'] = np.log1p(all_predictions[feats].fillna(0).sum(axis=1))
    all_predictions['t_nb_sess'] = all_predictions[feats].isnull().sum(axis=1)
    full_data = pd.concat([df_user_group, all_predictions], axis=1)
    return full_data, feats


def train_session_level(train, test, y, excluded):
    folds = get_folds(df=train, n_splits=5)

    train_features = [_f for _f in train.columns if _f not in excluded]
    logger.info("Train features: {}".format(train_features))

    importances = pd.DataFrame()
    oof_reg_preds = np.zeros(train.shape[0])
    sub_reg_preds = np.zeros(test.shape[0])
    for fold_, (trn_, val_) in enumerate(folds):
        trn_x, trn_y = train[train_features].iloc[trn_], y.iloc[trn_]
        val_x, val_y = train[train_features].iloc[val_], y.iloc[val_]

        reg = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.9,
            colsample_bytree=.9,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(val_x, np.log1p(val_y))],
            early_stopping_rounds=50,
            verbose=100,
            eval_metric='rmse'
        )
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_reg_preds[oof_reg_preds < 0] = 0
        _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_reg_preds += np.expm1(_preds) / len(folds)

    feature_importance(importances, filename="session_feat_importance.png")
    mean_squared_error(np.log1p(y), oof_reg_preds) ** .5

    return np.expm1(oof_reg_preds), sub_reg_preds


def train_visit_level(full_data, test_full_data, y):
    folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

    oof_preds = np.zeros(full_data.shape[0])
    sub_preds = np.zeros(test_full_data.shape[0])
    vis_importances = pd.DataFrame()

    for fold_, (trn_, val_) in enumerate(folds):
        trn_x, trn_y = full_data.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_data.iloc[val_], y.iloc[val_]

        reg = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.9,
            colsample_bytree=.9,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
            eval_names=['TRAIN', 'VALID'],
            early_stopping_rounds=50,
            eval_metric='rmse',
            verbose=100
        )

        imp_df = pd.DataFrame()
        imp_df['feature'] = trn_x.columns
        imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)

        oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_preds[oof_preds < 0] = 0

        # Make sure features are in the same order
        _preds = reg.predict(test_full_data[full_data.columns], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_preds += _preds / len(folds)

    # logger.info("Validation MSE: {]".format(mean_squared_error(np.log1p(y), oof_preds) ** .5))
    # test_full_df = pd.DataFrame(index=train_full_df.index)
    test_full_df['PredictedLogRevenue'] = sub_preds
    test_full_df[['PredictedLogRevenue']].to_csv("multi_lvl_lgb.csv", index=True)


def train_user_level(train, test, y):
    try:
        folds = KFold(n_splits=5, shuffle=True, random_state=1123442)

        sub_preds = np.zeros(test.shape[0])
        oof_preds = np.zeros(train.shape[0])
        oof_scores = []

        lgb_params = {
            'learning_rate': 0.03,
            'n_estimators': 2000,
            'num_leaves': 128,
            'subsample': 0.2217,
            'colsample_bytree': 0.6810,
            'min_split_gain': np.power(10.0, -4.9380),
            'reg_alpha': np.power(10.0, -3.2454),
            'reg_lambda': np.power(10.0, -4.8571),
            'min_child_weight': np.power(10.0, 2),
            'silent': True
        }

        for fold_, (trn_, val_) in enumerate(folds.split(train)):
            model = lgb.LGBMRegressor(**lgb_params)

            model.fit(
                train.iloc[trn_], y.iloc[trn_],
                eval_set=[(train.iloc[trn_], y.iloc[trn_]),
                          (train.iloc[val_], y.iloc[val_])],
                eval_metric='rmse',
                early_stopping_rounds=200,
                verbose=0
            )

            oof_preds[val_] = model.predict(train.iloc[val_])
            curr_sub_preds = model.predict(test)
            curr_sub_preds[curr_sub_preds < 0] = 0
            sub_preds += curr_sub_preds / folds.n_splits

            logger.info('Fold %d RMSE (raw output) : %.5f' % (fold_ + 1, rmse(y.iloc[val_], oof_preds[val_])))
            oof_preds[oof_preds < 0] = 0
            oof_scores.append(rmse(y.iloc[val_], oof_preds[val_]))
            logger.info('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))

        logger.info(
            'Full OOF RMSE (zero clipped): %.5f +/- %.5f' % (rmse(y, oof_preds), float(np.std(oof_scores))))

        # Stay in logs for submission
        test['PredictedLogRevenue'] = sub_preds
        test[['PredictedLogRevenue']].to_csv("simple_lgb.csv", index=True)

        logger.info('Submission data shape : {}'.format(test[['PredictedLogRevenue']].shape))

        hist, bin_edges = np.histogram(np.hstack((oof_preds, sub_preds)), bins=25)
        plt.figure(figsize=(12, 7))
        plt.title('Distributions of OOF and TEST predictions', fontsize=15, fontweight='bold')
        plt.hist(oof_preds, label='OOF predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.hist(sub_preds, label='TEST predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.legend()
        plt.savefig('distributions.png')

    except Exception as err:
        logger.exception("Unexpected error")

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
    test_df = test_df.drop(columns=const_test)

    #%%
    # Dump data to pickles
    data_to_pickle(train_df, 'data/reduced_train_df.pickle')
    data_to_pickle(test_df, 'data/reduced_test_df.pickle')


    #%%
    # print(missing_values_table(train_df))
    # print(missing_values_table(test_df))
    # df_show_value_types(train_df)
    # df_show_value_types(test_df)
    # plot_transaction_revenue(train_df)

    #%%
    # Load reduced df
    train_path = 'data/reduced_train_df.pickle'
    test_path = 'data/reduced_test_df.pickle'
    train_df = load_pickle(train_path)
    test_df = load_pickle(test_path)

    #%%
    generate_features(train_df)
    generate_features(test_df)

    excluded_feat = [
        'visit_date', 'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue',
        'visitId', 'visitStartTime', 'nb_sessions', 'max_visits'
    ]
    # train_df = preprocess_missings(train_df)
    # test_df = preprocess_missings(test_df)
    train_df, cat_indexers, cat_feat = factorize_variables(train_df, excluded=excluded_feat)
    test_df, _, _ = factorize_variables(test_df, cat_indexers=cat_indexers, excluded=excluded_feat)

    #%%
    train_pred, test_pred = train_session_level(train_df, test_df, train_df['totals.transactionRevenue'], excluded_feat)

    #$$
    train_df['predictions'] = train_pred
    test_df['predictions'] = test_pred
    train_full_df, feats = stack_features(train_df)
    test_full_df, _ = stack_features(test_df, train_feats=feats)
    target = train_full_df['totals.transactionRevenue']
    train_full_df.drop(columns=['totals.transactionRevenue'], inplace=True)

    #%%
    stacked_train_path = 'data/stacked_train_df.pickle'
    stacked_test_path = 'data/stacked_test_df.pickle'
    # Dump data to pickles
    data_to_pickle(train_full_df, stacked_train_path)
    data_to_pickle(test_full_df, stacked_test_path)

    #%%
    # Load reduced df
    train_full_df = load_pickle(stacked_train_path)
    test_full_df = load_pickle(stacked_test_path)

    #%%
    train_users, target2 = generate_user_aggregate_features(train_df)
    test_users, _ = generate_user_aggregate_features(test_df)

    #%%
    # train_user_level(train_users, test_users, target)
    trn_visitor_target = train_df[['fullVisitorId', 'totals.transactionRevenue']].groupby('fullVisitorId').sum()
    train_visit_level(train_full_df, test_full_df, target)
