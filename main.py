import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns

from utils import data_to_pickle, load_pickle, get_logger

from datacleaning import load_csv

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


RANDOM_SEED = 42

gc.enable()


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




def align_train_test(train_df, test_df):
    return train_df.align(test_df, join='inner', axis=1)



### Particular dataset functions

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits, )
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


def generate_features(df):
    # Add date features
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear

    df['totals.timeOnSite'].fillna(0, inplace=True)
    df['totals.timeOnSite'] = df['totals.timeOnSite'].astype(float)

    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    df['weekofyear_unique_user_count'] = df.groupby('weekofyear')['fullVisitorId'].transform('nunique')

    # device based

    df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']

    df['visitNumber'] = np.log1p(df['visitNumber'].astype(float))
    df['totals.hits'] = np.log1p(df['totals.hits'])

    df['totals.pageviews'] = np.log1p(df['totals.pageviews'].astype(float).fillna(0))

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform(
        'sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')[
        'totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')[
        'totals.pageviews'].transform('mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['mean_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('sum')

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform(
        'sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')[
        'totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')[
        'totals.pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')

    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('count')
    df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('count')
    df['mean_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('mean')

    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')

    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')

    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()

    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']

    # Add cumulative count for user
    # df['dummy'] = 1
    # df['user_cumcnt_per_day'] = (df[['fullVisitorId','visit_date', 'dummy']].groupby(['fullVisitorId','visit_date'])['dummy'].cumcount()+1)
    # df['user_sum_per_day'] = df[['fullVisitorId','visit_date', 'dummy']].groupby(['fullVisitorId','visit_date'])['dummy'].transform(sum)
    # df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day']
    # df.drop('dummy', axis=1, inplace=True)


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


def feature_importance_plot(feat_importance, filename="distributions.png"):
    feat_importance['gain_log'] = np.log1p(feat_importance['gain'])
    mean_gain = feat_importance[['gain', 'feature']].groupby('feature').mean()
    feat_importance['mean_gain'] = feat_importance['feature'].map(mean_gain['gain'])

    plt.figure(figsize=(22, 12))
    sns.barplot(x='gain_log', y='feature', data=feat_importance.sort_values('mean_gain', ascending=False))
    plt.savefig(filename)


def train_lgb_user_grouped(train, y, feats):
    n_folds = 10
    folds = get_folds(df=train, n_splits=n_folds)
    train = train[feats]


    feature_importance = pd.DataFrame()
    oof_preds = np.zeros(train.shape[0])
    # val_preds = np.zeros(test.shape[0])

    scores = list()
    models = list()

    params = {"objective": "regression", "metric": "rmse", "max_depth": 12, "min_child_samples": 20, "reg_alpha": 0.1,
              "reg_lambda": 0.1,
              "num_leaves": 1024, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.9}
    model = lgb.LGBMRegressor(
        **params,
        n_estimators=20000,
        n_jobs=-1
    )

    for fold_, (trn_idx, val_idx) in enumerate(folds):
        logger.info("Executing fold #{}".format(fold_))
        trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]

        model.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            early_stopping_rounds=100,
            verbose=500,
            eval_metric='rmse',
        )

        imp_df = pd.DataFrame()
        imp_df['feature'] = feats
        imp_df['gain'] = model.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        feature_importance = pd.concat([feature_importance, imp_df], axis=0, sort=False)

        oof_preds[val_idx] = model.predict(val_x, num_iteration=model.best_iteration_)
        oof_preds[oof_preds < 0] = 0

        # _preds = model.predict(test, num_iteration=model.best_iteration_)
        # _preds[_preds < 0] = 0
        # val_preds += oof_preds/n_folds

        models.append(model)
        scores.append(mean_squared_error(val_y, oof_preds[val_idx]) ** .5)

    _, ax = plt.subplots(1, 1, figsize=(30, 12))
    feat_plt = lgb.plot_importance(model, ax=ax, max_num_features=50)
    feat_plt.get_figure().savefig("feature_importance.png")
    oof_score = mean_squared_error(y, oof_preds) ** .5
    feature_importance_plot(feature_importance, filename='lgb_cv_{}_st_{}_usergroup.png'.format(np.mean(scores),
                                                                                                     np.std(scores)))

    return models


def train_lgb_kfold(train, test, y):
    params = {"objective" : "regression", "metric" : "rmse", "max_depth": 12, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "num_leaves" : 1024, "learning_rate" : 0.01, "subsample" : 0.9, "colsample_bytree" : 0.9}
    n_fold = 10
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=RANDOM_SEED)

    model = lgb.LGBMRegressor(
        **params,
        n_estimators=20000,
        n_jobs=-1)

    feature_importance = pd.DataFrame()
    val_preds = np.zeros(train.shape[0])
    scores = list()
    models = list()

    for fold_n, (train_index, test_index) in enumerate(folds.split(train)):
        print('Fold:', fold_n)
        # print(f'Train samples: {len(train_index)}. Valid samples: {len(test_index)}')
        X_train, X_valid = train.iloc[train_index], train.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                  verbose=500, early_stopping_rounds=100)

        val_preds[test_index] = model.predict(X_valid, num_iteration=model.best_iteration_)
        val_preds[val_preds < 0] = 0
        scores.append(mean_squared_error(y_valid, val_preds[test_index]) ** .5)
        models.append(model)
    val_preds /= n_fold

    _, ax = plt.subplots(1, 1, figsize=(30, 12))
    feat_plt = lgb.plot_importance(model, ax=ax, max_num_features=50)
    feat_plt.get_figure().savefig("feature_importance.png")

    return models


def generate_submission_file(test_ids, prediction, filename):
    test = pd.DataFrame(test_ids)
    test['predictedLogRevenue'] = prediction
    submission = test.groupby('fullVisitorId').agg({'predictedLogRevenue': 'sum'}).reset_index()
    submission['predictedLogRevenue'] = np.log1p(submission['predictedLogRevenue'])
    submission.to_csv("submissions/{}_{}.csv".format(filename, time.strftime("%Y%m%d_%H%M%S")), index=False)


def test_models(models, test, y, features, filename):
    n_folds = len(models)
    val_preds = np.zeros(test[features].shape[0])
    scores = list()

    for model in models:
        _preds = model.predict(test[features], num_iteration=model.best_iteration_)
        _preds[_preds < 0] = 0
        val_preds += _preds/n_folds
        scores.append(mean_squared_error(y, _preds) ** .5)
        logger.info("MSE on TEST: {}".format(np.mean(scores)))

    # Generate submission files
    generate_submission_file(test['fullVisitorId'], val_preds, '{}_{:.5f}_st_{:.5f}'.format(filename, np.mean(scores),
                                                                                                     np.std(scores)))


logger = get_logger(__name__)

if __name__ == "__main__":
    #%%
    # print(missing_values_table(train_df))
    # print(missing_values_table(test_df))
    # df_show_value_types(train_df)
    # df_show_value_types(test_df)
    # plot_transaction_revenue(train_df)

    from datacleaning import main as main_datacleaning
    # main_datacleaning()
    # Score improvement with the geonetwork data cleaning. It's not worth your time
    # 1.4328 to
    # 1.4326 :(

    # Load reduced df
    # train_path = 'data/redu_geo_fix_train_df.pickle'
    train_path = 'data/reduced_train_df.pickle'
    train_df = load_pickle(train_path)
    logger.info("Loaded train with shape {}".format(train_df.shape))

    test_path = 'data/reduced_test_df.pickle'
    test_df = load_pickle(test_path)
    logger.info("Loaded test with shape {}".format(test_df.shape))

    #%%
    generate_features(train_df)
    generate_features(test_df)

    num_cols = ['visitNumber', 'totals.timeOnSite', 'totals.hits', 'totals.pageviews', 'month_unique_user_count',
                'day_unique_user_count', 'mean_hits_per_day'
                                         'sum_pageviews_per_network_domain', 'sum_hits_per_network_domain',
                'count_hits_per_network_domain', 'sum_hits_per_region',
                'sum_hits_per_day', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
                'weekday_unique_user_count',
                'sum_pageviews_per_region', 'count_pageviews_per_region', 'mean_pageviews_per_region',
                'user_pageviews_count', 'user_hits_count',
                'count_hits_per_region', 'mean_hits_per_region', 'user_pageviews_sum', 'user_hits_sum',
                'user_pageviews_sum_to_mean',
                'user_hits_sum_to_mean', 'user_pageviews_to_region', 'user_hits_to_region',
                'mean_pageviews_per_network_domain',
                'mean_hits_per_network_domain']

    no_use = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId", "visitStartTime",
              'totals.transactionRevenue', 'trafficSource.referralPath']
    cat_cols = [col for col in train_df.columns if col not in num_cols and col not in no_use]

    for col in cat_cols:
        if col != 'trafficSource.campaignCode':
            print(col)
            lbl = LabelEncoder()
            lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
            train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
            test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    no_use.append('trafficSource.campaignCode')

    t = time.time()
    features = [_f for _f in train_df.columns if _f not in no_use]
    logger.info("Train features: {}".format(features))

    train_df = train_df.sort_values('date')

    target = np.log1p(train_df['totals.transactionRevenue'])
    test_target = np.log1p(test_df['totals.transactionRevenue'])

    models_user_group = train_lgb_user_grouped(train_df, target, features)
    models_kfold = train_lgb_kfold(train_df[features], test_df[features], target)

    # Generate submission files
    test_models(models_user_group, test_df, test_target, features, "lgb_usergroup")
    test_models(models_kfold, test_df, test_target, features, "lgb_kfolds")

    logger.info("PredictionTime: {}".format(time.time()-t))