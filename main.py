import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import data_to_pickle, load_pickle, get_logger

from datacleaning import load_csv

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

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


def generate_features(df):
    # Add date features
    df['date'] = pd.to_datetime(df['date'], unit='s')
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


def feature_importance(feat_importance, filename="distributions.png"):
    feat_importance['gain_log'] = np.log1p(feat_importance['gain'])
    mean_gain = feat_importance[['gain', 'feature']].groupby('feature').mean()
    feat_importance['mean_gain'] = feat_importance['feature'].map(mean_gain['gain'])

    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain_log', y='feature', data=feat_importance.sort_values('mean_gain', ascending=False))
    plt.savefig(filename)


def train_full(train, test, y, excluded):
    n_folds = 5
    folds = get_folds(df=train, n_splits=n_folds)
    # folds = KFold(n_splits=n_folds, shuffle=False, random_state=42)

    params = {"objective": "regression", "metric": "rmse", "max_depth": 12, "min_child_samples": 20, "reg_alpha": 0.1,
              "reg_lambda": 0.1,
              "num_leaves": 1024, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.9}

    train_features = [_f for _f in train.columns if _f not in excluded]
    logger.info("Train features: {}".format(train_features))

    importances = pd.DataFrame()
    oof_reg_preds = np.zeros(train.shape[0])
    sub_reg_preds = np.zeros(test.shape[0])

    model = lgb.LGBMRegressor(
        **params,
        n_estimators=5000,
        n_jobs=-1
    )

    for fold_, (trn_, val_) in enumerate(folds):
        logger.info("Executing fold #{}".format(fold_))
        trn_x, trn_y = train[train_features].iloc[trn_], y.iloc[trn_]
        val_x, val_y = train[train_features].iloc[val_], y.iloc[val_]

        model.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            early_stopping_rounds=100,
            verbose=500,
            eval_metric='rmse',
        )

        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = model.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        oof_reg_preds[val_] = model.predict(val_x, num_iteration=model.best_iteration_)
        oof_reg_preds[oof_reg_preds < 0] = 0
        _preds = model.predict(test[train_features], num_iteration=model.best_iteration_)
        _preds[_preds < 0] = 0
        sub_reg_preds += _preds/n_folds

    _, ax = plt.subplots(1, 1, figsize=(30, 12))
    feat_plt = lgb.plot_importance(model, ax=ax, max_num_features=50)
    feat_plt.get_figure().savefig("feature_importance.png")
    mean_squared_error(y, oof_reg_preds) ** .5
    feature_importance(importances, filename="session_feat_importance.png")

    return oof_reg_preds, sub_reg_preds


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

"""

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
"""


def train_test(train, test, y):
    params = {"objective" : "regression", "metric" : "rmse", "max_depth": 12, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "num_leaves" : 1024, "learning_rate" : 0.01, "subsample" : 0.9, "colsample_bytree" : 0.9}
    n_fold = 10
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    # Cleaning and defining parameters for LGBM
    model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)

    prediction = np.zeros(test.shape[0])

    for fold_n, (train_index, test_index) in enumerate(folds.split(train)):
        print('Fold:', fold_n)
        # print(f'Train samples: {len(train_index)}. Valid samples: {len(test_index)}')
        X_train, X_valid = train.iloc[train_index], train.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                  verbose=500, early_stopping_rounds=100)

        y_pred = model.predict(test, num_iteration=model.best_iteration_)
        prediction += y_pred
    prediction /= n_fold

    return prediction


def generate_submission_file(test_ids, prediction, filename):
    test = pd.DataFrame(test_ids)
    test['predictedLogRevenue'] = prediction
    submission = test.groupby('fullVisitorId').agg({'predictedLogRevenue': 'sum'}).reset_index()
    submission['predictedLogRevenue'] = np.log1p(submission['predictedLogRevenue'])
    submission.to_csv("submissions/{}_{}.csv".format(filename, time.strftime("%Y%m%d_%H%M%S")), index=False)


logger = get_logger(__name__)

#%%
if __name__ == "__main__":
    #%%
    # print(missing_values_table(train_df))
    # print(missing_values_table(test_df))
    # df_show_value_types(train_df)
    # df_show_value_types(test_df)
    # plot_transaction_revenue(train_df)

    from datacleaning import main as main_datacleaning
    # main_datacleaning()

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

    #%%
    t = time.time()
    train_features = [_f for _f in train_df.columns if _f not in no_use]
    train_df = train_df.sort_values('date')
    # X = train.drop([col for col in no_use if col in train.columns], axis=1)
    # y = train['totals.transactionRevenue']
    # X_test = test.drop([col for col in no_use if col in test.columns], axis=1)
    target = np.log1p(train_df['totals.transactionRevenue'])
    train_pred, test_pred = train_full(train_df, test_df, target, no_use)
    # test_pred = train_test(train_df[train_features], test_df[train_features], target)
    generate_submission_file(test_df['fullVisitorId'], test_pred, 'lgb_normal')
    # generate_submission_file(test_df['fullVisitorId'], test_pred, 'lgb_oof')
    logger.info("PredictionTime: {}".format(time.time()-t))


    """
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
    train_visit_level(train_full_df, test_full_df, target)
    """

