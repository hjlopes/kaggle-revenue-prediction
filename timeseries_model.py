import time
import pandas as pd
import numpy as np

import lightgbm as lgb

from main import train_lgb_user_grouped, test_models
from utils import factorize_variables, get_logger

train_path = 'data/reduced_train_df.pickle'
test_path = 'data/reduced_test_df.pickle'

# Training period set 1==> 2016/08/01 ~ 2017/1/15 (5.5 month)
# Target period set 1 ==> 2017/03/1 ~ 2017/04/30 (2 month)
# Training period set 2==> 2017/06/01 ~ 2017/11/15 (5.5 month)
# Target period set 2 ==> 2018/1/1 ~ 2018/02/30 (2 month)


def generate_features(df):
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
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


def main(logger=None):
    t = time.time()  # Processing start time
    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)

    one_hot_features = ['day', 'month', 'weekday']

    no_use = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId", "visitStartTime",
              'totals.transactionRevenue', 'trafficSource.referralPath']

    generate_features(train)
    generate_features(test)

    train, cat_indexers, cat_feat = factorize_variables(train, excluded=no_use)
    test, _, _ = factorize_variables(test, cat_indexers=cat_indexers, excluded=no_use)

    train_period_1 = train[(train['date']<=pd.datetime(2017, 1, 15)) & (train['date']>=pd.datetime(2016,8,1))]
    train_predict_1 = train[(train['date']<=pd.datetime(2017,4,30)) & (train['date']>=pd.datetime(2017,3,1))]
    train_period_2 = train[(train['date']<=pd.datetime(2017,11,15)) & (train['date']>=pd.datetime(2017,6,1))]
    train_predict_2 = train[(train['date']<=pd.datetime(2018,2,28)) & (train['date']>=pd.datetime(2018,1,1))]

    valid_period = train[(train['date']<=pd.datetime(2017,10,15)) & (train['date']>=pd.datetime(2017,5,1))]
    valid_predict_preiod = train[(train['date']<=pd.datetime(2018,1,31)) & (train['date']>=pd.datetime(2017,12,1))]

    features = [_f for _f in train.columns if _f not in no_use]

    # Period 2
    valid_target = np.log1p(valid_period['totals.transactionRevenue'])
    valid_predict_target = np.log1p(valid_predict_preiod['totals.transactionRevenue'])
    test_target = np.log1p(test['totals.transactionRevenue'])

    period2_models = train_lgb_user_grouped(valid_period, valid_target, features)
    test_models(period2_models, valid_predict_preiod, valid_predict_target, features, "lgb_timeslice")


logger = get_logger(__name__)
if __name__ == '__main__':
    main()
