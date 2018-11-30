import pickle
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SUBMISSIONS_FOLDER = "submissions"


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


def factorize_variables(df, excluded=[], cat_indexers=None, logger=None):
    categorical_features = [
        _f for _f in df.columns
        if (_f not in excluded) & (df[_f].dtype in ['object'])
    ]

    if cat_indexers is None:
        cat_indexers = {}
        for f in categorical_features:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in categorical_features:
            if logger is not None:
                logger.info("Factorizing categorical: {}".format(f))
            df[f] = cat_indexers[f].get_indexer(df[f])

    return df, cat_indexers, categorical_features


def data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_logger(name):
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('train_revenue_prediction.log')
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