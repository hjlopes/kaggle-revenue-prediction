import pickle


def data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
