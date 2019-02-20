import numpy as np
import hashlib


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]  # 0 to test_set_size
    train_indices = shuffled_indices[test_set_size:]  # test_set_size indice to last position
    return data.iloc[train_indices], data.iloc[test_indices]


''' These two last functions has the same result of Scikit-learn train_test_split function '''


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_test_set_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
