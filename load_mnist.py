"""Function to load MNIST with unique IDs.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: April 2019
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import numpy as np
from tensorflow.keras.datasets import mnist


def load_id_mnist():
    # load mnist with keras datasets
    train, test = mnist.load_data(path="mnist.npz")

    # sanity check we are hashing the actual data
    assert bytes(train[0][0].reshape(-1).tolist()) == train[0][0].tostring()
    assert hashlib.md5(train[0][0].tostring()).hexdigest() == hashlib.md5(train[0][0].data).hexdigest()

    # create function to map data and label to unique hash ID
    id_hash = lambda x_data, y_label: "{}_{}".format(y_label, hashlib.md5(x_data.tostring()).hexdigest()[:10])

    # create unique MNIST IDs
    train_ids = np.asarray(list(map(id_hash, train[0], train[1])))
    test_ids = np.asarray(list(map(id_hash, test[0], test[1])))

    # return as (train, test) where each set contains (ids, data, labels)
    return ((train_ids, ) + train,
            (test_ids, ) + test)
