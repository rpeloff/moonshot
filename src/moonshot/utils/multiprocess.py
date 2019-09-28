"""Utility functions for multiprocessing cpu intensive map functions.

Example
-------
>>> from moonshot.utils import multiprocess as mp
>>> shared_arg_list = ("hello world", [1, 2], np.array([2, 1]))
>>> def func(x):
...     return "{}! {} + {} * {} = {}".format(
...         mp.SHARED_ARGS[0], mp.SHARED_ARGS[1], mp.SHARED_ARGS[2],
...         x, np.asarray(mp.SHARED_ARGS[1]) + mp.SHARED_ARGS[2] * x)
>>> results = mp.multiprocess_map(func, range(4), *shared_arg_list)
>>> results
['hello world! [1, 2] + [2 1] * 0 = [1 2]',
 'hello world! [1, 2] + [2 1] * 1 = [3 3]',
 'hello world! [1, 2] + [2 1] * 2 = [5 4]',
 'hello world! [1, 2] + [2 1] * 3 = [7 5]']

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections.abc import Sequence


import multiprocessing as mp


# multiprocessing functions should access shared memory variables stored in
# global list multiprocess.SHARED_ARGS upon worker initialisation
SHARED_ARGS = None


def num_cpus():
    return mp.cpu_count()

def init_worker(*shared_args_list):
    """Initialise worker with global shared memory arguments."""
    global SHARED_ARGS
    SHARED_ARGS = shared_args_list


def multiprocess_map(func, iterable, *worker_args, n_cores=None, mode="map", **pool_kwargs):
    """Multiprocess a function with shared memory arguments.

    `mode`: one of "map", "imap" or "starmap"

    TODO(rpeloff) notes on worker_args and global SHARED_ARGS
    """
    results = []

    with mp.Manager() as manager:
        shared_args_proxy = None
        if worker_args is not None:
            shared_args_proxy = manager.list(worker_args)

        with mp.Pool(processes=n_cores, initializer=init_worker,
                     initargs=shared_args_proxy, **pool_kwargs) as pool:
            if mode == "map":
                results = pool.map(func, iterable)
            elif mode == "starmap":
                results = pool.starmap(func, iterable)
            elif mode == "imap":
                for result in pool.imap(func, iterable):
                    results.append(result)

    return results
