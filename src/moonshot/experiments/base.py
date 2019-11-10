"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


from absl import logging


import numpy as np


class Experiment(abc.ABC):
    """TODO(rpeloff).

    - generate episodes from one-shot evaluation data (or background data)
    - emit episode support set for one-shot learning
    - emit episode evaluation task
    - receive agent predictions/actions
    - store and emit results
    - repeat for next episode
    - reset experiment
    """

    def __init__(self, num_episodes=None, seed=42):
        self.num_episodes = num_episodes  # TODO: used?
        self.seed = seed

        self.curr_episode_train = None
        self.curr_episode_test = None

        # random number generator for this experiment
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

    def sample_episode(self, *args, **kwargs):
        # TODO: handle episode update logic and check for final episode?
        return self._sample_episode(*args, **kwargs)

    def batch_generator(self, batch_size, *args, **kwargs):
        for _ in range(batch_size):
            self.sample_episode(*args, **kwargs)
            yield (self.learning_samples, self.evaluation_samples)

    def batch_episodes(self, batch_size, *args, **kwargs):
        # train across tasks e.g.
        # for task in zip(*train_b, *test_b): train_x, train_y, test_x, test_y = task

        # TODO remove if sure replacing with below
        # train_batch, test_batch = [], []
        # for _ in range(batch_size):

        #     self.sample_episode(*args, **kwargs)

        #     train = self.learning_samples
        #     test = self.evaluation_samples

        #     train_batch.append(train)
        #     test_batch.append(test)

        # train_batch = tuple(np.stack(x) for x in zip(*train_batch))
        # test_batch = tuple(np.stack(x) for x in zip(*test_batch))

        # return train_batch, test_batch

        # train across tasks e.g. `for task in batch: train, test = task`
        return tuple(self.batch_generator(batch_size, *args, **kwargs))

    @abc.abstractproperty
    def data(self):
        """TODO"""

    @property
    def learning_samples(self):
        """Return the task learning samples for the current episode."""
        if self.curr_episode_train is None:
            raise IndexError("Call sample_episode(*args, **kwargs) first!")
        return self._learning_samples

    @property
    def evaluation_samples(self):
        """Return the task evaluation samples for the current episode."""
        if self.curr_episode_train is None:
            raise IndexError("Call sample_episode(*args, **kwargs) first!")
        return self._evaluation_samples

    @abc.abstractmethod
    def _sample_episode(self, *args, **kwargs):
        """TODO"""

    @abc.abstractproperty
    def _learning_samples(self):
        """TODO"""

    @abc.abstractproperty
    def _evaluation_samples(self):
        """TODO"""

    def evaluate(self, task, action):
        """TODO: Store & evaluate agents actions for an evaluation task?"""

    def reset(self):
        """TODO: Reset experiment"""
        self.curr_episode_train = None
        self.curr_episode_test = None

        self.rng = np.random.Generator(np.random.PCG64(self.seed))
