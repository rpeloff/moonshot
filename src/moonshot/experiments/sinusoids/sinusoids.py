"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import logging


import numpy as np


from moonshot.experiments import base


class Sinusoids(base.Experiment):
    """TODO(rpeloff).
    """

    def __init__(self):
        super(Sinusoids, self).__init__()

        self.rng = np.random.Generator(np.random.PCG64(42))

    def _sample_episode(self, K, N):

        # sample episode learning task (defined by sin phase and amplitude)
        ep_amplitude = self.rng.uniform(low=0.1, high=.5)
        ep_phase = self.rng.uniform(low=0., high=np.pi)

        # sample learning examples from episode task
        x_train = self.rng.uniform(low=-5., high=5., size=(K, 1))
        x_train = x_train.astype(np.float32)
        y_train = ep_amplitude * np.sin(x_train + ep_phase)

        # sample evaluation examples from episode task
        x_test = self.rng.uniform(low=-5., high=5., size=(N, 1))
        x_test = x_test.astype(np.float32)
        y_test = ep_amplitude * np.sin(x_test + ep_phase)

        self.curr_episode_train = x_train, y_train
        self.curr_episode_test = x_test, y_test

    @property
    def _learning_samples(self):
        return self.curr_episode_train

    @property
    def _evaluation_samples(self):
        return self.curr_episode_test
