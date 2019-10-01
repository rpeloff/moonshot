"""

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import logging


from moonshot.baselines import base


class MAML(base.Model):
    """TODO(rpeloff).
    """

    def __init__(self):
        super(MAML, self).__init__()



    def train(self):
        """Train and validate model on background data."""
        pass

    def _adapt_model(self, train_x, train_y):
        """Adapt model on samples from a few-shot learning task."""
        # self.memory = (train_x, train_y)

    def predict(self, test_x, k_neighbours=1):
        """Make predictions on samples from a few-shot learning task."""
        # if self.memory is None:
        #     raise IndexError("Call adapt_model(...) first!")

        # distances = self.dist_func(test_x, self.memory[0])

        # k_nn_idx = np.argsort(distances, axis=-1)[:, :k_neighbours]
        # k_nn_labels = self.memory[1][k_nn_idx]

        # predictions = np.apply_along_axis(base.majority_vote, 1, k_nn_labels)

        # return predictions
