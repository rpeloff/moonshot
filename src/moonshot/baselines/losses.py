"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def weighted_cross_entropy_with_logits(pos_weight=1., label_smoothing=0):
    """Cross entropy computed from unnormalised log probabilities (logits)."""

    def loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred, tf.float32)

        if label_smoothing > 0:
            # label smoothing between binary classes (Szegedy et al. 2015)
            labels *= 1.0 - label_smoothing
            labels += 0.5 * label_smoothing

        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels, logits=logits, pos_weight=pos_weight),
            axis=-1)

    return loss


def focal_loss_with_logits(alpha=0.25, gamma=2.0):
    """Modulated cross entropy for imbalanced classes (Lin et al., 2017)."""

    def loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred, tf.float32)

        per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        prediction_probabilities = tf.sigmoid(logits)

        p_t = labels * prediction_probabilities
        p_t += (1 - labels) * (1 - prediction_probabilities)

        modulating_factor = 1.0
        if gamma is not None:
            modulating_factor = tf.pow(1.0 - p_t, gamma)

        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = labels*alpha + (1 - labels)*(1 - alpha)

        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent)

        return tf.reduce_mean(focal_cross_entropy_loss, axis=-1)

    return loss


# sourced from https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L40-L83
def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.

    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.maximum(
        pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared +
            tf.cast(error_mask, tf.float32) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), tf.float32))

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


# sourced from https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L122-L138
def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.

    Returns:
        masked_maximums: N-D `Tensor`.
            The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = tf.math.reduce_max(
        tf.math.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


# sourced from https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L141-L157
def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the minimum.

    Returns:
        masked_minimums: N-D `Tensor`.
            The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = tf.math.reduce_min(
        tf.math.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


# sourced from https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L160-L239
def triplet_semihard_loss(margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin
    constant (called semi-hard negative) in the mini-batch. If no such negative
    exists, uses the largest negative distance instead.

    See: https://arxiv.org/abs/1503.03832.

    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of multiclass
            integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
            be l2 normalized.
        margin: Float, margin term in the loss definition.

    Returns:
        triplet_loss: tf.float32 scalar.
    """

    def loss(labels, embeddings):
        labels = tf.cast(labels, tf.int32)
        embeddings = tf.cast(embeddings, tf.float32)

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(labels)
        assert lshape.shape == 1
        labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix.
        pdist_matrix = pairwise_distance(embeddings, squared=True)
        # Build pairwise binary adjacency matrix.
        adjacency = tf.equal(labels, tf.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = tf.logical_not(adjacency)

        batch_size = tf.size(labels)

        # Compute the mask.
        pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
        mask = tf.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.math.greater(
                pdist_matrix_tile, tf.reshape(
                    tf.transpose(pdist_matrix), [-1, 1])))
        mask_final = tf.reshape(
            tf.math.greater(
                tf.math.reduce_sum(
                    tf.cast(mask, dtype=tf.float32), 1, keepdims=True),
                0.0), [batch_size, batch_size])
        mask_final = tf.transpose(mask_final)

        adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(
            masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = tf.tile(
            masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
        semi_hard_negatives = tf.where(
            mask_final, negatives_outside, negatives_inside)

        loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = tf.cast(
            adjacency, dtype=tf.float32) - tf.linalg.diag(
                tf.ones([batch_size]))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = tf.math.reduce_sum(mask_positives)

        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(
                    tf.math.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name="triplet_semihard_loss")

        return triplet_loss

    return loss
