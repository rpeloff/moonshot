"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools


import tensorflow as tf


def weighted_cross_entropy_with_logits(pos_weight=1., label_smoothing=0):
    """Cross entropy computed from unnormalised log probabilities (logits)."""

    # @tf.function
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

    # @tf.function
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


def cosine_distance(x1, x2):
    """Compute the cosine distance between two matrices.

    Note: `x1` and `x2` are assumed to be unit-normalised.
    """
    x1 = tf.cast(x1, dtype=tf.float32)
    x2 = tf.cast(x2, dtype=tf.float32)

    # dot product between rows of `x_1` and rows of `x_2`
    # "ij,ij->i" := output[i] = sum_j x1[i, j] * x2[i, j]
    cos_thetas = tf.linalg.einsum("ij,ij->i", x1, x2)
    cos_distances = 1 - cos_thetas

    # deal with numerical inaccuracies setting small negatives to zero
    cos_distances = tf.maximum(cos_distances, 0.0)

    return cos_distances


def cosine_pairwise_distance(x):
    """Computes the pairwise cosine distance matrix.

    Note: `x` is assumed to be unit-normalised.
    """
    x = tf.cast(x, dtype=tf.float32)

    # dot product between rows of `x` and columns of its transpose
    cos_thetas = tf.linalg.matmul(x, x, transpose_b=True)
    pairwise_distances = 1 - cos_thetas

    # deal with numerical inaccuracies setting small negatives to zero
    pairwise_distances = tf.maximum(pairwise_distances, 0.0)

    # explicitly set the diagonals to zero
    mask_diagonals = tf.ones_like(pairwise_distances)
    mask_diagonals -= tf.linalg.diag(tf.ones([tf.shape(x)[0]]))

    pairwise_distances = tf.multiply(pairwise_distances, mask_diagonals)

    return pairwise_distances


def euclidean_distance(x1, x2, squared=False):
    """Compute the cosine distance between two matrices."""
    x1 = tf.cast(x1, dtype=tf.float32)
    x2 = tf.cast(x2, dtype=tf.float32)

    # squared euclidean distance
    distances_squared = tf.reduce_sum((x1 - x2) ** 2, axis=1)

    # deal with numerical inaccuracies setting small negatives to zero
    distances_squared = tf.maximum(distances_squared, 0.0)

    # optionally take the square root
    if squared:
        distances = distances_squared
    else:
        # get mask where distances are zero
        error_mask = tf.less_equal(distances_squared, 0.0)

        # compute stable square root
        distances = tf.math.sqrt(
            distances_squared + tf.cast(error_mask, tf.float32) * 1e-16)

        # undo conditionally adding 1e-16
        distances = tf.math.multiply(
            distances, tf.cast(tf.math.logical_not(error_mask), tf.float32))

    return distances


def euclidean_pairwise_distance(x, squared=False):
    # sourced from:
    # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L40-L83
    """Compute the pairwise euclidean distance matrix with numerical stability.

    output[i, j] = || x[i, :] - x[j, :] ||_2

    Args:
        x: 2-D Tensor of size [number of data, x dimension].
        squared: Boolean, whether or not to square the pairwise distances.

    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(x), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(x)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(x, tf.transpose(x))

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

    num_data = tf.shape(x)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    # sourced from:
    # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L122-L138
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


def masked_minimum(data, mask, dim=1):
    # sourced from:
    # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L141-L157
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


def triplet_semihard_loss(margin=1.0, metric="cosine"):
    """Triplet loss with online semi-hard negative mining using labels.

    `metric` should be one of ['cosine', 'euclidean', 'squared_euclidean'].
    """
    assert metric in ["cosine", "euclidean", "squared_euclidean"]

    @tf.function
    def loss(labels, embeddings):
        # sourced from:
        # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py#L160-L239
        """Computes the triplet loss with semi-hard negative mining.

        The loss encourages the positive distances (between a pair of embeddings
        with the same labels) to be smaller than the minimum negative distance
        among which are at least greater than the positive distance plus the
        margin constant (called semi-hard negative) in the mini-batch. If no
        such negative exists, uses the largest negative distance instead.

        See: https://arxiv.org/abs/1503.03832.

        Args:
            labels: 1-D tf.int32 `Tensor` with shape [batch_size] of multiclass
                integer labels.
            embeddings: 2-D float `Tensor` of embedding vectors. Embeddings
                should be l2 normalized.
            margin: Float, margin term in the loss definition.

        Returns:
            triplet_loss: tf.float32 scalar.
        """
        labels = tf.cast(labels, tf.int32)
        embeddings = tf.cast(embeddings, tf.float32)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(labels)
        if lshape.shape == 1:
            labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix.
        if metric == "cosine":
            pdist_matrix = cosine_pairwise_distance(embeddings)
        elif metric == "euclidean":
            pdist_matrix = euclidean_pairwise_distance(
                embeddings, squared=False)
        elif metric == "squared_euclidean":
            pdist_matrix = euclidean_pairwise_distance(
                embeddings, squared=True)
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
                    tf.math.multiply(loss_mat, mask_positives), y=0.0)),
            num_positives, name="triplet_semihard_loss")

        return triplet_loss

    return loss


def triplet_imposter_random_sample_loss(margin=1.0, metric="cosine"):
    r"""Triplet margin ranking loss with randomly sampled imposters.

    \sum_j[ max(0., d(a_j, p_j) - d(a_j, \bar{p}_j) + m) + max(0., d(a_j, p_j) - d(\bar{a}_j, p_j) + m)]

    where a_j and p_j are the j-th input anchor/positive pair, \bar{a}_j
    and \bar{p}_j are randomly sampled imposter examples, and m is the
    margin hyperparameter.

    `metric` should be one of ['cosine', 'euclidean', 'squared_euclidean'].
    """
    assert metric in ["cosine", "euclidean", "squared_euclidean"]

    @tf.function
    def loss(anchor_embeddings, positive_embeddings):
        # based on:
        # https://github.com/dharwath/DAVEnet-pytorch/blob/23a6482859dd2221350307c9bfb5627a5902f6f0/steps/util.py#L88-L116
        """Compute triplet loss for anchor/positive pairs and randomly sampled imposters.

        `anchor_embeddings` should contain embeddings each paired
        with a positive embedding in `positive_embeddings`.

        For example, anchor and positive embeddings may result from image pairs
        or image and spoken caption pairs.
        """
        anchor_shape = tf.shape(anchor_embeddings)
        positive_shape = tf.shape(positive_embeddings)

        # should have same dimensions
        assert anchor_shape.shape == positive_shape.shape

        anchor_embeddings = tf.cast(anchor_embeddings, tf.float32)
        anchor_embeddings = tf.nn.l2_normalize(anchor_embeddings, axis=1)

        positive_embeddings = tf.cast(positive_embeddings, tf.float32)
        positive_embeddings = tf.nn.l2_normalize(positive_embeddings, axis=1)

        batch_size = anchor_shape[0]

        # indices of each pair in the batch
        pair_idx = tf.range(batch_size, dtype=tf.int32)

        # uniformly sample index offsets in range [1, batch_size) for each pair
        anchor_uniform_idx = tf.random.uniform(
            [batch_size], minval=1, maxval=batch_size, dtype=tf.int32)
        positive_uniform_idx = tf.random.uniform(
            [batch_size], minval=1, maxval=batch_size, dtype=tf.int32)

        anchor_imposter_idx = pair_idx + anchor_uniform_idx
        positive_imposter_idx = pair_idx + positive_uniform_idx

        overflow_mask = tf.greater_equal(anchor_imposter_idx, batch_size)
        anchor_imposter_idx -= tf.cast(overflow_mask, tf.int32) * batch_size

        overflow_mask = tf.greater_equal(positive_imposter_idx, batch_size)
        positive_imposter_idx -= tf.cast(overflow_mask, tf.int32) * batch_size

        # compute distances between anchor/positive pairs and imposter pairs
        if metric == "cosine":
            dist = cosine_distance
        elif metric == "euclidean":
            dist = euclidean_distance
        elif metric == "squared_euclidean":
            dist = functools.partial(euclidean_distance, squared=True)

        dist_a_p = dist(anchor_embeddings, positive_embeddings)
        dist_a_imposter_p = dist(
            tf.gather(anchor_embeddings, anchor_imposter_idx), positive_embeddings)
        dist_a_p_imposter = dist(
            anchor_embeddings, tf.gather(positive_embeddings, positive_imposter_idx))

        loss = tf.math.maximum(0., dist_a_p - dist_a_imposter_p + margin)
        loss += tf.math.maximum(0., dist_a_p - dist_a_p_imposter + margin)

        return tf.math.truediv(
            tf.math.reduce_sum(loss), tf.cast(batch_size, tf.float32))

    return loss


def triplet_imposter_semi_hard_loss(margin=1.0, metric="cosine"):
    r"""Triplet margin ranking loss with semi-hard mining of imposters.

    TODO

    `metric` should be one of ['cosine', 'euclidean', 'squared_euclidean'].
    """
    assert metric in ["cosine", "euclidean", "squared_euclidean"]

    # @tf.function
    def loss(anchor_embeddings, positive_embeddings):
        """TODO
        """
        raise NotImplementedError

    return loss
