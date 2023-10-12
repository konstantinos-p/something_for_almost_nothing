import jax.numpy as jnp
# From https://github.com/SamsungLabs/pytorch-ensembles/blob/master/metrics.py#L83
# The implementation of "Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning"
# arxiv: https://arxiv.org/abs/2002.06470
import numpy as np
import jax
from jax import lax


def get_ece(preds, targets, n_bins=15, **args):
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = jnp.max(preds, 1), jnp.argmax(preds, 1)
    accuracies = (predictions == targets)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = jnp.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = jnp.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = jnp.mean(accuracies[in_bin])
            avg_confidence_in_bin = jnp.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += jnp.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece


def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape

    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = jnp.sort(cur_class_conf)

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins

        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind: bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind: bin_end_ind]
            avg_confidence_in_bin = jnp.mean(bin_conf)
            avg_accuracy_in_bin = jnp.mean(bin_acc)
            delta = jnp.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            #             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)

    return res


def get_brier(preds, targets, **args):
    """
    An implementation of the Brier score.
    Parameters
    ----------
    preds: jax.tensor
        (n, num_classes) tensor containing the categorical distribution for the predicted label for each input
        signal
    targets:  jax.tensor
        (n,) tensor containing the target label for each input signal
    args:
        auxiliary arguments
    Returns
    -------
        : jax.tensor
        (n,) tensor containing the Brier score for each input signal

    """

    one_hot_targets = jnp.zeros(preds.shape)
    one_hot_targets = one_hot_targets.at[jnp.arange(len(targets)), targets].set(1.0)
    return jnp.mean(jnp.sum((preds - one_hot_targets) ** 2, axis=1))


def get_brier_decomposition(preds, targets):
    """
    An implementation of the Brier score decomposition into uncertainty, resolution, and reliability.

      [Proper scoring rules][1] measure the quality of probabilistic predictions;
      any proper scoring rule admits a [unique decomposition][2] as
      `Score = Uncertainty - Resolution + Reliability`, where:

      * `Uncertainty`, is a generalized entropy of the average predictive
        distribution; it can both be positive or negative.
      * `Resolution`, is a generalized variance of individual predictive
        distributions; it is always non-negative.  Difference in predictions reveal
        information, that is why a larger resolution improves the predictive score.
      * `Reliability`, a measure of calibration of predictions against the true
        frequency of events.  It is always non-negative and a lower value here
        indicates better calibration.

    #### References
    [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
    [2]: Jochen Broecker.  Reliability, sufficiency, and the decomposition of
       proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456


    Parameters
    ----------
    preds: jax.tensor
        (n, num_classes) tensor containing the categorical distribution for the predicted label for each input
        signal
    targets:  jax.tensor
        (n,) tensor containing the target label for each input signal
    args:
        auxiliarry arguments
    Returns
    -------
    uncertainty: jax.tensor
        (n,) tensor containing the uncertainty part of the Brier score for each input signal
    resolution: jax.tensor
        (n,) tensor containing the resolution part of the Brier score for each input signal
    reliability: jax.tensor
        (n,) tensor containing the reliability part of the Brier score for each input signal

    """
    try:
        pred_class = jnp.argmax(preds, -1)
        confusion_matrix = get_confusion_matrix(pred_class, targets) # note how in this implementation the true labels are on the y (horizontal) axis
        dist_weights = jnp.sum(confusion_matrix, axis=1)
        dist_weights /= jnp.sum(dist_weights)
        pbar = jnp.sum(confusion_matrix, axis=0)
        pbar /= jnp.sum(pbar)

        # dist_mean[k,:] contains the empirical distribution for the set M_k
        # Some outcomes may not realize, corresponding to dist_weights[k] = 0
        dist_mean = confusion_matrix / jnp.expand_dims(jnp.sum(confusion_matrix, axis=1) + 1.0e-7, axis=1)

        # Uncertainty: quadratic entropy of the average label distribution
        uncertainty = jnp.sum(pbar-jnp.square(pbar))

        # Resolution: expected quadratic divergence of predictive to mean
        resolution = jnp.square(dist_mean - jnp.expand_dims(pbar, 1))
        resolution = jnp.sum(dist_weights * jnp.sum(resolution, axis=1))

        # Reliability: expected quadratic divergence of predictive to true
        prob_true = dist_mean[pred_class, :]
        reliability = jnp.sum(jnp.square(preds - prob_true), axis=1)
        reliability = jnp.mean(reliability)
    except:
        uncertainty = 0
        resolution = 0
        reliability = 0

    return uncertainty, resolution, reliability


def get_confusion_matrix(labels, pred_class):
    """
    Computes the confusion matrix.

    Parameters
    ----------
    pred_class: jax.tensor
        (n,) tensor containing the predicted label for each input signal
    labels: jax.tensor
        (n,) tensor containing the target label for each input signal
    num_classes: int
        the number of classes in the classification problem

    Returns
    -------

    """

    num_classes = jnp.max(labels)+1

    cm, _ = jax.lax.scan(
        lambda carry, pair: (carry.at[pair].add(1), None),
        jnp.zeros((num_classes, num_classes), dtype=jnp.uint32),
        (labels, pred_class)
        )

    return cm


def mutual_information(X, Y):
    """

    Parameters
    ----------
    X: numpy.array
        The values of the categorical variable.
    Y: numpy.array
        The values of the other categorical variable.

    Returns
    -------
    mi: float
        The mutual information between the two random variables.
    """
    # Calculate joint and marginal probabilities
    joint_probs = np.histogram2d(X, Y, bins=(len(np.unique(X)), len(np.unique(Y))))[0]
    joint_probs /= np.sum(joint_probs)

    marginal_X = np.sum(joint_probs, axis=1)
    marginal_Y = np.sum(joint_probs, axis=0)

    # Calculate mutual information
    mi = 0
    for i in range(len(np.unique(X))):
        for j in range(len(np.unique(Y))):
            if joint_probs[i, j] > 0:
                mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (marginal_X[i] * marginal_Y[j]))

    return mi
